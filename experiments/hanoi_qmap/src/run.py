"""Hanoi-only Q training: exhaustive peg-orbit data, distance and candidate ranking."""
import argparse
import json
from pathlib import Path
import shutil
import time

import numpy as np
import torch
from torch.nn import functional as F

from grounded_llm.tasks.hanoi.data import build
from grounded_llm.tasks.hanoi.evaluate import decisions, encode, evaluate, pair_metrics, ranking_loss, task_indices, tower_policy
from grounded_llm.tasks.hanoi.model import MODEL_TYPE, QMap, soft_rank_loss
from grounded_llm.core.qmap.sampling import CoverageSampler, StratifiedEpochSampler
from grounded_llm.tasks.hanoi.environment import Hanoi
from grounded_llm.tasks.hanoi.tower_data import TowerData

def make_environment(task, oracle_path=None):
    if not task.startswith("hanoi") or not task[5:].isdigit():
        raise ValueError("Hanoi training accepts only Hanoi tasks")
    return Hanoi(int(task[5:]))


def train(env, config, data, out, device, source_commit, checkpoint=None):
    if not env.name.startswith("hanoi") or config["task"] != env.name:
        raise ValueError("Hanoi training requires a matching Hanoi contract")
    torch.manual_seed(config["seed"])
    rng = np.random.default_rng(config["seed"] + 1)
    options = config["training"]
    if any(key in options for key in ("distance_supervision", "progress_margin_weight", "progress_margin")):
        raise ValueError("Archived loss experiments require their recorded source revision")
    torch.set_num_threads(options["cpu_threads"])
    model = QMap(env.feature_dim, **config["model"], cap=env.diameter).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=options["learning_rate"])
    parent = None
    if checkpoint is not None:
        saved = torch.load(checkpoint, map_location=device, weights_only=False)
        if (saved.get("model_type") != MODEL_TYPE or saved["task"] != env.name or
                saved["actions"] != list(env.actions) or
                {"metric": "l1", **saved["model_config"]} != {"metric": "l1", **config["model"]}):
            raise ValueError("Continuation requires a matching pure-Q checkpoint")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        for group in optimizer.param_groups:
            group["lr"] = options["learning_rate"]
        parent = {"checkpoint": str(checkpoint), "source_commit": saved["source_commit"], "step": saved["step"]}
    scheduler = (torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, options["steps"],
                 eta_min=options["minimum_learning_rate"]) if "minimum_learning_rate" in options else None)
    pairs = data["pairs_train"]
    tasks, children = data["tasks_train"], data["children_train"]
    pair_sampler = rank_sampler = None
    tower_fraction = config.get("sampling", {}).get("tower_fraction", 0.)
    if tower_fraction and (not 0 < tower_fraction < 1 or not config["data"].get("require_all_tower_relations", False)):
        raise ValueError("Tower replay requires complete tower relations and a mixed sampling fraction")
    tower = TowerData(env, data, rng) if tower_fraction else None
    if "sampling" in config:
        sampling = config["sampling"]
        parent_distance = data.get("parent_distances_train")
        if parent_distance is None:
            parent_distance = env.distance(tasks[:, 0], tasks[:, 1])
        if sampling.get("kind") == "stratified_epochs":
            pair_sampler = StratifiedEpochSampler(pairs[:, 2], rng)
            rank_sampler = StratifiedEpochSampler(parent_distance, rng)
            for sampler, batch in ((pair_sampler, options["batch_size"]), (rank_sampler, options["decision_batch_size"])):
                if sampler.steps_for_passes(batch, sampling["minimum_full_passes"]) > options["steps"]:
                    raise ValueError("Training budget cannot cover the contracted layer epochs")
        else:
            pair_sampler = CoverageSampler(pairs[:, 2], rng, data.get("broad_pair_count", len(pairs)),
                                           sampling["broad_fraction"])
            rank_sampler = CoverageSampler(parent_distance, rng, data.get("broad_task_count", len(tasks)),
                                           sampling["broad_fraction"],
                                           int(data["context_task_start"]) if "context_task_start" in data else None,
                                           sampling["context_fraction"])
            for sampler, batch in ((pair_sampler, options["batch_size"]), (rank_sampler, options["decision_batch_size"])):
                if options["steps"] * round(round(batch * (1 - tower_fraction)) * sampling["broad_fraction"]) / sampler.broad_count < sampling["minimum_broad_passes"]:
                    raise ValueError("Training budget cannot cover the contracted breadth")

    def coverage():
        if isinstance(pair_sampler, StratifiedEpochSampler):
            return {"pair_min_layer_passes": pair_sampler.passes, "decision_min_layer_passes": rank_sampler.passes,
                    "pair_layers": pair_sampler.coverage(), "decision_layers": rank_sampler.coverage()}
        result = ({"pair_broad_passes": pair_sampler.passes, "decision_broad_passes": rank_sampler.passes}
                  if pair_sampler is not None else {})
        if tower is not None:
            result.update(tower.report())
        return result

    # Features only: validation/test labels never enter this training store.
    ids = np.unique(np.concatenate((pairs[:, :2].ravel(), tasks.ravel(), children[children >= 0])))
    lookup = np.full(env.size, -1, dtype=np.int64)
    lookup[ids] = np.arange(len(ids))
    features = torch.as_tensor(env.features(ids), device=device)

    def q(states):
        indices = lookup[states]
        if np.any(indices < 0):
            raise ValueError("State outside the training feature store")
        return model.encode(features[torch.as_tensor(indices, device=device)])

    val_idx = task_indices(env, data, "validation", config["evaluation"]["decision_count"], config)
    probe_pairs = probe_tasks = None
    if options.get("monitor_training", False):
        probe_rng = np.random.default_rng(config["seed"] + 51)
        groups = [np.flatnonzero(pairs[:, 2] == d) for d in np.unique(pairs[:, 2])]
        probe_pairs = pairs[np.concatenate([probe_rng.choice(g, min(len(g), 6000 // len(groups)), replace=False)
                                            for g in groups])]
        probe_tasks = tasks[task_indices(env, data, "train", 1000, config)]

    def validate():
        model.eval()
        metrics = pair_metrics(model, env, data["pairs_validation"], device)
        val_tasks = data["tasks_validation"][val_idx]
        rank = ranking_loss(model, env, val_tasks, data["child_distances_validation"][val_idx],
                            device, options["rank_temperature"])
        actions = decisions(model, env, val_tasks, device)
        score = metrics["huber"] + options["rank_weight"] * rank
        return score, metrics, actions["q"], rank

    def training_monitor():
        pm = pair_metrics(model, env, probe_pairs, device)
        candidates = env.successors(probe_tasks[:, 0])
        truth = env.distance(np.maximum(candidates, 0), probe_tasks[:, 1, None])
        rank = ranking_loss(model, env, probe_tasks, truth, device, options["rank_temperature"])
        target = torch.as_tensor(truth, dtype=torch.float32, device=device)
        valid = torch.as_tensor(candidates >= 0, device=device)
        logits = (-target / options["rank_temperature"]).masked_fill(~valid, -torch.inf)
        entropy = float(-(torch.softmax(logits, -1) *
                          torch.log_softmax(logits, -1).masked_fill(~valid, 0)).sum(-1).mean())
        result = {"pairs": pm, "soft_rank_loss": rank, "rank_target_entropy": entropy, "rank_kl": rank - entropy,
                  "objective": pm["huber"] + options["rank_weight"] * rank,
                  "decisions": decisions(model, env, probe_tasks, device)["q"]}
        return result

    def save(step, destination):
        torch.save({"model_type": MODEL_TYPE, "task": env.name, "cap": env.diameter,
                    "actions": list(env.actions), "model": model.state_dict(),
                    "model_config": config["model"], "config": config, "source_commit": source_commit,
                    "step": step, "parent": parent, "coverage": coverage(), "optimizer": optimizer.state_dict()}, destination)

    started = time.monotonic()
    initial_score, initial_pairs, initial_decisions, _ = validate()
    tower_selection = config.get("selection", {}).get("kind") == "tower_restoration_with_relation_guard"
    initial_policy = tower_policy(model, env, device) if tower_selection else None
    initial_training = training_monitor() if probe_pairs is not None else None
    initial_test = evaluate(model, env, data, config, device) if options.get("evaluate_initial_test", True) else None
    best, best_step, stale = float("inf"), 0, 0
    if options.get("retain_initial", False):
        best = -initial_policy["within_budget"] if tower_selection else initial_score
        save(0, out / "best.pt")
    with (out / "progress.jsonl").open("w") as log:
        for step in range(1, options["steps"] + 1):
            model.train()
            tower_count = round(options["batch_size"] * tower_fraction)
            broad_count = options["batch_size"] - tower_count
            sampled_ids = (pair_sampler.draw(broad_count) if pair_sampler is not None
                           else rng.integers(len(pairs), size=broad_count))
            if tower is not None:
                sampled_ids = np.r_[sampled_ids, tower.draw(tower_count, False)]
            sampled = pairs[sampled_ids]
            pair_loss = F.smooth_l1_loss(model.distance(q(sampled[:, 0]), q(sampled[:, 1])),
                                         torch.as_tensor(sampled[:, 2], dtype=torch.float32, device=device))
            tower_count = round(options["decision_batch_size"] * tower_fraction)
            broad_count = options["decision_batch_size"] - tower_count
            idx = (rank_sampler.draw(broad_count) if rank_sampler is not None
                   else rng.integers(len(tasks), size=broad_count))
            if tower is not None:
                idx = np.r_[idx, tower.draw(tower_count, True)]
            candidate = children[idx]
            valid = torch.as_tensor(candidate >= 0, device=device)
            # Use an actual training parent for padding, never an arbitrary heldout ID.
            safe = np.where(candidate >= 0, candidate, tasks[idx, 0, None])
            candidate_q = q(safe)
            goal_q = q(tasks[idx, 1])
            scores = model.distance(candidate_q, goal_q[:, None])
            truth = torch.as_tensor(data["child_distances_train"][idx], device=device)
            rank_loss = soft_rank_loss(scores, truth, valid, options["rank_temperature"])
            loss = pair_loss + options["rank_weight"] * rank_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            if step == 1 or step % options["eval_every"] == 0 or step == options["steps"]:
                score, metrics, actions, val_rank = validate()
                covered = coverage()
                eligible = (step >= options.get("min_steps", 0) and
                            (not covered or isinstance(pair_sampler, StratifiedEpochSampler) or
                             min(covered.values()) >= config["sampling"]["minimum_broad_passes"]))
                policy = tower_policy(model, env, device) if tower_selection else None
                selection_score = score
                if tower_selection:
                    eligible &= metrics["mae"] <= initial_pairs["mae"] + config["selection"]["maximum_validation_mae_increase"]
                    selection_score = -policy["within_budget"]
                if eligible and selection_score < best - options["min_improvement"]:
                    best, best_step, stale = selection_score, step, 0
                    save(step, out / "best.pt")
                elif eligible:
                    stale += 1
                record = {"step": step, "seconds": round(time.monotonic() - started, 2),
                          "distance_loss": float(pair_loss.detach()), "rank_loss": float(rank_loss.detach()),
                          "validation_mae": metrics["mae"], "validation_huber": metrics["huber"],
                          "validation_soft_rank_loss": val_rank,
                          "validation_shortest_action_rate": actions["shortest_action_rate"],
                          "validation_objective": score, "selection_eligible": eligible,
                          "learning_rate": optimizer.param_groups[0]["lr"], "best_step": best_step, **covered}
                if tower is not None:
                    record["tower_data"] = tower.report()
                if policy is not None:
                    record["all_state_towers"] = policy
                if probe_pairs is not None:
                    record["training_monitor"] = training_monitor()
                    val_tasks = data["tasks_validation"][val_idx]
                    val_depth = env.distance(val_tasks[:, 0], val_tasks[:, 1])
                    record["validation_decisions_by_depth"] = {
                        str(d): decisions(model, env, val_tasks[val_depth == d], device)["q"] for d in np.unique(val_depth)}
                log.write(json.dumps(record) + "\n")
                log.flush()
                print(json.dumps(record), flush=True)
                if eligible and options["patience"] is not None and stale >= options["patience"]:
                    break
    last_coverage = coverage()
    if options.get("save_last", False):
        save(step, out / "last.pt")
    saved = torch.load(out / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(saved["model"])
    result = {"model_type": MODEL_TYPE, "task": env.name, "source_commit": source_commit, "best_step": best_step, "last_step": step,
              "training_seconds": round(time.monotonic() - started, 2), "device": str(device),
              "best_model_training_coverage": saved.get("coverage", {}),
              "last_model_training_coverage": last_coverage, "parent": parent,
              "stop_reason": "step_budget_exhausted" if step == options["steps"] else "validation_patience",
              "torch_version": torch.__version__, "numpy_version": np.__version__,
              "selection": ("all-state tower restoration with fixed validation distance guard" if tower_selection else
                            "validation distance Huber + rank_weight * full soft rank cross entropy"),
              "initial_tower_policy": initial_policy,
              "initial_validation_objective": initial_score, "initial_validation_pairs": initial_pairs,
              "initial_validation_decisions": initial_decisions, "initial": initial_test,
              "initial_training_monitor": initial_training,
              "final": evaluate(model, env, data, config, device)}
    # Compact interface examples, not an oracle table disguised as learned Q.
    export_ids = np.arange(env.size) if env.size < 1000 else np.unique(data["pairs_test"][:2048, :2])
    np.savez_compressed(out / "q_vectors.npz", states=export_ids, q=encode(model, env, export_ids, device).cpu().numpy())
    (out / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--prepared-run", type=Path, help="Reuse this frozen data/oracle instead of rebuilding")
    parser.add_argument("--checkpoint", type=Path, help="Continue matching pure Q and Adam; start a new sampling stream")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    if args.prepared_run is not None:
        source = args.prepared_run.resolve()
        prepared_config = json.loads((source / "config.json").read_text())
        if any(config[key] != prepared_config[key] for key in ("task", "rules", "data", "augmentation")):
            raise ValueError("Prepared data does not match this contract")
        for name in ("data.npz", "oracle_depth.npy"):
            if (source / name).exists():
                (args.out / name).symlink_to(source / name)
        shutil.copyfile(source / "data_summary.json", args.out / "data_summary.json")
        env = make_environment(config["task"], args.out / "oracle_depth.npy")
        data = dict(np.load(args.out / "data.npz", allow_pickle=False))
        summary = {"reused_prepared_run": str(source)}
    else:
        env = make_environment(config["task"], args.out / "oracle_depth.npy")
        data, summary = build(env, config, args.out)
    print(json.dumps({"data": summary}), flush=True)
    result = train(env, config, data, args.out, args.device, args.source_commit, args.checkpoint)
    print(json.dumps({"complete": str(args.out), "best_step": result["best_step"],
                      "test_rollout": result["final"]["test"]["rollout"]}), flush=True)


if __name__ == "__main__":
    main()
