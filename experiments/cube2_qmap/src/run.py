"""Cube-only Q training: distance relations, full candidates and optional local graphs."""
import argparse
import json
from pathlib import Path
import shutil
import time

import numpy as np
import torch
from torch.nn import functional as F

from grounded_llm.tasks.cube2.relation_data import build
from grounded_llm.tasks.cube2.evaluate import decisions, encode, evaluate, pair_metrics, ranking_loss, task_indices
from grounded_llm.tasks.cube2.model import MODEL_TYPE, QMap, soft_rank_loss
from grounded_llm.core.qmap.sampling import CoverageSampler, StratifiedEpochSampler
from grounded_llm.tasks.cube2.environment import Cube2

def make_environment(task, oracle_path=None):
    if task not in ("cube2", "cube2_qtm"):
        raise ValueError("Cube training accepts only cube tasks")
    return Cube2(oracle_path, metric="QTM_URF_6" if task == "cube2_qtm" else "HTM_URF_9")


def train(env, config, data, out, device, source_commit, checkpoint=None):
    if not env.name.startswith("cube2") or config["task"] != env.name:
        raise ValueError("Cube training requires a matching cube contract")
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
    online = None
    pair_sampler = rank_sampler = None
    if config["data"].get("builder") == "local_graph":
        from grounded_llm.tasks.cube2.local_graph import LocalGraphSampler
        online = LocalGraphSampler(env, data, config, rng)
        if online.sampler.steps_for_passes(options["graph_batch_size"], config["sampling"]["minimum_full_passes"]) > options["steps"]:
            raise ValueError("Training budget cannot traverse the contracted relative layers")
    elif "sampling" in config:
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
                if options["steps"] * round(batch * sampling["broad_fraction"]) / sampler.broad_count < sampling["minimum_broad_passes"]:
                    raise ValueError("Training budget cannot cover the contracted breadth")

    def coverage():
        if online is not None:
            return online.coverage()
        if isinstance(pair_sampler, StratifiedEpochSampler):
            return {"pair_min_layer_passes": pair_sampler.passes, "decision_min_layer_passes": rank_sampler.passes,
                    "pair_layers": pair_sampler.coverage(), "decision_layers": rank_sampler.coverage()}
        return ({"pair_broad_passes": pair_sampler.passes, "decision_broad_passes": rank_sampler.passes}
                if pair_sampler is not None else {})

    # Features only: validation/test labels never enter this training store.
    if online is None:
        ids = np.unique(np.concatenate((pairs[:, :2].ravel(), tasks.ravel(), children[children >= 0])))
        lookup = np.full(env.size, -1, dtype=np.int64)
        lookup[ids] = np.arange(len(ids))
        features = torch.as_tensor(env.features(ids), device=device)
    else:
        # Fixed rule tables construct the same 192 one-hot inputs directly on
        # the device. Fresh poses need no full-state feature cache or new head.
        permutations = torch.as_tensor(env.permutations, device=device)
        twists = torch.as_tensor(env.twists, device=device)

    def online_encode(states):
        ids = torch.as_tensor(states, device=device, dtype=torch.long)
        joint = permutations[ids // 729] * 3 + twists[ids % 729]
        return model.encode(F.one_hot(joint, 24).reshape(-1, env.feature_dim).float())

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
    initial_training = training_monitor() if probe_pairs is not None else None
    initial_test = evaluate(model, env, data, config, device) if options.get("evaluate_initial_test", True) else None
    best, best_step, stale = float("inf"), 0, 0
    if options.get("retain_initial", False):
        best = initial_score
        save(0, out / "best.pt")
    with (out / "progress.jsonl").open("w") as log:
        for step in range(1, options["steps"] + 1):
            model.train()
            if online is not None:
                batch = online.draw()
                vectors = online_encode(batch["states"])
                pair_indices = torch.as_tensor(batch["pairs"], device=device)
                pair_prediction = model.distance(vectors[pair_indices[:, 0]], vectors[pair_indices[:, 1]])
                pair_target = torch.as_tensor(batch["pair_distances"], dtype=torch.float32, device=device)
                pair_loss = F.smooth_l1_loss(pair_prediction, pair_target)
                candidate_q = vectors[torch.as_tensor(batch["children"], device=device)]
                goal_q = vectors[torch.as_tensor(batch["goals"], device=device)]
                scores = model.distance(candidate_q, goal_q[:, None])
                truth = torch.as_tensor(batch["child_distances"], device=device)
                valid = torch.ones_like(truth, dtype=torch.bool)
            else:
                sampled_ids = (pair_sampler.draw(options["batch_size"]) if pair_sampler is not None
                               else rng.integers(len(pairs), size=options["batch_size"]))
                sampled = pairs[sampled_ids]
                pair_loss = F.smooth_l1_loss(model.distance(q(sampled[:, 0]), q(sampled[:, 1])),
                                             torch.as_tensor(sampled[:, 2], dtype=torch.float32, device=device))
                idx = (rank_sampler.draw(options["decision_batch_size"]) if rank_sampler is not None
                       else rng.integers(len(tasks), size=options["decision_batch_size"]))
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
                            (not covered or online is not None or isinstance(pair_sampler, StratifiedEpochSampler) or
                             min(covered.values()) >= config["sampling"]["minimum_broad_passes"]))
                if eligible and score < best - options["min_improvement"]:
                    best, best_step, stale = score, step, 0
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
              "selection": "validation distance Huber + rank_weight * full soft rank cross entropy",
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
        if env.name.startswith("cube2"):
            env.build_oracle(args.out / "oracle_depth.npy")
        data, summary = build(env, config, args.out)
    print(json.dumps({"data": summary}), flush=True)
    result = train(env, config, data, args.out, args.device, args.source_commit, args.checkpoint)
    print(json.dumps({"complete": str(args.out), "best_step": result["best_step"],
                      "test_rollout": result["final"]["test"]["rollout"]}), flush=True)


if __name__ == "__main__":
    main()
