"""Puzzle8-only complete-state relation training with the established Qmap objective."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from torch.nn import functional as F
from grounded_llm.qmap_sampling import CoverageSampler, task_indices
from grounded_llm.qmap_metrics import pair_metrics, decisions, ranking_loss
from .environment import Puzzle8
from .data import build
from .model import QMap, MODEL_TYPE, soft_rank_loss
from .evaluate import evaluate
from .policy_audit import audit
from .data_sampling import DataMix
from .branch_data import BranchData
from .graph_data import GraphData


def train(env, config, data, out, device, source_commit, checkpoint=None):
    torch.manual_seed(config["seed"])
    options = config["training"]
    torch.set_num_threads(options["cpu_threads"])
    rng = np.random.default_rng(config["seed"] + 1)
    model = QMap(**config["model"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=options["learning_rate"])
    parent = None
    if checkpoint is not None:
        saved = torch.load(checkpoint, map_location=device, weights_only=False)
        if (saved.get("model_type") != MODEL_TYPE or saved["actions"] != list(env.actions)
                or saved["config"]["model"] != config["model"] or "repair_config" in saved):
            raise ValueError("Continuation must use a matching original-objective Q checkpoint")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        for group in optimizer.param_groups:
            group["lr"] = options["learning_rate"]
        parent = {"checkpoint": str(checkpoint), "source_commit": saved["source_commit"], "step": saved["step"]}
    pairs, tasks = data["pairs_train"], data["tasks_train"]
    ps = CoverageSampler(pairs[:, 2], rng, len(pairs), .5)
    rs = CoverageSampler(data["parent_distances_train"], rng, len(tasks), .5)
    for sampler, batch in ((ps, options["batch_size"]), (rs, options["decision_batch_size"])):
        if checkpoint is None and options["steps"] * round(batch * .5) < sampler.broad_count:
            raise ValueError("Budget must cover every training relation and decision")
    mixture = (GraphData(env, data, config["graph_data"], rng, out) if "graph_data" in config else
               BranchData(env, data, config["branch_data"], rng, out) if "branch_data" in config else
               DataMix(env, data, config["data_sampling"], rng) if "data_sampling" in config else None)
    features = torch.as_tensor(env.features(np.arange(env.size)), device=device)
    def q(ids):
        return model.encode(features[torch.as_tensor(ids, device=device, dtype=torch.long)])
    idx = task_indices(env, data, "validation", config["evaluation"]["decision_count"], config)
    vt = data["tasks_validation"][idx]
    best, best_step = float("inf"), 0
    started = time.monotonic()
    def save(step, path):
        torch.save({"model_type": MODEL_TYPE, "task": env.name, "actions": list(env.actions),
                    "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "config": config, "source_commit": source_commit,
                    "step": (parent["step"] if parent else 0) + step, "continuation_step": step, "parent": parent,
                    "coverage": (mixture.report() if mixture is not None else
                                 {"pair_passes": ps.passes, "decision_passes": rs.passes})}, path)
    initial_policy = initial_validation_mae = None
    reference = None
    if parent is not None:
        initial_policy, state = audit(model, env, device, cohort=data["tasks_restoration"][:, 0])
        reference = state["within"].copy()
        if mixture is not None:
            mixture.refresh(state["hard_parents"])
        initial_validation_mae = pair_metrics(model, env, data["pairs_validation"], device)["mae"]
        best = -initial_policy["within_budget"]
        save(0, out / "best.pt")
        print(json.dumps({"initial_policy": initial_policy, "initial_validation_mae": initial_validation_mae}), flush=True)
    with (out / "progress.jsonl").open("w") as log:
        for step in range(1, options["steps"] + 1):
            model.train()
            rows = (pairs[ps.draw(options["batch_size"])] if mixture is None
                    else mixture.pairs(options["batch_size"]))
            pair_loss = F.smooth_l1_loss(model.distance(q(rows[:, 0]), q(rows[:, 1])),
                                        torch.as_tensor(rows[:, 2], device=device, dtype=torch.float32))
            if mixture is None:
                sampled = rs.draw(options["decision_batch_size"])
                batch_tasks, children, labels = tasks[sampled], data["children_train"][sampled], data["child_distances_train"][sampled]
            else:
                batch_tasks, children, labels = mixture.tasks(options["decision_batch_size"])
            valid = torch.as_tensor(children >= 0, device=device)
            scores = model.distance(q(np.maximum(children, 0)), q(batch_tasks[:, 1])[:, None])
            truth = torch.as_tensor(labels, device=device)
            rank_loss = soft_rank_loss(scores, truth, valid, options["rank_temperature"])
            loss = pair_loss + options["rank_weight"] * rank_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step()
            if step == 1 or step % options["eval_every"] == 0 or step == options["steps"]:
                model.eval()
                pm = pair_metrics(model, env, data["pairs_validation"], device)
                rank = ranking_loss(model, env, vt, data["child_distances_validation"][idx],
                                     device, options["rank_temperature"])
                score = pm["huber"] + options["rank_weight"] * rank
                eligible = min(ps.passes, rs.passes) >= 1
                selection_score = score
                policy = None
                if parent is not None:
                    policy, state = audit(model, env, device, reference=reference, cohort=data["tasks_restoration"][:, 0])
                    if mixture is not None:
                        mixture.refresh(state["hard_parents"])
                    selection_score = -policy["within_budget"]
                    eligible = pm["mae"] <= initial_validation_mae + config["continuation"]["maximum_validation_mae_increase"]
                if eligible and selection_score < best:
                    best, best_step = selection_score, step
                    save(step, out / "best.pt")
                record = {"step": step, "seconds": round(time.monotonic() - started, 2),
                          "distance_loss": float(pair_loss.detach()), "rank_loss": float(rank_loss.detach()),
                          "validation_mae": pm["mae"], "validation_objective": score,
                          "validation_shortest_action_rate": decisions(model, env, vt, device)["q"]["shortest_action_rate"],
                          "pair_passes": ps.passes, "decision_passes": rs.passes,
                          "selection_eligible": eligible, "best_step": best_step}
                if parent is not None:
                    record.update(policy=policy, sampling=mixture.report() if mixture is not None else None)
                    if mixture is not None:
                        record.update(pair_passes=mixture.ps.passes, decision_passes=mixture.rs.passes)
                log.write(json.dumps(record) + "\n"); log.flush()
                print(json.dumps(record), flush=True)
    save(step, out / "last.pt")
    saved = torch.load(out / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(saved["model"])
    summary = {"task": env.name, "source_commit": source_commit, "best_step": best_step,
               "last_step": step, "training_seconds": round(time.monotonic() - started, 2),
               "selection": ("all-state restoration count subject to fixed validation MAE guard"
                             if parent else "heldout intermediate goal distance Huber + full soft rank cross entropy after one coverage pass"),
               "parent": parent, "initial_policy": initial_policy,
               "torch_version": torch.__version__, "numpy_version": np.__version__,
               "final": evaluate(model, env, data, config, device)}
    if parent is not None:
        summary["all_state_policy"], _ = audit(model, env, device, reference=reference, cohort=data["tasks_restoration"][:, 0])
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--prepared-run", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--full-distance-table", type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if config["task"] != "puzzle8":
        raise ValueError("Puzzle8 trainer requires its own task contract")
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    env = Puzzle8()
    if args.prepared_run is None:
        data, summary = build(env, config, args.out)
    else:
        parent_config = json.loads((args.prepared_run / "config.json").read_text())
        if any(config[k] != parent_config[k] for k in ("task", "rules", "data", "model")):
            raise ValueError("Prepared labels must match the base contract")
        oracle = np.load(args.prepared_run / "oracle.npz")
        env.tables = oracle["distances"]
        env.goal_columns = {int(g): i for i, g in enumerate(oracle["goals"])}
        data = dict(np.load(args.prepared_run / "data.npz"))
        summary = {"prepared_run": str(args.prepared_run)}
    if args.full_distance_table is not None:
        oracle_summary = json.loads((args.full_distance_table.parent / "summary.json").read_text())
        if not oracle_summary["complete"] or oracle_summary["shape"] != [env.size, env.size]:
            raise ValueError("The full source-goal oracle is incomplete or mismatched")
        env.full_distances = np.load(args.full_distance_table, mmap_mode="r")
        (args.out / "full_oracle_source.json").write_text(json.dumps({"path": str(args.full_distance_table),
                  "summary": oracle_summary}, indent=2) + "\n")
    print(json.dumps({"data": summary}), flush=True)
    train(env, config, data, args.out, args.device, args.source_commit, args.checkpoint)


if __name__ == "__main__":
    main()
