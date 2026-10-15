"""One small Q-only pipeline: exact data -> train -> frozen evaluation/export."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from .data import build
from .evaluate import decisions, encode, evaluate, pair_metrics, ranking_loss, task_indices
from .model import MODEL_TYPE, QMap, distance, soft_rank_loss
from .sampling import CoverageSampler
from .tasks import make_environment


def train(env, config, data, out, device, source_commit):
    torch.manual_seed(config["seed"])
    rng = np.random.default_rng(config["seed"] + 1)
    options = config["training"]
    torch.set_num_threads(options["cpu_threads"])
    model = QMap(env.feature_dim, **config["model"], cap=env.diameter).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=options["learning_rate"])
    pairs = data["pairs_train"]
    tasks, children = data["tasks_train"], data["children_train"]
    pair_sampler = rank_sampler = None
    if "sampling" in config:
        sampling = config["sampling"]
        pair_sampler = CoverageSampler(pairs[:, 2], rng, data.get("broad_pair_count", len(pairs)),
                                       sampling["broad_fraction"])
        parent_distance = data.get("parent_distances_train")
        if parent_distance is None:
            parent_distance = env.distance(tasks[:, 0], tasks[:, 1])
        rank_sampler = CoverageSampler(parent_distance, rng, data.get("broad_task_count", len(tasks)),
                                       sampling["broad_fraction"],
                                       int(data["context_task_start"]) if "context_task_start" in data else None,
                                       sampling["context_fraction"])
        for sampler, batch in ((pair_sampler, options["batch_size"]), (rank_sampler, options["decision_batch_size"])):
            if options["steps"] * round(batch * sampling["broad_fraction"]) / sampler.broad_count < sampling["minimum_broad_passes"]:
                raise ValueError("Training budget cannot cover the contracted breadth")

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

    def validate():
        model.eval()
        metrics = pair_metrics(model, env, data["pairs_validation"], device)
        val_tasks = data["tasks_validation"][val_idx]
        rank = ranking_loss(model, env, val_tasks, data["child_distances_validation"][val_idx],
                            device, options["rank_temperature"])
        actions = decisions(model, env, val_tasks, device)
        score = metrics["huber"] + options["rank_weight"] * rank
        return score, metrics, actions["q"], rank

    started = time.monotonic()
    initial_score, initial_pairs, initial_decisions, _ = validate()
    initial_test = evaluate(model, env, data, config, device)
    best, best_step, stale = float("inf"), 0, 0
    with (out / "progress.jsonl").open("w") as log:
        for step in range(1, options["steps"] + 1):
            model.train()
            sampled_ids = (pair_sampler.draw(options["batch_size"]) if pair_sampler is not None
                           else rng.integers(len(pairs), size=options["batch_size"]))
            sampled = pairs[sampled_ids]
            pair_loss = F.smooth_l1_loss(distance(q(sampled[:, 0]), q(sampled[:, 1])),
                                         torch.as_tensor(sampled[:, 2], dtype=torch.float32, device=device))
            idx = (rank_sampler.draw(options["decision_batch_size"]) if rank_sampler is not None
                   else rng.integers(len(tasks), size=options["decision_batch_size"]))
            candidate = children[idx]
            valid = torch.as_tensor(candidate >= 0, device=device)
            # Use an actual training parent for padding, never an arbitrary heldout ID.
            safe = np.where(candidate >= 0, candidate, tasks[idx, 0, None])
            candidate_q = q(safe)
            goal_q = q(tasks[idx, 1])
            scores = distance(candidate_q, goal_q[:, None])
            truth = torch.as_tensor(data["child_distances_train"][idx], device=device)
            rank_loss = soft_rank_loss(scores, truth, valid, options["rank_temperature"])
            loss = pair_loss + options["rank_weight"] * rank_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            if step == 1 or step % options["eval_every"] == 0 or step == options["steps"]:
                score, metrics, actions, val_rank = validate()
                coverage = ({"pair_broad_passes": pair_sampler.passes, "decision_broad_passes": rank_sampler.passes}
                            if pair_sampler is not None else {})
                eligible = (step >= options.get("min_steps", 0) and
                            (not coverage or min(coverage.values()) >= config["sampling"]["minimum_broad_passes"]))
                if eligible and score < best - options["min_improvement"]:
                    best, best_step, stale = score, step, 0
                    torch.save({"model_type": MODEL_TYPE, "task": env.name, "cap": env.diameter,
                                "actions": list(env.actions), "model": model.state_dict(),
                                "model_config": config["model"], "config": config, "source_commit": source_commit,
                                "step": step, "coverage": coverage, "optimizer": optimizer.state_dict()}, out / "best.pt")
                elif eligible:
                    stale += 1
                record = {"step": step, "seconds": round(time.monotonic() - started, 2),
                          "distance_loss": float(pair_loss.detach()), "rank_loss": float(rank_loss.detach()),
                          "validation_mae": metrics["mae"], "validation_huber": metrics["huber"],
                          "validation_soft_rank_loss": val_rank,
                          "validation_shortest_action_rate": actions["shortest_action_rate"],
                          "validation_objective": score, "selection_eligible": eligible,
                          "best_step": best_step, **coverage}
                log.write(json.dumps(record) + "\n")
                log.flush()
                print(json.dumps(record), flush=True)
                if eligible and stale >= options["patience"]:
                    break
    saved = torch.load(out / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(saved["model"])
    result = {"model_type": MODEL_TYPE, "task": env.name, "source_commit": source_commit, "best_step": best_step, "last_step": step,
              "training_seconds": round(time.monotonic() - started, 2), "device": str(device),
              "best_model_training_coverage": saved.get("coverage", {}),
              "torch_version": torch.__version__, "numpy_version": np.__version__,
              "selection": "validation Huber + rank_weight * full soft rank cross entropy",
              "initial_validation_objective": initial_score, "initial_validation_pairs": initial_pairs,
              "initial_validation_decisions": initial_decisions, "initial": initial_test,
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
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    env = make_environment(config["task"], args.out / "oracle_depth.npy")
    if env.name == "cube2":
        env.build_oracle(args.out / "oracle_depth.npy")
    data, summary = build(env, config, args.out)
    print(json.dumps({"data": summary}), flush=True)
    result = train(env, config, data, args.out, args.device, args.source_commit)
    print(json.dumps({"complete": str(args.out), "best_step": result["best_step"],
                      "test_rollout": result["final"]["test"]["rollout"]}), flush=True)


if __name__ == "__main__":
    main()
