"""Frozen comparisons on identical endpoints, with explicit move metrics."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from experiments.puzzle_qmap.src.evaluate import decisions, pair_metrics, ranking_loss, rollout, task_indices
from experiments.puzzle_qmap.src.model import MapReader
from experiments.puzzle_qmap.src.post_eval import first_error
from experiments.puzzle_qmap.src.tasks import make_environment


def profile(reader, env, pairs, tasks, roll_tasks, config, device, horizon):
    truth = env.distance(env.successors(tasks[:, 0]), tasks[:, 1, None])
    temperature = config["training"]["rank_temperature"]
    logits = torch.as_tensor(-truth / temperature, device=device)
    entropy = float(-(torch.softmax(logits, -1) * torch.log_softmax(logits, -1)).sum(-1).mean())
    rank = ranking_loss(reader.model, env, tasks, truth, device, temperature)
    depth = env.distance(tasks[:, 0], tasks[:, 1])
    relabeled = np.column_stack((pairs[:, :2], env.distance(pairs[:, 0], pairs[:, 1])))
    pm = pair_metrics(reader.model, env, relabeled, device)
    return {"actions": list(env.actions), "metric": env.metric, "pairs": pm,
            "soft_rank_loss": rank, "rank_target_entropy": entropy, "rank_kl": rank - entropy,
            "objective": pm["huber"] + config["training"]["rank_weight"] * rank,
            "decisions": decisions(reader.model, env, tasks, device)["q"],
            "decisions_by_distance": {str(d): decisions(reader.model, env, tasks[depth == d], device)["q"]
                                      for d in np.unique(depth)},
            "first_error": first_error(reader.model, env, roll_tasks, device),
            "rollout": rollout(reader.model, env, roll_tasks, device, config["evaluation"]["budget_multiplier"],
                               horizon=horizon, return_tasks=True)["q"]}


def compare(run, reference_run, device):
    config = json.loads((run / "config.json").read_text())
    data = dict(np.load(run / "data.npz", allow_pickle=False))
    env = make_environment(config["task"], run / "oracle_depth.npy")
    old_env = make_environment("cube2", reference_run / "oracle_depth.npy")
    best = MapReader(env, run / "best.pt", device)
    last = MapReader(env, run / "last.pt", device)
    old = MapReader(old_env, reference_run / "best.pt", device)
    horizon = int(env.diameter * config["evaluation"]["budget_multiplier"])
    result = {"run": str(run), "reference_run": str(reference_run), "common_horizon": horizon,
              "comparison_scope": "same endpoints; actions, data, initialization and training differ; no single-factor causal claim",
              "cohorts": {}}
    for split in ("test", "reference_test"):
        idx = task_indices(env, data, split, config["evaluation"]["decision_count"], config)
        roll_idx = task_indices(env, data, split, config["evaluation"]["rollout_count"], config)
        tasks, roll_tasks = data[f"tasks_{split}"][idx], data[f"tasks_{split}"][roll_idx]
        cohort = {"rollout_state_pairs": roll_tasks.tolist(), "models": {}}
        for name, reader, rules in (("six_best", best, env), ("six_last", last, env),
                                    ("nine_reference", old, old_env), ("nine_weights_six_actions", old, env)):
            cohort["models"][name] = profile(reader, rules, data[f"pairs_{split}"], tasks, roll_tasks,
                                               config, device, horizon)
        new = np.array(cohort["models"]["six_best"]["rollout"]["per_task"]["solved"])
        previous = np.array(cohort["models"]["nine_reference"]["rollout"]["per_task"]["solved"])
        cohort["paired_best_vs_nine"] = {"both_solved": int((new & previous).sum()),
                                         "six_only": int((new & ~previous).sum()),
                                         "nine_only": int((~new & previous).sum()),
                                         "neither": int((~new & ~previous).sum())}
        result["cohorts"][split] = cohort
    (run / "comparison.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({s: {k: v["rollout"]["solved"] for k, v in c["models"].items()}
                      for s, c in result["cohorts"].items()}), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--reference-run", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    torch.set_num_threads(4)
    compare(args.run, args.reference_run, args.device)
