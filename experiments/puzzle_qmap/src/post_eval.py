"""Frozen-Q fit and first decision error; no optimization or oracle policy aid."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .evaluate import decisions, encode, evaluate, pair_metrics, rollout, task_indices
from .model import MapReader, distance
from .tasks import make_environment


@torch.no_grad()
def first_error(model, env, tasks, device):
    current, goals = tasks[:, 0].copy(), tasks[:, 1]
    active = current != goals
    records = []
    for step in range(1, env.diameter + 1):
        ids = np.flatnonzero(active)
        if not len(ids):
            break
        children = env.successors(current[ids])
        scores = distance(encode(model, env, np.maximum(children, 0), device),
                          encode(model, env, goals[ids], device)[:, None]).cpu().numpy()
        scores[children < 0] = np.inf
        chosen = children[np.arange(len(ids)), scores.argmin(axis=1)]
        before = env.distance(current[ids], goals[ids])
        after = env.distance(chosen, goals[ids])
        wrong = after != before - 1
        for depth, next_depth in zip(before[wrong], after[wrong]):
            records.append((step, int(depth), "same_layer" if next_depth == depth else "away"))
        current[ids] = chosen
        active[ids[wrong | (chosen == goals[ids])]] = False
    return {"count": len(tasks), "all_steps_shortest": len(tasks) - len(records),
            "first_error_step": {str(s): sum(r[0] == s for r in records) for s in sorted({r[0] for r in records})},
            "first_error_parent_depth": {str(d): sum(r[1] == d for r in records) for d in sorted({r[1] for r in records})},
            "first_error_type": {kind: sum(r[2] == kind for r in records) for kind in ("same_layer", "away")}}


def post_eval(run, device, analysis_commit, compare_checkpoint=None):
    config = json.loads((run / "config.json").read_text())
    data = dict(np.load(run / "data.npz", allow_pickle=False))
    env = make_environment(config["task"], run / "oracle_depth.npy")
    reader = MapReader(env, run / "best.pt", device)
    rng = np.random.default_rng(config["seed"] + 37)
    pairs = data["pairs_train"]
    ids = np.concatenate([rng.choice(np.flatnonzero(pairs[:, 2] == d), min(6000 // env.diameter, int((pairs[:, 2] == d).sum())), replace=False)
                          for d in np.unique(pairs[:, 2]) if d > 0])
    tasks = data["tasks_train"][task_indices(env, data, "train", 1000, config)]
    result = {"analysis_commit": analysis_commit, "training_fit_distance_balanced": {
        "pairs": pair_metrics(reader.model, env, pairs[ids], device),
        "decisions": decisions(reader.model, env, tasks, device)["q"],
        "rollout": rollout(reader.model, env, tasks[:512], device, config["evaluation"]["budget_multiplier"])["q"]}}
    idx = task_indices(env, data, "test", config["evaluation"]["rollout_count"], config)
    result["first_error_test"] = first_error(reader.model, env, data["tasks_test"][idx], device)
    if compare_checkpoint is not None:
        other = MapReader(env, compare_checkpoint, device)
        compared = evaluate(other.model, env, data, config, device)
        result["previous_pure_q_same_tasks"] = {name: compared[name] for name in ("test", "reference_test")}
    (run / "post_eval.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"task": env.name, "train_shortest": result["training_fit_distance_balanced"]["decisions"]["shortest_action_rate"],
                      "first_error": result["first_error_test"]}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--analysis-commit", required=True)
    parser.add_argument("--compare-checkpoint", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(4)
    post_eval(args.run, args.device, args.analysis_commit, args.compare_checkpoint)
