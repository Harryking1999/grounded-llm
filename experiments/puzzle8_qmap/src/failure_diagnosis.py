"""Exact closed-loop error diagnosis, without giving oracle actions to Q."""
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
import torch

from grounded_llm.core.qmap.sampling import task_indices
from .checkpoint_audit import load_model
from grounded_llm.tasks.puzzle8.environment import Puzzle8
from grounded_llm.tasks.puzzle8.policy_audit import audit


@torch.no_grad()
def goal_heights(model, features, goal):
    target = model.encode(features[goal:goal + 1])
    return np.concatenate([model.distance(model.encode(part), target).cpu().numpy()
                           for part in features.split(8192)])


def traces(env, heights, tasks):
    rows = []
    for start, goal in tasks:
        s, goal = int(start), int(goal)
        initial = int(env.distance(s, goal))
        seen, first_error, cycle = set(), None, None
        for step in range(63):
            if s == goal:
                break
            if s in seen:
                cycle = s
                break
            seen.add(s)
            children = env.next[s]
            valid = children >= 0
            legal = children[valid]
            scores = heights[legal]
            labels = env.distance(legal, goal)
            chosen = int(scores.argmin())
            true = int(env.distance(s, goal))
            if first_error is None and labels[chosen] != true - 1:
                first_error = {"parent": s, "goal": goal, "step": step,
                               "remaining_distance": true, "degree": len(legal),
                               "optimal_actions": int((labels == true - 1).sum()),
                               "chosen": int(legal[chosen]),
                               "candidates": [{"action": env.actions[a], "state": int(child),
                                               "truth": int(label), "score": float(score)}
                                              for a, child, label, score in
                                              zip(np.flatnonzero(valid), legal, labels, scores)]}
            s = int(legal[chosen])
        rows.append({"start": int(start), "goal": goal, "true_distance": initial,
                     "success": s == goal, "cycle_state": cycle,
                     "first_nonoptimal": first_error})
    return rows


def summarize(rows):
    bins = {}
    for row in rows:
        b = bins.setdefault(str(row["true_distance"]), {"tasks": 0, "success": 0})
        b["tasks"] += 1; b["success"] += row["success"]
    errors = [r["first_nonoptimal"] for r in rows if not r["success"] and r["first_nonoptimal"]]
    return {"tasks": len(rows), "success": sum(r["success"] for r in rows),
            "by_true_distance": bins,
            "first_error_remaining_distance": dict(Counter(e["remaining_distance"] for e in errors)),
            "first_error_step": dict(Counter(e["step"] for e in errors)),
            "first_error_degree": dict(Counter(e["degree"] for e in errors))}


def run(checkpoint, prepared, out):
    torch.set_num_threads(1)
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    env = Puzzle8()
    with np.load(prepared / "oracle.npz") as packed:
        env.tables = packed["distances"]
        env.goal_columns = {int(g): i for i, g in enumerate(packed["goals"])}
    with np.load(prepared / "data.npz") as packed:
        data = {k: packed[k] for k in packed.files if not k.endswith("_train")}
    model = load_model(saved, "cpu")
    features = torch.as_tensor(env.features(np.arange(env.size)))
    policy, state = audit(model, env, "cpu")
    depth = env.distance(np.arange(env.size), 0)
    result = {"checkpoint": str(checkpoint), "step": saved["continuation_step"],
              "adam_steps": sorted({int(v["step"]) for v in saved["optimizer"]["state"].values()}),
              "policy": policy,
              "restoration_by_depth": {str(int(d)): {"states": int((depth == d).sum()),
                   "success": int(state["within"][depth == d].sum()),
                   "wrong_decisions": int((~state["correct"][depth == d]).sum())} for d in np.unique(depth)},
              "scope": "Oracle labels only; original greedy policy; monitoring goals excluded from repair labels"}
    for split in ("validation", "test"):
        ids = task_indices(env, data, split, saved["config"]["evaluation"]["rollout_count"], saved["config"])
        tasks = data[f"tasks_{split}"][ids]
        rows = []
        for goal in np.unique(tasks[:, 1]):
            heights = goal_heights(model, features, int(goal))
            rows.extend(traces(env, heights, tasks[tasks[:, 1] == goal]))
        result[split] = {"summary": summarize(rows), "rows": rows}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result) + "\n")
    print(json.dumps({"policy": policy, "test": result["test"]["summary"]}), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--prepared-run", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    run(args.checkpoint, args.prepared_run, args.out)
