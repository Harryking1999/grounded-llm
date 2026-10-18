"""User-requested fixed exact-distance 6--8 probes on the unchanged monitor roots."""
import argparse
from collections import Counter
import json
from pathlib import Path
import random

import torch

from .full_graph_eval import evaluate_rows
from .oracle import DistanceOracle, successors
from .tree_supervision_eval import QScorer
from experiments.gcml_counterexamples.src import blocks


def make_clear_cases(monitor, seed):
    """Residual starts, exact6/7/8 to zero, same unseen root boards; no Q filtering."""
    rng = random.Random(seed)
    oracle = DistanceOracle(cache_limit=8000000, seconds=14400)
    result = []
    for board, record in enumerate(monitor["roots"]):
        tiles = [blocks.PLACEMENTS[blocks.BY_ACTION[(a["shape_id"], a["row"], a["col"]) ]][0]
                 for a in record["construction"]]
        wanted = 6 + board % 3
        for _ in range(1000):
            source = 0
            for tile in rng.sample(tiles, wanted):
                source |= tile
            if oracle.distance(source, 0) == wanted:
                result.append({"board": board, "source": str(source), "goal": "0",
                               "true_distance": wanted})
                break
        else:
            raise RuntimeError(f"Could not construct exact {wanted}-move clearing on root {board}")
    return result


def make_cases(monitor, seed):
    roots = {r["board"]: r["source"] for r in monitor}
    rng = random.Random(seed)
    oracle = DistanceOracle(cache_limit=8000000, seconds=14400)
    result = []
    for index, (board, source) in enumerate(roots.items()):
        wanted = 6 + index % 3
        for _ in range(100):
            state, goals = int(source), []
            while True:
                choices = successors(state)
                if not choices:
                    break
                _, state = rng.choice(choices)
                if state and oracle.distance(int(source), state) == wanted:
                    goals.append(state)
            if goals:
                goal = rng.choice(goals)
                result.append({"board": board, "source": source, "goal": str(goal),
                               "true_distance": wanted})
                break
        else:
            raise RuntimeError(f"Could not construct an exact {wanted}-move goal on monitor root {board}")
    return result


def run(monitor, checkpoint, cases, out, reference=None, seed=2026101904, goal_kind="nonempty"):
    torch.set_num_threads(2)
    if cases.exists():
        dataset = json.loads(cases.read_text())
    else:
        raw = json.loads(monitor.read_text())
        dataset = {"seed": seed, "goal_kind": goal_kind,
            "rows": make_clear_cases(raw, seed) if goal_kind == "clear" else make_cases(raw["rows"], seed),
            "scope": "Fixed exact-distance6/7/8 on unchanged unseen monitor roots, model-independent generation. Clear tasks use residual starts and goal0; nonempty tasks use full-root starts. Original9--13 and full-root clearing tasks unchanged. Repeated monitoring, not sealed final acceptance or training data."}
        cases.parent.mkdir(parents=True, exist_ok=True)
        pending = cases.with_suffix(".pending")
        pending.write_text(json.dumps(dataset) + "\n")
        pending.replace(cases)
    rows, bins, trajectory = evaluate_rows(dataset["rows"], QScorer(checkpoint, "cpu"),
                        DistanceOracle(cache_limit=8000000, seconds=14400))
    if any(r["true_distance"] not in (6, 7, 8) for r in rows):
        raise ValueError("Probe contains an out-of-range exact distance")
    old = json.loads(reference.read_text())["rows"] if reference else None
    if old and [(r["source"], r["goal"]) for r in old] != [(r["source"], r["goal"]) for r in rows]:
        raise ValueError("Baseline and candidate must share the exact fixed tasks")
    def summary(indices):
        after = [rows[i]["outcome"] == "reached" for i in indices]
        result = {"tasks": len(indices), "success": sum(after),
                  "outcomes": dict(Counter(rows[i]["outcome"] for i in indices))}
        if old is not None:
            before = [old[i]["outcome"] == "reached" for i in indices]
            lost = sum(a and not b for a, b in zip(before, after))
            gained = sum(b and not a for a, b in zip(before, after))
            result.update(old_success=sum(before), lost=lost, gained=gained, net=gained-lost)
        return result
    result = {"checkpoint": str(checkpoint), "cases": str(cases), "seed": dataset["seed"],
              "scope": dataset["scope"], "6-8": summary(list(range(len(rows)))),
              "by_distance": {str(d): summary([i for i, r in enumerate(rows) if r["true_distance"] == d])
                              for d in (6, 7, 8)}, "rows": rows, "trajectory": trajectory, "distance_bins": bins}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"checkpoint": str(checkpoint), "6-8": result["6-8"],
                      "by_distance": result["by_distance"]}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--monitor-cases", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--goal-kind", choices=["nonempty", "clear"], default="nonempty")
    args = parser.parse_args()
    run(args.monitor_cases, args.checkpoint, args.cases, args.out, args.reference, goal_kind=args.goal_kind)
