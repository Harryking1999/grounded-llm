"""Fixed twelve-placement, original-eight-shape evaluation of an unchanged Qmap."""
import argparse
from collections import Counter
import json
from pathlib import Path
import random

import numpy as np
import torch

from experiments.gcml_counterexamples.src import blocks
from .oracle import DistanceOracle, successors
from .tree_supervision_eval import QScorer


def construct(count, placements, seed):
    rng = random.Random(seed)
    roots, seen = [], set()
    while len(roots) < count:
        state, witness = 0, []
        for _ in range(placements):
            shape = rng.randrange(len(blocks.SHAPES))
            legal = [(tile, action) for tile, action in blocks.PLACEMENTS
                     if action["shape_id"] == shape and not tile & state]
            if not legal:
                break
            tile, action = rng.choice(legal)
            state |= tile
            witness.append(action)
        if len(witness) == placements and state not in seen:
            seen.add(state)
            roots.append({"initial": str(state), "construction": witness})
    return roots


def trace(scorer, source, goal, oracle):
    true_distance = oracle.distance(source, goal)
    if true_distance < 0:
        raise ValueError("Evaluation tasks require a certified reachable goal")
    state, actions = source, []
    # Every legal removal clears at least two cells. This covers every finite
    # legal path; twelve construction placements do not impose a move budget.
    for _ in range(source.bit_count() // 2 + 1):
        if state == goal:
            return {"outcome": "reached", "moves": len(actions),
                    "true_distance": true_distance, "extra_moves": len(actions) - true_distance}
        candidates = successors(state)
        if not candidates:
            return {"outcome": "no_candidate", "moves": len(actions), "true_distance": true_distance}
        choice = int(np.argmin(scorer.scores([child for _, child in candidates], goal)))
        action, child = candidates[choice]
        actions.append(action)
        if oracle.distance(child, goal) < 0:
            kind = ("removed_goal_cell" if child & goal != goal else
                    "local_coverage_failure" if not blocks.locally_supported(child ^ goal) else
                    "globally_unreachable")
            return {"outcome": kind, "moves": len(actions), "true_distance": true_distance}
        state = child
    raise AssertionError("A strictly decreasing removal path exceeded its cell bound")


def run(config, checkpoint, out, source_commit, device):
    out.mkdir(parents=True, exist_ok=False)
    (out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    roots = construct(config["boards"], config["construction_placements"], config["seed"])
    oracle = DistanceOracle(cache_limit=config["oracle_cache_limit"], seconds=config["oracle_seconds"])
    scorer = QScorer(checkpoint, device)
    rng = random.Random(config["seed"] + 1)
    rows = []
    for index, record in enumerate(roots):
        source = int(record["initial"])
        path, state = [], source
        while True:
            choices = successors(state)
            if not choices:
                break
            _, state = rng.choice(choices)
            if state:
                path.append(state)
        goals = [0]
        if path:
            goals.extend(dict.fromkeys((path[0], path[-1])))
        for goal in goals:
            rows.append({"board": index, "source": str(source), "goal": str(goal),
                         **trace(scorer, source, goal, oracle)})
        if (index + 1) % 25 == 0:
            print(json.dumps({"evaluated_boards": index + 1}), flush=True)
    result = {"analysis_commit": source_commit, "scorer": scorer.identity,
              "scope": "fresh twelve-placement boards, original eight shapes; greedy Q without oracle access",
              "boards": len(roots), "sections": {}}
    for name, subset in (("clear", [r for r in rows if r["goal"] == "0"]),
                         ("intermediate", [r for r in rows if r["goal"] != "0"])):
        counts = Counter(row["outcome"] for row in subset)
        result["sections"][name] = {"tasks": len(subset), "outcomes": dict(counts),
                                    "success_rate": counts["reached"] / len(subset) if subset else None}
    (out / "cases.json").write_text(json.dumps({"roots": roots, "rows": rows}) + "\n")
    (out / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    torch.set_num_threads(2)
    run(json.loads(args.config.read_text()), args.checkpoint, args.out, args.source_commit, args.device)


if __name__ == "__main__":
    main()
