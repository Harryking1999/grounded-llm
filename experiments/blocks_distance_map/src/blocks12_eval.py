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
    if config.get("mode") == "clear_sets":
        return clear_sets(config, checkpoint, out, source_commit, device)
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


def clear_sets(config, checkpoint, out, source_commit, device):
    """Frozen text questions plus disjoint seeded clearing cohorts; no training."""
    out.mkdir(parents=True, exist_ok=False)
    (out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    suite = json.loads(Path(config["frozen_suite"]).read_text())
    frozen = [r for r in suite["cases"] if r["task"] == "blocks"]
    if len(frozen) != 40 or any(int(r["goal"]) != 0 or r.get("shape_options")
                                or r["generation_placements"] != 12 for r in frozen):
        raise ValueError("Expected unchanged original-eight-shape twelve-placement40 clearing tasks")
    excluded = {int(r["initial"]) for r in frozen}
    for filename in config.get("exclude_roots", []):
        data = json.loads(Path(filename).read_text())
        roots = data if isinstance(data, list) else data["roots"]
        excluded.update(int(r["initial"]) for r in roots)
    initial_excluded = len(excluded)
    cohorts = [("text_original40", frozen)]
    for seed in config["seeds"]:
        rows, skipped = [], 0
        # Extra records handle duplicate/excluded roots without changing seed.
        for r in construct(config["boards_per_seed"] * 2, 12, seed):
            state = int(r["initial"])
            if state in excluded:
                skipped += 1
                continue
            excluded.add(state)
            rows.append({**r, "id": f"seed{seed}_{len(rows):04d}", "goal": 0})
            if len(rows) == config["boards_per_seed"]:
                break
        if len(rows) != config["boards_per_seed"]:
            raise ValueError("Insufficient distinct roots; no incomplete cohort published")
        cohorts.append((f"seed_{seed}", rows))
    scorer = QScorer(checkpoint, device)
    oracle = DistanceOracle(cache_limit=config["oracle_cache_limit"], seconds=config["oracle_seconds"])
    result = {"source_commit": source_commit, "checkpoint": str(checkpoint),
              "model_identity": scorer.identity, "frozen_suite": config["frozen_suite"],
              "initial_excluded_roots": initial_excluded, "groups": {},
              "scope": "Unchanged greedy, all legal successors, goal0. Oracle labels diagnose failures only; no search or oracle action selection. New cohorts exclude supplied roots and each other, not a certificate of unseen intermediate states or all training roots."}
    for name, cases in cohorts:
        rows = []
        for case in cases:
            state = int(case["initial"])
            row = {"id": case.get("id"), "source": str(state), "goal": "0",
                   **trace(scorer, state, 0, oracle)}
            reference = case.get("reference", {}).get("length")
            if reference is not None and reference != row["true_distance"]:
                raise ValueError("Exact distance differs from original text reference")
            rows.append(row)
        outcomes = Counter(r["outcome"] for r in rows)
        result["groups"][name] = {"count": len(rows), "success": outcomes["reached"],
            "success_rate": outcomes["reached"] / len(rows), "outcomes": dict(outcomes),
            "failure_within_first_two": sum(r["outcome"] != "reached" and r["moves"] <= 2 for r in rows),
            "distance_counts": dict(Counter(r["true_distance"] for r in rows)), "rows": rows}
        print(json.dumps({"group": name, **{k: v for k, v in result["groups"][name].items()
                                            if k != "rows"}}), flush=True)
    generated = [g for name, g in result["groups"].items() if name.startswith("seed_")]
    result["new_seed_aggregate"] = {"count": sum(g["count"] for g in generated),
                                  "success": sum(g["success"] for g in generated)}
    (out / "results.json").write_text(json.dumps(result, indent=2) + "\n")
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
