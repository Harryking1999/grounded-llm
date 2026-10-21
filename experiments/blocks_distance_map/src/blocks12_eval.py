"""Fixed twelve-placement, original-eight-shape evaluation of an unchanged Qmap."""
import argparse
from collections import Counter
import json
from pathlib import Path
import random

import numpy as np
import torch

import grounded_llm.tasks.blocks.rules as blocks
from grounded_llm.tasks.blocks.oracle import DistanceOracle, OracleBudgetExceeded, successors
from .tree_supervision_eval import QScorer


from grounded_llm.tasks.blocks.construction import construct


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
    if config.get("mode") == "frozen_clear":
        return frozen_clear(config, checkpoint, out, source_commit, device)
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


def frozen_clear(config, checkpoint, out, source_commit, device):
    """Replay the fixed clearing benchmark without an oracle in execution."""
    cases = json.loads(Path(config["cases"]).read_text())["cases"]
    if (len(cases) != config["case_count"] or len({c["source"] for c in cases}) != len(cases)
            or len({c["id"] for c in cases}) != len(cases)
            or any(int(c["goal"]) != 0 for c in cases)):
        raise ValueError("Fixed clearing benchmark identity/count/goal mismatch")
    scorer = QScorer(checkpoint, device)
    rows, failed_paths = [], []
    for case in cases:
        state, moves = int(case["source"]), 0
        budget = state.bit_count() // 2
        path = []
        while state and moves < budget:
            candidates = successors(state)
            if not candidates:
                break
            choice = int(np.argmin(scorer.scores([child for _, child in candidates], 0)))
            if config.get("diagnose_failures"):
                path.append((state, candidates[choice][0], candidates[choice][1]))
            state = candidates[choice][1]
            moves += 1
        rows.append({**case, "moves": moves, "outcome": "reached" if state == 0 else "no_candidate",
                     "final_state": str(state)})
        if state and config.get("diagnose_failures"):
            failed_paths.append((case, path))
    result = {"source_commit": source_commit, "checkpoint": str(checkpoint),
              "model_identity": scorer.identity, "cases": config["cases"], "count": len(rows),
              "success": sum(r["outcome"] == "reached" for r in rows),
              "policy": "original_argmin_all_legal_successors_goal0_no_search_no_oracle",
              "rows": rows}
    reference_path = config.get("reference_results")
    if reference_path:
        reference = json.loads(Path(reference_path).read_text())
        old = reference.get("rows")
        if old is None:
            old = [r for name, group in reference["groups"].items()
                   if name.startswith("seed_") for r in group["rows"]]
        if [(r["id"], r["source"], r["goal"]) for r in old] != [
                (r["id"], r["source"], r["goal"]) for r in rows]:
            raise ValueError("Reference results must contain the identical ordered1000 cases")
        before = np.asarray([r["outcome"] == "reached" for r in old])
        after = np.asarray([r["outcome"] == "reached" for r in rows])
        result["matched_reference"] = {"old_success": int(before.sum()),
            "new_success": int(after.sum()), "lost": int((before & ~after).sum()),
            "gained": int((~before & after).sum())}
    result["distance_bins"] = {}
    for distance in sorted({r["true_distance"] for r in rows}):
        subset = [r for r in rows if r["true_distance"] == distance]
        result["distance_bins"][str(distance)] = {"count": len(subset),
            "success": sum(r["outcome"] == "reached" for r in subset)}
    if config.get("historical_510_manifest"):
        from .full_graph_eval import historical_retention
        result["historical_510"] = historical_retention(config["historical_510_manifest"], checkpoint,
                                                       config.get("historical_510_reference"))
    out.mkdir(parents=True, exist_ok=False)
    if config.get("diagnose_failures"):
        # Offline diagnosis starts only after ALL greedy executions finish.
        oracle = DistanceOracle(cache_limit=2000000, seconds=config.get("diagnostic_seconds", 90))
        decisions, unknown = [], 0
        for case, path in failed_paths:
            try:
                for depth, (state, action, child) in enumerate(path):
                    if oracle.distance(child, 0) >= 0:
                        continue
                    candidates = successors(state)
                    scores = scorer.scores([c for _, c in candidates], 0)
                    truths = [oracle.distance(c, 0) for _, c in candidates]
                    decisions.append({"id": case["id"], "initial_distance": case["true_distance"],
                        "depth": depth, "state": str(state), "chosen_action": action,
                        "remaining_distance": oracle.distance(state, 0), "fork_degree": len(candidates),
                        "chosen_shape": blocks.PLACEMENTS[action][1]["shape_id"],
                        "candidates": [{"action": a, "child": str(c), "distance": d, "score": float(v)}
                                       for (a, c), d, v in zip(candidates, truths, scores)]})
                    break
            except OracleBudgetExceeded:
                unknown += 1
        result["failure_diagnosis"] = {"failed_cases": len(failed_paths), "analyzed": len(decisions),
            "unknown_budget": unknown, "first_trap_remaining_distance": dict(Counter(
                str(d["remaining_distance"]) for d in decisions)),
            "first_trap_shape": dict(Counter(str(d["chosen_shape"]) for d in decisions))}
        (out / "failure_decisions.json").write_text(json.dumps(decisions) + "\n")
    (out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    (out / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k not in ("rows", "historical_510")}), flush=True)
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
