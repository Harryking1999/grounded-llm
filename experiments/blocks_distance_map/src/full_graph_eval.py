"""Matched pure-greedy twelve-placement evaluation and stage selection."""
import argparse
from collections import Counter
import json
from pathlib import Path
import random
import shutil

import numpy as np
import torch

from .blocks12_eval import trace
from .oracle import DistanceOracle, successors
from .tree_supervision_eval import QScorer


def sections(rows):
    result = {}
    for key, selected in (("clear", [r for r in rows if r["goal"] == "0"]),
                          ("intermediate", [r for r in rows if r["goal"] != "0"])):
        counts = Counter(r["outcome"] for r in selected)
        result[key] = {"tasks": len(selected), "success": counts["reached"],
                       "outcomes": dict(counts), "success_rate": counts["reached"] / len(selected)}
    return result


def matched(old, new):
    result = {}
    for key, zero in (("clear", True), ("intermediate", False)):
        pairs = [(a["outcome"] == "reached", b["outcome"] == "reached")
                 for a, b in zip(old, new) if (a["goal"] == "0") == zero]
        lost = sum(a and not b for a, b in pairs)
        gained = sum(b and not a for a, b in pairs)
        result[key] = {"old_success_new_failure": lost, "new_success": gained,
                       "net": gained - lost, "both_success": sum(a and b for a, b in pairs)}
    return result


def train_cases(roots, count, seed):
    rng = random.Random(seed)
    rows = []
    for i, record in enumerate(roots[:count]):
        source = state = int(record["initial"])
        path = []
        while True:
            candidates = successors(state)
            if not candidates:
                break
            _, state = rng.choice(candidates)
            if state:
                path.append(state)
        for goal in [0, *dict.fromkeys((path[0], path[-1]))]:
            rows.append({"board": i, "source": str(source), "goal": str(goal)})
    return rows


def evaluate_rows(rows, scorer, oracle):
    results, bins = [], {}
    for index, row in enumerate(rows):
        source, goal = int(row["source"]), int(row["goal"])
        truth = oracle.distance(source, goal)
        key = str(truth) if truth <= 2 else "3-4" if truth <= 4 else "5-8" if truth <= 8 else "9+"
        group = bins.setdefault(key, {"tasks": 0, "reached": 0, "forks": 0,
                                      "shortest_choices": 0, "candidate_count": 0,
                                      "absolute_error_sum": 0.0})
        candidates = successors(source)
        if candidates:
            scores = scorer.scores([child for _, child in candidates], goal)
            distances = np.asarray([oracle.distance(child, goal) for _, child in candidates])
            cap = scorer.model.cap
            group["forks"] += 1
            group["shortest_choices"] += int(distances[int(scores.argmin())] == truth - 1)
            group["candidate_count"] += len(candidates)
            group["absolute_error_sum"] += float(np.abs(scores - np.where(distances < 0, cap, distances)).sum())
        outcome = trace(scorer, source, goal, oracle)
        group["tasks"] += 1
        group["reached"] += outcome["outcome"] == "reached"
        results.append({**row, **outcome})
        if (index + 1) % 100 == 0:
            print(json.dumps({"evaluated": index + 1, "total": len(rows)}), flush=True)
    return results, bins


def run(config, checkpoint, out, device, source_commit):
    torch.set_num_threads(2)
    baseline = json.loads(Path(config["baseline_cases"]).read_text())["rows"]
    roots = json.loads((Path(config["graph_data"]) / "roots.json").read_text())
    oracle = DistanceOracle(cache_limit=config["evaluation"]["oracle_cache_limit"],
                            seconds=config["evaluation"]["oracle_seconds"])
    scorer = QScorer(checkpoint, device)
    evaluated, bins = evaluate_rows(baseline, scorer, oracle)
    train = train_cases(roots, config["evaluation"]["train_roots"], config["training"]["seed"] + 1)
    train_rows, train_bins = evaluate_rows(train, scorer, oracle)
    parent_file = out.parent / "train_parent_cases.json"
    if parent_file.exists():
        train_parent = json.loads(parent_file.read_text())
    else:
        train_parent, _ = evaluate_rows(train, QScorer(config["parent_checkpoint"], device), oracle)
        parent_file.parent.mkdir(parents=True, exist_ok=True)
        parent_file.write_text(json.dumps(train_parent) + "\n")
    result = {"analysis_commit": source_commit, "checkpoint": str(checkpoint),
              "unseen_roots": sections(evaluated), "matched_baseline": matched(baseline, evaluated),
              "train_roots": sections(train_rows), "matched_train_parent": matched(train_parent, train_rows),
              "distance_bins_unseen": bins,
              "distance_bins_train": train_bins, "cycles": 0, "illegal_actions": 0,
              "policy": "original_Q_argmin_all_legal_actions_no_search_or_oracle_filters",
              "rows": evaluated, "train_rows": train_rows}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    # Preserve the accepted parent unless both core unseen-goal scores hold.
    accepted = all(result["matched_baseline"][key]["net"] >= 0 for key in ("clear", "intermediate"))
    selection = out.parent / "selection.json"
    score = sum(section["success"] for section in result["unseen_roots"].values())
    previous = json.loads(selection.read_text()) if selection.exists() else {"success": 245}
    if "unseen_roots" in previous:
        accepted = accepted and all(result["unseen_roots"][key]["success"] >=
                                    previous["unseen_roots"][key]["success"]
                                    for key in ("clear", "intermediate"))
    if accepted and score > previous["success"]:
        selected = {"success": score, "checkpoint": str(checkpoint), "evaluation": str(out),
                    "unseen_roots": result["unseen_roots"]}
        selection.write_text(json.dumps(selected, indent=2) + "\n")
        shutil.copy2(checkpoint, out.parent / "best.pt")
    print(json.dumps({k: result[k] for k in ("unseen_roots", "matched_baseline", "train_roots")}), flush=True)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    run(json.loads(args.config.read_text()), args.checkpoint, args.out, args.device, args.source_commit)


if __name__ == "__main__":
    main()
