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


def distance_goal_success(rows):
    result = {}
    for band, low, high in (("6-8", 6, 8), ("9-13", 9, 13)):
        for kind, empty in (("clear", True), ("nonempty", False)):
            chosen = [r for r in rows if low <= r["true_distance"] <= high and (r["goal"] == "0") == empty]
            result[f"{band}_{kind}"] = {"tasks": len(chosen),
                "success": sum(r["outcome"] == "reached" for r in chosen)}
    return result


def historical_retention(manifest, checkpoint, reference=None):
    from experiments.flamingo_map_reader.src.blocks import FrozenBoardMap
    from experiments.flamingo_map_reader.src.blocks_sft import greedy_demonstration
    data = json.loads(Path(manifest).read_text())
    records = [r for r in data["records"] if r["split"] == "test"]
    if len(records) != 510:
        raise ValueError("Historical retention requires the unchanged fixed510 tasks")
    qmap = FrozenBoardMap.load(checkpoint, "cpu")
    rows = []
    for r in records:
        demo = greedy_demonstration(qmap, int(r["start"]), int(r["goal"]),
            rng=np.random.default_rng(r["sample_seed"]),
            max_actions=data["config"]["maximum_demonstration_actions"],
            reported=data["config"]["data"]["reported_candidates"])
        rows.append({"trajectory_id": r["trajectory_id"], "true_distance": r["shortest_moves"],
                     "success": demo.success})
    result = {"tasks": len(rows), "success": sum(r["success"] for r in rows),
            "6-8_success": sum(r["success"] for r in rows if 6 <= r["true_distance"] <= 8),
            "rows": rows, "scope": "Exact original fixed510, original greedy/RNG/budget; no oracle action selection"}
    if reference:
        previous = json.loads(Path(reference).read_text())
        before = (previous["models"]["after_first_10000"]["rows"] if "models" in previous
                  else previous.get("historical_510", previous)["rows"])
        if [r["trajectory_id"] for r in before] != [r["trajectory_id"] for r in rows]:
            raise ValueError("Historical retention task identities changed")
        result["old_success_lost"] = sum(a["success"] and not b["success"] for a, b in zip(before, rows))
        result["new_success_gained"] = sum(not a["success"] and b["success"] for a, b in zip(before, rows))
    return result


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
    trajectory = {"forks": 0, "shortest_choices": 0, "reachable_choices": 0,
                  "complete_candidate_relations": 0, "finite_at_or_above_original_cap": 0,
                  "cap_conflict_forks": 0, "cap_conflict_reachable_choices": 0,
                  "distance_bins": {}}

    class AuditedScorer:
        """Labels are computed after Q scores; they never filter/select actions."""
        def __init__(self, source):
            self.source = source

        def scores(self, masks, goal):
            scores = scorer.scores(masks, goal)
            truth = oracle.distance(self.source, goal)
            labels = np.asarray([oracle.distance(mask, goal) for mask in masks])
            choice = int(scores.argmin())
            finite = labels[labels >= 0]
            label_cap = getattr(scorer, "label_cap", scorer.model.cap)
            conflict = bool(len(finite) and np.any(labels < 0) and finite.min() >= label_cap)
            trajectory["forks"] += 1
            trajectory["shortest_choices"] += int(labels[choice] == truth - 1)
            trajectory["reachable_choices"] += int(labels[choice] >= 0)
            trajectory["complete_candidate_relations"] += len(labels)
            trajectory["finite_at_or_above_original_cap"] += int((labels >= scorer.model.cap).sum())
            trajectory["cap_conflict_forks"] += conflict
            trajectory["cap_conflict_reachable_choices"] += conflict and bool(labels[choice] >= 0)
            key = str(truth) if truth <= 2 else "3-4" if truth <= 4 else "5-8" if truth <= 8 else "9+"
            bucket = trajectory["distance_bins"].setdefault(key, {"forks": 0, "shortest": 0, "reachable": 0})
            bucket["forks"] += 1
            bucket["shortest"] += int(labels[choice] == truth - 1)
            bucket["reachable"] += int(labels[choice] >= 0)
            self.source = masks[choice]
            return scores
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
            cap = getattr(scorer, "label_cap", scorer.model.cap)
            group["forks"] += 1
            group["shortest_choices"] += int(distances[int(scores.argmin())] == truth - 1)
            group["candidate_count"] += len(candidates)
            group["absolute_error_sum"] += float(np.abs(scores - np.where(distances < 0, cap, distances)).sum())
        outcome = trace(AuditedScorer(source), source, goal, oracle)
        group["tasks"] += 1
        group["reached"] += outcome["outcome"] == "reached"
        results.append({**row, **outcome})
        if (index + 1) % 100 == 0:
            print(json.dumps({"evaluated": index + 1, "total": len(rows)}), flush=True)
    return results, bins, trajectory


def run(config, checkpoint, out, device, source_commit):
    torch.set_num_threads(2)
    baseline = json.loads(Path(config["baseline_cases"]).read_text())["rows"]
    roots = json.loads((Path(config["graph_data"]) / "roots.json").read_text())
    oracle = DistanceOracle(cache_limit=config["evaluation"]["oracle_cache_limit"],
                            seconds=config["evaluation"]["oracle_seconds"])
    scorer = QScorer(checkpoint, device)
    evaluated, bins, unseen_trajectory = evaluate_rows(baseline, scorer, oracle)
    train = train_cases(roots, config["evaluation"]["train_roots"], config["training"]["seed"] + 1)
    train_rows, train_bins, train_trajectory = evaluate_rows(train, scorer, oracle)
    parent_file = out.parent / "train_parent_cases.json"
    if parent_file.exists():
        train_parent = json.loads(parent_file.read_text())
    else:
        train_parent, _, _ = evaluate_rows(train, QScorer(config["parent_checkpoint"], device), oracle)
        parent_file.parent.mkdir(parents=True, exist_ok=True)
        parent_file.write_text(json.dumps(train_parent) + "\n")
    result = {"analysis_commit": source_commit, "checkpoint": str(checkpoint),
              "unseen_roots": sections(evaluated), "matched_baseline": matched(baseline, evaluated),
              "train_roots": sections(train_rows), "matched_train_parent": matched(train_parent, train_rows),
              "distance_bins_unseen": bins,
              "distance_bins_train": train_bins, "cycles": 0, "illegal_actions": 0,
              "trajectory_forks_unseen": unseen_trajectory, "trajectory_forks_train": train_trajectory,
              "policy": "original_Q_argmin_all_legal_actions_no_search_or_oracle_filters",
              "rows": evaluated, "train_rows": train_rows}
    result["distance_goal_success"] = distance_goal_success(evaluated)
    settings = config["evaluation"]
    if settings.get("intermediate_6_8_cases"):
        medium = json.loads(Path(settings["intermediate_6_8_cases"]).read_text())["rows"]
        medium_rows, _, _ = evaluate_rows(medium, scorer, oracle)
        result["intermediate_6_8_rows"] = medium_rows
        result["distance_goal_success"] = distance_goal_success(evaluated + medium_rows)
    if settings.get("clear_6_8_cases"):
        residual = json.loads(Path(settings["clear_6_8_cases"]).read_text())["rows"]
        clear_rows, _, _ = evaluate_rows(residual, scorer, oracle)
        result["clear_6_8_rows"] = clear_rows
        result["distance_goal_success"] = distance_goal_success(
            evaluated + result.get("intermediate_6_8_rows", []) + clear_rows)
    if settings.get("historical_510_manifest"):
        result["historical_retention"] = historical_retention(settings["historical_510_manifest"], checkpoint,
                                                             settings.get("historical_510_reference"))
    if settings.get("minimum_joint_success"):
        result["joint_quality"] = {"clear": result["unseen_roots"]["clear"]["success"],
            "nonempty_6_8": result["distance_goal_success"]["6-8_nonempty"]["success"],
            "nonempty_9_13": result["distance_goal_success"]["9-13_nonempty"]["success"],
            "historical_510": result["historical_retention"]["success"],
            "historical_6_8": result["historical_retention"]["6-8_success"]}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    # Preserve the accepted parent unless both core unseen-goal scores hold.
    accepted = all(result["matched_baseline"][key]["net"] >= 0 for key in ("clear", "intermediate"))
    selection = out.parent / "selection.json"
    score = sum(section["success"] for section in result["unseen_roots"].values())
    previous = json.loads(selection.read_text()) if selection.exists() else {"success": 245}
    if "joint_quality" in result:
        quality = result["joint_quality"]
        floors = settings["minimum_joint_success"]
        accepted = accepted and all(quality[k] >= v for k, v in floors.items())
        accepted = accepted and all(quality[k] >= v for k, v in previous.get("joint_quality", {}).items())
        score = sum(quality.values())
        # The old selector's score includes trivial one-step tasks; do not compare unlike sums.
        if "joint_quality" not in previous:
            previous = {**previous, "success": sum(floors.values())}
    if "unseen_roots" in previous:
        accepted = accepted and all(result["unseen_roots"][key]["success"] >=
                                    previous["unseen_roots"][key]["success"]
                                    for key in ("clear", "intermediate"))
    if accepted and score > previous["success"]:
        selected = {"success": score, "checkpoint": str(checkpoint), "evaluation": str(out),
                    "unseen_roots": result["unseen_roots"]}
        if "joint_quality" in result:
            selected["joint_quality"] = result["joint_quality"]
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
