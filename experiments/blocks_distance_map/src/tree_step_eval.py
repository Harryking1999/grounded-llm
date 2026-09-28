"""Audit reachable and shortest-path choices made by a frozen Q scorer."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

import numpy as np

from .oracle import DistanceOracle, successors
from .tree_supervision_eval import QScorer


def inspect_tasks(scorer, tasks, oracle):
    totals = Counter()
    by_step = defaultdict(Counter)
    by_true_length = defaultdict(Counter)
    for source, goal, true_length in tasks:
        by_true_length[true_length]["tasks"] += 1
        state = source
        for step in range(1, 13):
            if state == goal:
                totals["reached_tasks"] += 1
                by_true_length[true_length]["reached"] += 1
                break
            current_distance = oracle.distance(state, goal)
            if current_distance < 0:
                raise AssertionError("A rollout continued after an unreachable action")
            children = list(dict.fromkeys(child for _, child in successors(state)))
            if not children:
                totals["dead_end_tasks"] += 1
                break
            choice = children[int(np.argmin(scorer.scores(children, goal)))]
            next_distance = oracle.distance(choice, goal)
            reachable = next_distance >= 0
            shortest = next_distance == current_distance - 1
            totals["decisions"] += 1
            totals["reachable_choices"] += reachable
            totals["shortest_choices"] += shortest
            by_step[step]["decisions"] += 1
            by_step[step]["reachable_choices"] += reachable
            by_step[step]["shortest_choices"] += shortest
            if not reachable:
                totals["unreachable_tasks"] += 1
                break
            state = choice
        else:
            totals["step_limit_tasks"] += 1
    return {
        "tasks": len(tasks),
        "totals": dict(totals),
        "reachable_choice_accuracy": totals["reachable_choices"] / totals["decisions"],
        "shortest_choice_accuracy": totals["shortest_choices"] / totals["decisions"],
        "by_step": {str(k): dict(v) for k, v in sorted(by_step.items())},
        "by_true_length": {str(k): dict(v) for k, v in sorted(by_true_length.items())},
    }


def evaluate(comparison_path, clear_path, checkpoint, out, device="cuda:0"):
    comparison = json.loads(Path(comparison_path).read_text(encoding="utf-8"))
    clear = json.loads(Path(clear_path).read_text(encoding="utf-8"))
    scorer = QScorer(checkpoint, device)
    oracle = DistanceOracle(cache_limit=2_000_000, seconds=3600)
    nonempty_tasks = [(int(row["source"]), int(row["goal"]), int(row["true_steps"]))
                      for row in comparison["task_rows"]]
    clear_tasks = [(int(row["source"]), 0, int(row["true_steps"]))
                   for row in clear["rows"]]
    result = {"scorer": scorer.identity,
              "nonempty": inspect_tasks(scorer, nonempty_tasks, oracle),
              "clear": inspect_tasks(scorer, clear_tasks, oracle)}
    if result["nonempty"]["totals"]["reached_tasks"] != comparison["results"]["tree"]["all"]["outcomes"]["reached"]:
        raise AssertionError("Nonempty task outcomes changed")
    if result["clear"]["totals"]["reached_tasks"] != clear["summary"]["outcomes"]["reached"]:
        raise AssertionError("Clear task outcomes changed")
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({name: {key: block[key] for key in
                             ("tasks", "totals", "reachable_choice_accuracy",
                              "shortest_choice_accuracy")}
                      for name, block in result.items() if name != "scorer"}, ensure_ascii=False))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--clear", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    evaluate(args.comparison, args.clear, args.checkpoint, args.out, args.device)


if __name__ == "__main__":
    main()
