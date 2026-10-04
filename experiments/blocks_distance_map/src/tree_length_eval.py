"""Evaluate one-step and six-step goals on the same sealed fresh boards."""
import argparse
import json
from pathlib import Path

import numpy as np

from .hardbranch_failure import summarize, trace
from .multiboard_data import official_boards, sample_paths
from .oracle import DistanceOracle
from .tree_step_eval import inspect_tasks
from .tree_supervision_eval import QScorer


def evaluate(data_path, config_path, official_path, comparison_path,
             checkpoint_path, output_path, device="cuda:0"):
    data = dict(np.load(data_path, allow_pickle=False))
    comparison = json.loads(Path(comparison_path).read_text())
    config = json.loads(Path(config_path).read_text())
    old_ood = len(data["ood_board_rows"])
    skip = int(comparison.get("skip_fresh_boards", 0))
    count = len(comparison["board_rows"])
    _, boards = official_boards(official_path, config["board_seed"],
                                len(data["train_board_rows"]), old_ood + skip + count)
    if [board[0] for board in boards[:old_ood]] != data["ood_board_rows"].tolist():
        raise ValueError("Historical OOD split changed")
    fresh = boards[old_ood + skip:]
    if [board[0] for board in fresh] != comparison["board_rows"]:
        raise ValueError("Length evaluation boards differ from main evaluation")
    rng = np.random.default_rng(20261012)
    oracle = DistanceOracle(cache_limit=2_000_000, seconds=3600)
    scorer = QScorer(checkpoint_path, device)
    tasks = {1: [], 6: []}
    for row, source, reference in fresh:
        paths = sample_paths((row, source, reference), 12, rng)
        one_step_goal = paths[0][1]
        tasks[1].append((row, source, one_step_goal, 1))
        six_step = None
        for path in paths:
            for start, state in enumerate(path[:-6]):
                for goal in path[start + 6:]:
                    if goal and oracle.distance(state, goal) == 6:
                        six_step = (row, state, goal, 6)
                        break
                if six_step:
                    break
            if six_step:
                break
        if six_step:
            tasks[6].append(six_step)
    result = {"scorer": scorer.identity, "board_rows": comparison["board_rows"],
              "by_true_length": {}}
    for length, rows in tasks.items():
        traces = [trace(scorer, source, goal, length, oracle)
                  for _, source, goal, _ in rows]
        result["by_true_length"][str(length)] = {
            "task_rows": [{"board_row": row, "source": str(source),
                           "goal": str(goal), "true_steps": length,
                           "outcome": outcome["outcome"]}
                          for (row, source, goal, _), outcome in zip(rows, traces)],
            "summary": summarize(traces),
            "step_choice": inspect_tasks(scorer, [(source, goal, length)
                                                  for _, source, goal, _ in rows], oracle)}
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({length: {"tasks": len(block["task_rows"]),
                               "reached": block["summary"]["outcomes"].get("reached", 0)}
                      for length, block in result["by_true_length"].items()}), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--official", type=Path, required=True)
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    evaluate(args.data, args.config, args.official, args.comparison,
             args.checkpoint, args.out, args.device)


if __name__ == "__main__":
    main()
