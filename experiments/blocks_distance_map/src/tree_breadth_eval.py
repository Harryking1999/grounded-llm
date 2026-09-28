"""Broad, fixed multi-start/multi-goal tasks by exact shortest-path length."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

import numpy as np

from experiments.gcml_counterexamples.src import blocks
from .rollout_eval import trace
from .multiboard_data import official_boards, sample_paths
from .oracle import DistanceOracle, successors
from .tree_supervision_eval import QScorer


def task_pool(data_path, config_path, official_path, split, seed, paths_per_board):
    data = dict(np.load(data_path, allow_pickle=False))
    config = json.loads(Path(config_path).read_text())
    train_count = len(data["train_board_rows"])
    historical = len(data["ood_board_rows"])
    if split == "development":
        _, boards = official_boards(official_path, config["board_seed"],
                                     train_count, historical)
    elif split == "sealed":
        _, all_ood = official_boards(official_path, config["board_seed"],
                                     train_count, historical + 400)
        boards = all_ood[historical + 200:]
    else:
        raise ValueError(split)
    if [board[0] for board in boards[:historical if split == "development" else 0]] != (
            data["ood_board_rows"].tolist() if split == "development" else []):
        raise ValueError("Historical board split changed")
    rng = np.random.default_rng(seed)
    oracle = DistanceOracle(cache_limit=2_000_000, seconds=3600)
    tasks = []
    for board_row, board_mask, reference in boards:
        paths = sample_paths((board_row, board_mask, reference), paths_per_board, rng)
        candidates = defaultdict(dict)
        for path in paths:
            for index, source in enumerate(path[:-1]):
                for goal in path[index + 1:]:
                    steps = oracle.distance(source, goal)
                    if not 1 <= steps <= 8:
                        continue
                    goal_type = ("empty" if goal == 0 else "isolated" if
                                 not blocks.locally_supported(goal) else "ordinary")
                    candidates[(goal_type, steps)][(source, goal)] = None
        # At most one task per goal type and true distance on each board.
        # Rotate the selected source depth across boards rather than always
        # choosing the full initial board.
        for (goal_type, steps), pairs in sorted(candidates.items()):
            choices = list(pairs)
            source, goal = choices[int(rng.integers(len(choices)))]
            tasks.append({"board_row": int(board_row), "source": str(source),
                          "goal": str(goal), "goal_type": goal_type,
                          "true_steps": int(steps),
                          "source_cells": source.bit_count(),
                          "goal_cells": goal.bit_count()})
    return tasks, oracle


def build(data_path, config_path, official_path, out, split, seed, paths_per_board):
    tasks, _ = task_pool(data_path, config_path, official_path, split, seed,
                         paths_per_board)
    counts = Counter((task["goal_type"], task["true_steps"]) for task in tasks)
    result = {"split": split, "seed": seed, "paths_per_board": paths_per_board,
              "boards": len({task["board_row"] for task in tasks}),
              "tasks": tasks,
              "counts": {f"{goal_type}:{steps}": count
                         for (goal_type, steps), count in sorted(counts.items())}}
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"split": split, "boards": result["boards"],
                      "tasks": len(tasks), "counts": result["counts"]}), flush=True)
    return result


def score(task_path, checkpoint_path, out, device="cuda:0"):
    suite = json.loads(Path(task_path).read_text())
    scorer = QScorer(checkpoint_path, device)
    oracle = DistanceOracle(cache_limit=2_000_000, seconds=3600)
    by_type_length = defaultdict(Counter)
    by_length = defaultdict(Counter)
    by_goal_type = defaultdict(Counter)
    by_source_size = defaultdict(Counter)
    first = Counter()
    first_by_type = defaultdict(Counter)
    outcomes = []
    for task in suite["tasks"]:
        source, goal, steps = int(task["source"]), int(task["goal"]), task["true_steps"]
        outcome = trace(scorer, source, goal, steps, oracle)
        success = outcome["outcome"] == "reached"
        outcomes.append(success)
        children = list(dict.fromkeys(child for _, child in successors(source)))
        chosen = children[int(np.argmin(scorer.scores(children, goal)))]
        next_steps = oracle.distance(chosen, goal)
        for record in (first, first_by_type[task["goal_type"]]):
            record["decisions"] += 1
            record["reachable"] += next_steps >= 0
            record["shortest"] += next_steps == steps - 1
        for record in (by_type_length[(task["goal_type"], steps)],
                       by_length[steps], by_goal_type[task["goal_type"]],
                       by_source_size[task["source_cells"]]):
            record["tasks"] += 1
            record["reached"] += success
    def present(records):
        return {str(key): {"tasks": count["tasks"], "reached": count["reached"],
                           "accuracy": count["reached"] / count["tasks"]}
                for key, count in sorted(records.items(), key=lambda item: str(item[0]))}
    result = {"scorer": scorer.identity, "split": suite["split"],
              "tasks": len(outcomes), "reached": int(sum(outcomes)),
              "accuracy": float(np.mean(outcomes)),
              "by_type_length": present(by_type_length),
              "by_length": present(by_length), "by_goal_type": present(by_goal_type),
              "by_source_cells": present(by_source_size),
              "first_step": dict(first),
              "first_step_by_goal_type": {key: dict(value)
                                          for key, value in sorted(first_by_type.items())},
              "outcomes": outcomes}
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"checkpoint": str(checkpoint_path), "split": suite["split"],
                      "tasks": len(outcomes), "reached": sum(outcomes),
                      "by_length": result["by_length"],
                      "by_goal_type": result["by_goal_type"]}), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--official", type=Path)
    parser.add_argument("--split", choices=("development", "sealed"))
    parser.add_argument("--seed", type=int, default=20261014)
    parser.add_argument("--paths-per-board", type=int, default=8)
    parser.add_argument("--tasks", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.checkpoint:
        score(args.tasks, args.checkpoint, args.out, args.device)
    else:
        build(args.data, args.config, args.official, args.out,
              args.split, args.seed, args.paths_per_board)


if __name__ == "__main__":
    main()
