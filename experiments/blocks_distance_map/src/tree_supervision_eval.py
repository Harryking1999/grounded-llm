"""Frozen-Q rollout on fresh official boards beyond the previously used OOD split."""
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
import torch

from experiments.gcml_counterexamples.src import blocks
from .hardbranch_failure import summarize, trace
from .model import BoardEncoder, board_bits, distance
from .multiboard_data import official_boards, sample_paths
from .oracle import DistanceOracle


def fresh_tasks(data, contract, official_path, board_count, tasks_per_board, seed,
                skip_fresh_boards=0):
    train_count = len(data["train_board_rows"])
    old_ood_count = len(data["ood_board_rows"])
    _, ood = official_boards(official_path, contract["board_seed"], train_count,
                             old_ood_count + skip_fresh_boards + board_count)
    if not np.array_equal([board[0] for board in ood[:old_ood_count]],
                          data["ood_board_rows"]):
        raise ValueError("The historical OOD board split changed")
    fresh = ood[old_ood_count + skip_fresh_boards:]
    known = {int(mask) for mask in data["states"]}
    rng = np.random.default_rng(seed)
    oracle = DistanceOracle(cache_limit=2_000_000, seconds=3600)
    tasks = []
    for board in fresh:
        candidates = {}
        for path in sample_paths(board, 12, rng):
            for start_index, source in enumerate(path[:-2]):
                if source in known:
                    continue
                for goal in path[start_index + 2:start_index + 6]:
                    if not goal or goal in known:
                        continue
                    steps = oracle.distance(source, goal)
                    if steps >= 2:
                        candidates[(source, goal)] = (int(board[0]), source, goal, steps,
                                                      not blocks.locally_supported(goal))
        isolated = [item for item in candidates.values() if item[4]]
        ordinary = [item for item in candidates.values() if not item[4]]
        rng.shuffle(isolated)
        rng.shuffle(ordinary)
        selected = isolated[:tasks_per_board // 2] + ordinary[:tasks_per_board -
                                                               tasks_per_board // 2]
        if len(selected) < tasks_per_board:
            remainder = [item for item in isolated + ordinary if item not in selected]
            selected.extend(remainder[:tasks_per_board - len(selected)])
        if len(selected) != tasks_per_board:
            raise ValueError(f"Board row {board[0]} has only {len(selected)} unseen tasks")
        tasks.extend(selected)
    return tasks, oracle


class QScorer:
    def __init__(self, checkpoint, device):
        saved = torch.load(checkpoint, map_location=device, weights_only=True)
        config = saved["config"]
        self.model = BoardEncoder(config["model"]["state_dim"],
                                  config["model"]["encoder_hidden_dim"], saved["cap"],
                                  saved["metric"]).to(device)
        self.model.load_state_dict(saved["model"])
        self.model.eval()
        self.metric = saved["metric"]
        self.device = device
        self.cache = {}
        self.identity = {"checkpoint": str(checkpoint),
                         "source_commit": saved["source_commit"], "step": saved["step"]}

    @torch.no_grad()
    def scores(self, masks, goal):
        missing = [mask for mask in dict.fromkeys([*masks, goal]) if mask not in self.cache]
        for start in range(0, len(missing), 2048):
            chunk = missing[start:start + 2048]
            values = self.model.encode(board_bits(chunk).to(self.device))
            self.cache.update(zip(chunk, values.split(1)))
        children = torch.cat([self.cache[mask] for mask in masks])
        target = self.cache[goal].expand_as(children)
        return distance(children, target, self.metric).cpu().numpy()


class AreaScorer:
    identity = {"rule": "fewest remaining occupied cells among all legal successors"}

    def scores(self, masks, goal):
        return np.asarray([mask.bit_count() for mask in masks], dtype=np.float32)


def evaluate(data_path, contract_path, official_path, checkpoint_specs, out,
             board_count=200, tasks_per_board=5, seed=20261009, device="cuda:0",
             analysis_commit=None, skip_fresh_boards=0):
    data = dict(np.load(data_path, allow_pickle=False))
    contract = json.loads(Path(contract_path).read_text())
    tasks, oracle = fresh_tasks(data, contract, official_path, board_count,
                                tasks_per_board, seed, skip_fresh_boards)
    scorers = {"area": AreaScorer()}
    for name, checkpoint in checkpoint_specs:
        if name in scorers:
            raise ValueError(f"Duplicate scorer: {name}")
        scorers[name] = QScorer(checkpoint, device)
    result = {"analysis_commit": analysis_commit, "boards": board_count,
              "skip_fresh_boards": skip_fresh_boards,
              "tasks": len(tasks), "tasks_per_board": tasks_per_board,
              "seed": seed, "board_rows": [int(board) for board in dict.fromkeys(
                  task[0] for task in tasks)],
              "task_rows": [{"board_row": int(board), "source": str(source),
                             "goal": str(goal), "true_steps": int(steps),
                             "isolated_goal": bool(isolated)}
                            for board, source, goal, steps, isolated in tasks],
              "goal_types": dict(Counter("isolated" if task[4] else "ordinary" for task in tasks)),
              "true_distances": dict(Counter(str(task[3]) for task in tasks)),
              "scorers": {name: scorer.identity for name, scorer in scorers.items()},
              "results": {}}
    outcomes = {}
    for name, scorer in scorers.items():
        traces = [trace(scorer, source, goal, steps, oracle)
                  for _, source, goal, steps, _ in tasks]
        outcomes[name] = [item["outcome"] == "reached" for item in traces]
        result["results"][name] = {
            "all": summarize(traces),
            "isolated": summarize([row for row, task in zip(traces, tasks) if task[4]]),
            "ordinary": summarize([row for row, task in zip(traces, tasks) if not task[4]])}
        print(json.dumps({"scorer": name, "all": result["results"][name]["all"]}),
              flush=True)
    result["task_outcomes"] = outcomes
    result["paired"] = {}
    names = list(outcomes)
    for index, first in enumerate(names):
        first_boards = np.asarray(outcomes[first]).reshape(board_count, tasks_per_board).sum(1)
        for second in names[index + 1:]:
            second_boards = np.asarray(outcomes[second]).reshape(
                board_count, tasks_per_board).sum(1)
            result["paired"][f"{first}_to_{second}"] = {
                "rescued": sum(not a and b for a, b in zip(outcomes[first], outcomes[second])),
                "regressed": sum(a and not b for a, b in zip(outcomes[first], outcomes[second])),
                "boards_improved": int((second_boards > first_boards).sum()),
                "boards_regressed": int((second_boards < first_boards).sum()),
                "boards_equal": int((second_boards == first_boards).sum())}
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--official", type=Path, required=True)
    parser.add_argument("--checkpoint", action="append", default=[],
                        help="One or more NAME=PATH frozen Q checkpoints")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--board-count", type=int, default=200)
    parser.add_argument("--skip-fresh-boards", type=int, default=0)
    parser.add_argument("--tasks-per-board", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20261009)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--analysis-commit")
    args = parser.parse_args()
    specs = [entry.split("=", 1) for entry in args.checkpoint]
    if any(len(spec) != 2 for spec in specs):
        parser.error("--checkpoint must be NAME=PATH")
    evaluate(args.data, args.config, args.official, specs, args.out,
             args.board_count, args.tasks_per_board, args.seed, args.device,
             args.analysis_commit, args.skip_fresh_boards)


if __name__ == "__main__":
    main()
