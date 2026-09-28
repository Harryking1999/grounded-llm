"""Diagnose the first irreversible choice in held-out hard-branch rollouts."""
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
import torch

from experiments.gcml_counterexamples.src import blocks
from .hardbranch_eval import load_values
from .model import board_bits, distance
from .oracle import DistanceOracle, successors


class FrozenScorer:
    def __init__(self, checkpoint, data, device):
        self.saved, self.model, self.values = load_values(checkpoint, data, device)
        self.ids = {int(mask): index for index, mask in enumerate(data["states"])}
        self.extra = {}
        self.device = device

    @torch.no_grad()
    def scores(self, masks, goal):
        missing = [mask for mask in dict.fromkeys([*masks, goal])
                   if mask not in self.ids and mask not in self.extra]
        for start in range(0, len(missing), 2048):
            chunk = missing[start:start + 2048]
            encoded = self.model.encode(board_bits(chunk).to(self.device))
            self.extra.update(zip(chunk, encoded.split(1)))

        def values(items):
            return torch.cat([self.values[self.ids[mask]:self.ids[mask] + 1]
                              if mask in self.ids else self.extra[mask]
                              for mask in items], dim=0)

        return distance(values(masks), values([goal]).expand(len(masks), -1),
                        self.saved["metric"]).cpu().numpy()


def admissible(child, goal, policy):
    if policy == "pure":
        return True
    if child & goal != goal:
        return False
    return policy == "preserve" or blocks.locally_supported(child ^ goal)


def failure_kind(child, goal):
    if child & goal != goal:
        return "removed_goal_cell"
    if not blocks.locally_supported(child ^ goal):
        return "local_coverage_failure"
    return "globally_unreachable"


def trace(scorer, source, goal, true_distance, oracle, policy="pure"):
    state = source
    for step in range(1, 13):
        if state == goal:
            return {"outcome": "reached", "steps": step - 1}
        children = list(dict.fromkeys(child for _, child in successors(state)
                                      if admissible(child, goal, policy)))
        if not children:
            return {"outcome": "no_candidate", "step": step}
        scores = scorer.scores(children, goal)
        chosen_index = int(np.argmin(scores))
        chosen = children[chosen_index]
        if oracle.distance(chosen, goal) < 0:
            distances = [oracle.distance(child, goal) for child in children]
            reachable = np.asarray([d >= 0 for d in distances])
            assert reachable.any(), "A certified reachable parent lost all good children"
            good_scores = scores[reachable]
            return {"outcome": failure_kind(chosen, goal), "step": step,
                    "true_distance": true_distance, "candidate_count": len(children),
                    "reachable_count": int(reachable.sum()),
                    "best_good_rank": int(np.argsort(scores).tolist().index(
                        int(np.flatnonzero(reachable)[good_scores.argmin()])) + 1),
                    "chosen_score": float(scores[chosen_index]),
                    "best_good_score": float(good_scores.min()),
                    "correctable_by_local_rule": not admissible(chosen, goal, "local")}
        state = chosen
    if state == goal:
        return {"outcome": "reached", "steps": 12}
    return {"outcome": "step_limit", "step": 12}


def summarize(traces):
    outcomes = Counter(row["outcome"] for row in traces)
    first_error_steps = Counter(row.get("step") for row in traces
                                if row["outcome"] not in ("reached", "step_limit"))
    candidate_counts = [row["candidate_count"] for row in traces
                        if "candidate_count" in row]
    reachable_counts = [row["reachable_count"] for row in traces
                        if "reachable_count" in row]
    failures = [row for row in traces if "candidate_count" in row]
    return {"tasks": len(traces), "outcomes": dict(outcomes),
            "first_error_steps": dict(first_error_steps),
            "failures_correctable_by_local_rule": sum(
                row["correctable_by_local_rule"] for row in failures),
            "failed_choice_candidates_median": float(np.median(candidate_counts)) if failures else None,
            "failed_choice_reachable_median": float(np.median(reachable_counts)) if failures else None,
            "failed_choice_best_good_rank_median": float(np.median(
                [row["best_good_rank"] for row in failures])) if failures else None}


def analyze(data_path, checkpoints, count, out, device):
    data = dict(np.load(data_path, allow_pickle=False))
    states = [int(mask) for mask in data["states"]]
    scorers = {name: FrozenScorer(path, data, device) for name, path in checkpoints.items()}
    oracle = DistanceOracle(cache_limit=3_000_000, seconds=3600)
    result = {"checkpoints": {name: {"path": str(path),
                                  "source_commit": scorers[name].saved["source_commit"],
                                  "step": scorers[name].saved["step"],
                                  "finetune_source_commit": scorers[name].saved.get(
                                      "finetune_source_commit"),
                                  "finetune_step": scorers[name].saved.get("finetune_step")}
                              for name, path in checkpoints.items()}, "splits": {}}
    for split in ("ood_board", "unseen_goal"):
        cases = data[f"hard_{split}_cases"][:count]
        section = {"cases": len(cases), "models": {}, "paired": {}}
        traces = {}
        for name, scorer in scorers.items():
            traces[name] = {}
            for policy in ("pure", "preserve", "local"):
                rows = [trace(scorer, states[parent], states[goal], int(steps) + 1,
                              oracle, policy)
                        for parent, _, _, goal, steps in cases]
                traces[name][policy] = rows
                section["models"].setdefault(name, {})[policy] = summarize(rows)
        if len(scorers) == 2:
            first, second = scorers
            a = [row["outcome"] == "reached" for row in traces[first]["pure"]]
            b = [row["outcome"] == "reached" for row in traces[second]["pure"]]
            section["paired"] = {"both_success": sum(x and y for x, y in zip(a, b)),
                                 "both_fail": sum(not x and not y for x, y in zip(a, b)),
                                 "rescued": sum(not x and y for x, y in zip(a, b)),
                                 "regressed": sum(x and not y for x, y in zip(a, b))}
        result["splits"][split] = section
        print(json.dumps({split: section}), flush=True)
    if out:
        Path(out).write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--current", type=Path, required=True)
    parser.add_argument("--count", type=int, default=1000)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    analyze(args.data, {"base": args.base, "current": args.current},
            args.count, args.out, args.device)


if __name__ == "__main__":
    main()
