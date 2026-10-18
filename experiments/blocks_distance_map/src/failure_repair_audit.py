"""Exact cross-rank replay exposure and matched correction of old wrong forks."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .tree_supervision_eval import QScorer


def key(source, goal):
    return int(source) << 100 | int(goal)


def run(sampler_root, step, checkpoint, diagnosis, out):
    torch.set_num_threads(2)
    states = [torch.load(path, map_location="cpu", weights_only=False)
              for path in sorted(sampler_root.glob(f"sampler_{step:06d}_rank*.pt"))]
    if len(states) != 2:
        raise ValueError("Both rank samplers are required for global replay exposure")
    pairs, forks, candidates = (set().union(*(s["failure_repair"][name] for s in states))
                                for name in ("pairs_seen", "forks_seen", "candidates_seen"))
    old = json.loads(diagnosis.read_text())
    scorer = QScorer(checkpoint, "cpu")
    result = {"checkpoint": str(checkpoint), "diagnosis": str(diagnosis),
              "global_distinct_repair_huber_relations": len(pairs),
              "global_distinct_repair_forks": len(forks),
              "global_distinct_repair_candidate_relations": len(candidates),
              "ranking_score_gradient_draws": {k: sum(s["ranking_gradient_counts"][k] for s in states)
                  for k in states[0]["ranking_gradient_counts"]},
              "scope": "Exact global union across ranks and roots for the repair stream only. Candidate membership is loss exposure, not proof of nonzero parameter gradient. Score-gradient counts start at this repair phase, are draws and cover all training streams. Old phase lacks per-relation identities; its specific failure hits cannot be reconstructed."}
    for section in ("training_long", "monitor_long"):
        failures = [r["failure"] for r in old[section]["rows"] if not r["success"]]
        hits = {"old_failure_tasks": len(failures), "first_error_as_huber_exposed": 0,
                "first_error_as_complete_fork_exposed": 0, "first_error_candidate_drawn_relations": 0,
                "first_error_candidates_total": 0, "old_wrong_fork_now_reachable": 0,
                "old_wrong_fork_now_shortest": 0, "by_remaining_distance": {}}
        for failure in failures:
            k = key(failure["source"], failure["goal"])
            hits["first_error_as_huber_exposed"] += k in pairs
            hits["first_error_as_complete_fork_exposed"] += k in forks
            children = failure["candidates"]
            masks = [int(c["state"]) for c in children]
            labels = np.asarray([c["truth"] for c in children])
            scores = scorer.scores(masks, int(failure["goal"]))
            chosen = int(scores.argmin())
            reachable = bool(labels[chosen] >= 0)
            shortest = bool(labels[chosen] == failure["true_distance"] - 1)
            hits["old_wrong_fork_now_reachable"] += reachable
            hits["old_wrong_fork_now_shortest"] += shortest
            hits["first_error_candidates_total"] += len(children)
            hits["first_error_candidate_drawn_relations"] += sum(key(mask, failure["goal"]) in candidates
                                                                for mask in masks)
            bucket = hits["by_remaining_distance"].setdefault(str(failure["true_distance"]),
                       {"old_errors": 0, "now_reachable": 0, "now_shortest": 0})
            bucket["old_errors"] += 1
            bucket["now_reachable"] += reachable
            bucket["now_shortest"] += shortest
        result[section] = hits
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--sampler-root", type=Path, required=True)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--diagnosis", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    run(args.sampler_root, args.step, args.checkpoint, args.diagnosis, args.out)
