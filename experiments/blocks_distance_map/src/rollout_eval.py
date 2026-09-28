"""Trace frozen-Q decisions against exact goal reachability."""
from collections import Counter

import numpy as np

from experiments.gcml_counterexamples.src import blocks
from .oracle import successors


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
    failures = [row for row in traces if "candidate_count" in row]
    return {"tasks": len(traces), "outcomes": dict(outcomes),
            "first_error_steps": dict(first_error_steps),
            "failures_correctable_by_local_rule": sum(
                row["correctable_by_local_rule"] for row in failures),
            "failed_choice_candidates_median": float(np.median(
                [row["candidate_count"] for row in failures])) if failures else None,
            "failed_choice_reachable_median": float(np.median(
                [row["reachable_count"] for row in failures])) if failures else None,
            "failed_choice_best_good_rank_median": float(np.median(
                [row["best_good_rank"] for row in failures])) if failures else None}
