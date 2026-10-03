"""Distance, relationship and control metrics; no model/environment execution."""

from collections import Counter
import math
import re

from .relations import score_relationships, parse_ranking
from .evaluate_blocks import parse_control


def numeric_distances(answer, count):
    current = re.findall(r"(?m)^Current map distance to goal: ([^\n]+)\.$", answer)
    candidates = re.findall(r"(?m)^Candidate map distances to goal: ([^\n]+)\.$", answer)
    if len(current) != 1 or len(candidates) != 1:
        return None
    try:
        parsed = {}
        for item in candidates[0].split(";"):
            key, value = item.strip().split(":")
            key = int(key)
            if key in parsed:
                return None
            parsed[key] = float(value)
        if set(parsed) != set(range(1, count + 1)):
            return None
        values = [float(current[0]), *(parsed[i] for i in range(1, count + 1))]
        return values if all(math.isfinite(v) and v >= 0 for v in values) else None
    except (ValueError, OverflowError):
        return None


def summary_correct(answer, expected):
    patterns = (r"(?m)^Executed actions: (.*)$", r"(?m)^Summary: (.*)$")
    return all(re.findall(p, answer) == re.findall(p, expected) and len(re.findall(p, answer)) == 1
               for p in patterns)


def score_turn(step, answer, expected_terminal=None):
    try:
        chosen = parse_control(answer)
        valid_control = True
    except ValueError:
        chosen, valid_control = None, False
    result = dict(done=step.done, candidates=len(step.candidate_actions), valid_control=valid_control,
        premature_done=valid_control and chosen is None and not step.done,
        failed_to_stop=step.done and not (valid_control and chosen is None),
        legal_action=not step.done and valid_control and chosen is not None and
                     1 <= chosen <= len(step.candidate_actions),
        summary_correct=step.done and expected_terminal is not None and summary_correct(answer, expected_terminal))
    if not step.done and step.candidate_actions:
        result.update(score_relationships(step, answer))
        ranks = parse_ranking(answer, len(step.candidate_actions))
        direction_correct = 0
        if ranks is not None:
            for i, d in enumerate(step.candidate_map_distances, 1):
                actual = 0 if math.isclose(d, step.current_map_distance, rel_tol=1e-10, abs_tol=1e-12) else (-1 if d < step.current_map_distance else 1)
                direction_correct += (ranks[i] > ranks[0]) - (ranks[i] < ranks[0]) == actual
        result.update(current_relation_correct=direction_correct, current_relation_total=len(step.candidate_actions))
        distances = numeric_distances(answer, len(step.candidate_actions))
        truth = [step.current_map_distance, *step.candidate_map_distances]
        result.update(distance_valid=distances is not None, distance_expected=len(truth),
                      distance_count=len(truth) if distances else 0,
                      distance_absolute_error=sum(abs(x - y) for x, y in zip(distances, truth)) if distances else 0.)
    return result


def summarize_turns(rows):
    counts = Counter(turns=len(rows), terminal_turns=sum(r["done"] for r in rows))
    metrics = ("valid_control", "premature_done", "failed_to_stop", "legal_action", "summary_correct",
               "valid_ranking", "exact_ranking", "closest_candidate_set_exact", "action_map_minimum",
               "action_follows_ranking", "distance_valid", "pairwise_correct", "pairwise_total",
               "distance_count", "distance_expected", "distance_absolute_error", "generated_tokens", "seconds",
               "current_relation_correct", "current_relation_total", "action_environment_shortest")
    for r in rows:
        for key in metrics:
            counts[key] += r.get(key, 0)
    decisions = len(rows) - counts["terminal_turns"]
    result = dict(counts, decision_turns=decisions)
    for key in ("exact_ranking", "closest_candidate_set_exact", "action_map_minimum", "legal_action", "premature_done"):
        result[key + "_rate"] = counts[key] / decisions if decisions else None
    result["distance_mae_on_valid"] = counts["distance_absolute_error"] / counts["distance_count"] if counts["distance_count"] else None
    result["distance_invalid_turn_rate"] = 1 - counts["distance_valid"] / decisions if decisions else None
    result["failed_to_stop_rate"] = counts["failed_to_stop"] / counts["terminal_turns"] if counts["terminal_turns"] else None
    result["summary_correct_rate"] = counts["summary_correct"] / counts["terminal_turns"] if counts["terminal_turns"] else None
    return result
