"""Distance, relationship and control metrics; no model/environment execution."""

from collections import Counter
import math
import re

from .relations import score_relationships, parse_ranking
from .sft import reported_candidates
from .evaluate_blocks import parse_control
from .blocks_failure import declares_no_solution
from .analyze_blocks_ranking import ndcg


def summary_correct(answer, expected):
    patterns = (r"(?m)^Executed actions: (.*)$", r"(?m)^Summary: (.*)$")
    return all(re.findall(p, answer) == re.findall(p, expected) and len(re.findall(p, answer)) == 1
               for p in patterns)


def score_turn(step, answer, expected_terminal=None, reported=None):
    try:
        chosen = parse_control(answer)
        valid_control = True
    except ValueError:
        chosen, valid_control = None, False
    no_solution = declares_no_solution(answer)
    result = dict(done=step.done, candidates=len(step.candidate_actions), valid_control=valid_control,
        premature_done=valid_control and chosen is None and not step.done and not no_solution,
        failed_to_stop=step.done and not (valid_control and chosen is None and not no_solution),
        legal_action=not step.done and valid_control and chosen is not None and
                     1 <= chosen <= len(step.candidate_actions),
        summary_correct=step.done and expected_terminal is not None and summary_correct(answer, expected_terminal))
    result["declared_no_solution"] = no_solution
    result["valid_control"] = valid_control or no_solution
    if not step.done and step.candidate_actions:
        result.update(score_relationships(step, answer, reported))
        # A narrowed answer names the nearest candidates; every rate below is
        # measured over that named set, not over every legal move.
        named = reported_candidates(step, reported)
        ranks = parse_ranking(answer, len(step.candidate_actions), reported)
        value, chance = ndcg(step.candidate_map_distances, ranks, gain='rank_grade')
        result.update(ranking_ndcg_at_10=value, random_ndcg_at_10=chance)
        direction_correct = 0
        if ranks is not None and set(ranks) == {0, *named}:
            for i in named:
                d = step.candidate_map_distances[i - 1]
                actual = 0 if math.isclose(d, step.current_map_distance, rel_tol=1e-10, abs_tol=1e-12) else (-1 if d < step.current_map_distance else 1)
                direction_correct += (ranks[i] > ranks[0]) - (ranks[i] < ranks[0]) == actual
        result.update(current_relation_correct=direction_correct, current_relation_total=len(named))
    return result


def still_solvable(row):
    """Whether this turn was a live decision the model could still have got right.

    A turn played on an already-dead board cannot be broken by any action, so it
    is not evidence about the model's choice. A terminal turn is not a choice at
    all. Neither belongs in the rate or in the floor it is read against, and the
    field's mere presence cannot be the filter: reference shards attach it to the
    stopping turn too, where it is always false, which silently deflates the rate
    against a floor that skips those rows.
    """
    if row["done"] or not row.get("candidates"):
        return False
    remaining = row.get("remaining_shortest")
    return remaining is not None and remaining >= 0


def termination_counts(rows):
    """Additive counts; all done outputs include the none+done failure reply."""
    counts = Counter(done_outputs=0, done_at_goal=0, goal_claims=0,
                     correct_goal_claims=0, goal_states=0, no_solution_states=0,
                     correct_no_solution=0)
    for row in rows:
        answer = row.get('answer', '')
        emitted = '<done/>' in answer
        try:
            claim = parse_control(answer) is None and not declares_no_solution(answer)
        except ValueError:
            claim = False
        goal = bool(row['done'])
        negative = not goal and row.get('candidates') == 0
        counts.update(done_outputs=int(emitted), done_at_goal=int(emitted and goal),
                      goal_claims=int(claim), correct_goal_claims=int(claim and goal),
                      goal_states=int(goal), no_solution_states=int(negative),
                      correct_no_solution=int(negative and declares_no_solution(answer)))
    return dict(counts)


def summarize_turns(rows):
    counts = Counter(turns=len(rows), terminal_turns=sum(r["done"] for r in rows))
    metrics = ("valid_control", "premature_done", "failed_to_stop", "legal_action", "summary_correct",
               "valid_ranking", "exact_ranking", "closest_candidate_set_exact", "action_map_minimum",
               "action_follows_ranking", "pairwise_correct", "pairwise_total",
               "generated_tokens", "seconds",
               "current_relation_correct", "current_relation_total", "action_environment_shortest",
               "action_keeps_goal_reachable", "reachable_candidates", "candidate_slots",
               "declared_no_solution", "no_solution_expected", "no_solution_correct", "false_no_solution")
    for r in rows:
        for key in metrics:
            counts[key] += r.get(key, 0)
    decisions = len(rows) - counts["terminal_turns"]
    # Reachability is reported over the solvable subset, not over every decision
    # turn: on a board the model already made dead, no action could score, and
    # counting those turns charges it for a loss that happened earlier. The
    # all-turn counts stay in the summary unchanged.
    solvable = [r for r in rows if still_solvable(r)]
    counts["solvable_decisions"] = len(solvable)
    result = dict(counts, decision_turns=decisions)
    ranked = [r for r in rows if not r['done'] and r.get('candidates') and r.get('ranking_ndcg_at_10') is not None]
    result['ranking_ndcg_turns'] = len(ranked)
    result['ranking_ndcg_at_10'] = (sum(r['ranking_ndcg_at_10'] for r in ranked) / len(ranked)
                                     if ranked else None)
    result.update(termination_counts(rows))
    for name, numerator, denominator in (
            ('done_goal_precision', 'done_at_goal', 'done_outputs'),
            ('goal_claim_precision', 'correct_goal_claims', 'goal_claims'),
            ('goal_stop_recall', 'correct_goal_claims', 'goal_states')):
        result[name] = result[numerator] / result[denominator] if result[denominator] else None
    negatives = counts["no_solution_expected"]
    result["no_solution_recall"] = counts["no_solution_correct"] / negatives if negatives else None
    result["false_no_solution_rate"] = counts["false_no_solution"] / (len(rows) - negatives) if len(rows) > negatives else None
    for key in ("exact_ranking", "closest_candidate_set_exact", "action_map_minimum", "legal_action", "premature_done"):
        result[key + "_rate"] = counts[key] / decisions if decisions else None
    result["failed_to_stop_rate"] = counts["failed_to_stop"] / counts["terminal_turns"] if counts["terminal_turns"] else None
    result["summary_correct_rate"] = counts["summary_correct"] / counts["terminal_turns"] if counts["terminal_turns"] else None
    result["action_keeps_goal_reachable_rate"] = (sum(1 for r in solvable if r.get("action_keeps_goal_reachable", False)) / len(solvable)
                                                  if solvable else None)
    # The chance floor for that same subset: the fraction of legal candidates
    # that leave the goal reachable, i.e. what uniform picking scores there.
    slots = sum(r.get("candidate_slots", 0) for r in solvable)
    result["reachable_candidate_rate"] = (sum(r.get("reachable_candidates", 0) for r in solvable) / slots if slots else None)
    return result
