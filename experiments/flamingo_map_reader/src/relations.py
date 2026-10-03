"""Score a generated distance ranking against the frozen map geometry."""

from itertools import combinations
import re

import numpy as np

from .sft import reported_candidates


RANKING_PREFIX = "Map-distance ranking to the goal, closest to farthest: "


def first_ranked_candidate(answer: str, candidate_count: int) -> int | None:
    """Read the first candidate in the ranking line, even if the line is incomplete."""
    lines = [line.strip() for line in answer.splitlines()
             if line.strip().startswith(RANKING_PREFIX)]
    if len(lines) != 1:
        return None
    expression = lines[0][len(RANKING_PREFIX):]
    match = re.search(r"(?<![0-9])([1-9][0-9]*)(?![0-9])", expression)
    if match is None:
        return None
    candidate = int(match.group(1))
    return candidate if candidate <= candidate_count else None


def parse_ranking(answer: str, candidate_count: int,
                  reported: int | None = None) -> dict[int, int] | None:
    lines = [line.strip() for line in answer.splitlines()
             if line.strip().startswith(RANKING_PREFIX)]
    if len(lines) != 1 or candidate_count < 1 or not lines[0].endswith("."):
        return None
    expression = lines[0][len(RANKING_PREFIX):-1].strip()
    ranks = {}
    for rank, group in enumerate(re.split(r"\s*<\s*", expression)):
        for name in re.split(r"\s*=\s*", group):
            name = name.strip()
            if not re.fullmatch(r"current|[1-9][0-9]*", name):
                return None
            local_id = 0 if name == "current" else int(name)
            if local_id in ranks or local_id > candidate_count:
                return None
            ranks[local_id] = rank
    if reported is None:
        return ranks if set(ranks) == set(range(candidate_count + 1)) else None
    # A narrowed answer ranks current plus at most the reported candidates.
    return ranks if 0 in ranks and len(ranks) <= reported + 1 else None


def score_relationships(step, answer: str, reported: int | None = None) -> dict:
    """Include current in all pairwise checks; invalid ranks count as errors."""
    count = len(step.candidate_actions)
    if step.done or count < 1:
        raise ValueError("Relationship scoring needs a nonterminal map step")
    named = reported_candidates(step, reported)
    items = (0, *named)
    ranks = parse_ranking(answer, count, reported)
    complete = ranks is not None and set(ranks) == set(items)
    distances = (step.current_map_distance, *step.candidate_map_distances)
    pairwise_total = len(items) * (len(items) - 1) // 2
    correct = 0
    if complete:
        for left, right in combinations(items, 2):
            actual = 0 if np.isclose(distances[left], distances[right],
                                     rtol=1e-10, atol=1e-12) else (
                         -1 if distances[left] < distances[right] else 1)
            predicted = (ranks[left] > ranks[right]) - (ranks[left] < ranks[right])
            correct += predicted == actual
    predicted_best = (set() if not complete else
        {index for index in named if ranks[index] == min(ranks[i] for i in named)})
    markers = re.findall(r"<action>\s*([0-9]+)\s*</action>|(<done/>)", answer)
    chosen = int(markers[0][0]) if len(markers) == 1 and markers[0][0] else None
    return {
        "valid_ranking": complete,
        "pairwise_correct": correct,
        "pairwise_total": pairwise_total,
        "exact_ranking": complete and correct == pairwise_total,
        # Only the named candidates were ever put to the model, so compare
        # against the minimal ones among them, not against every legal move.
        "closest_candidate_set_exact": predicted_best == (set(step.map_minimal_candidates) & set(named)),
        "action_map_minimum": chosen in step.map_minimal_candidates,
        "action_follows_ranking": chosen in predicted_best,
        "chosen_id": chosen,
    }
