"""Score a generated distance ranking against the frozen map geometry."""

from itertools import combinations
import re
import math
from collections import defaultdict

import numpy as np

from .targets import reported_candidates


RANKING_PREFIX = "Map-distance ranking to the goal, closest to farthest: "
MEETING_METRICS = ('ndcg_inverse_rank_at_10', 'ndcg_minmax_distance_at_10',
                   'ndcg_inverse_rank_at_1', 'ndcg_minmax_distance_at_1')


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


def rank_grade_gains(distances, k=10):
    """Top-K ordinal grades K..1, others zero; true ties share mean slot grade."""
    k = min(k, len(distances))
    groups = []
    for i in sorted(range(len(distances)), key=lambda j: (distances[j], j)):
        if groups and math.isclose(distances[i], distances[groups[-1][0]], rel_tol=1e-10, abs_tol=1e-12):
            groups[-1].append(i)
        else:
            groups.append([i])
    gains = [0.0] * len(distances)
    position = 1
    for group in groups:
        grade = sum(max(k + 1 - p, 0) for p in range(position, position + len(group))) / len(group)
        for i in group:
            gains[i] = grade
        position += len(group)
    return gains


def inverse_rank_gains(distances):
    """Reciprocal true positions; equal distances share mean occupied gain."""
    ordered = sorted(range(len(distances)), key=lambda i: distances[i])
    gains = [0.0] * len(distances)
    offset = 0
    while offset < len(ordered):
        stop = offset + 1
        while stop < len(ordered) and math.isclose(distances[ordered[offset]],
                distances[ordered[stop]], rel_tol=1e-10, abs_tol=1e-12):
            stop += 1
        value = sum(1.0 / (i + 1) for i in range(offset, stop)) / (stop - offset)
        for i in ordered[offset:stop]:
            gains[i] = value
        offset = stop
    return gains


def ndcg(distances, ranks, k=10, gain='inverse_distance', discount='log2'):
    """Linear gains, average predicted ties, and zero gain for missing slots."""
    assert all(math.isfinite(d) and d >= 0 for d in distances)
    assert gain in ('inverse_distance', 'rank_grade', 'inverse_rank', 'reciprocal_distance',
                    'minmax_distance')
    if gain == 'rank_grade':
        gains = rank_grade_gains(distances, k)
    elif gain == 'inverse_rank':
        gains = inverse_rank_gains(distances)
    elif gain == 'reciprocal_distance':
        gains = [1.0 / max(d, 1e-12) for d in distances]
    elif gain == 'minmax_distance':
        if not distances:
            return None, None
        nearest, farthest = min(distances), max(distances)
        # Equal distances carry no ordering information; count these states
        # separately when reporting and give every candidate equal gain.
        gains = ([1.0] * len(distances) if math.isclose(nearest, farthest,
                 rel_tol=1e-10, abs_tol=1e-12) else
                 [(farthest - d) / (farthest - nearest) for d in distances])
    else:
        gains = [1.0 / (1.0 + d) for d in distances]
    k = min(k, len(gains))
    if not k:
        return None, None
    assert discount in ('log2', 'reciprocal_rank')
    weights = [1 / (i + 1) if discount == 'reciprocal_rank' else 1 / math.log2(i + 2) for i in range(k)]
    ideal = sum(g * w for g, w in zip(sorted(gains, reverse=True), weights))
    chance = sum(gains) / len(gains) * sum(weights) / ideal
    if ranks is None:
        return 0.0, chance
    groups = defaultdict(list)
    for candidate, rank in ranks.items():
        if candidate:
            assert 1 <= candidate <= len(gains)
            groups[rank].append(gains[candidate - 1])
    dcg, position = 0.0, 0
    for rank in sorted(groups):
        values = groups[rank]
        dcg += sum(values) / len(values) * sum(weights[position:min(position+len(values), k)])
        position += len(values)
        if position >= k:
            break
    value = dcg / ideal
    assert -1e-12 <= value <= 1+1e-12
    return min(1.0, max(0.0, value)), chance


def meeting_ranking_scores(distances, ranks):
    """Official gains: reciprocal true position and per-state normalized distance."""
    return {name: ndcg(distances, ranks, k=k, gain=gain)[0]
            for name, k, gain in zip(MEETING_METRICS, (10, 10, 1, 1),
                ('inverse_rank', 'minmax_distance', 'inverse_rank', 'minmax_distance'))}
