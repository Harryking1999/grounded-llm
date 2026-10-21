"""Certified clearing trajectories and coverage selection for blocks SFT."""
import heapq
import random
from . import rules
from .oracle import DistanceOracle, successors
from .protocol import action_key
from .sft import sorted_step


def reference_route(source, oracle):
    state = source
    actions, states = [], []
    while state:
        states.append(state)
        distance = oracle.distance(state, 0)
        if distance < 1:
            raise ValueError('Reference state is not clearable')
        for action, child in successors(state):
            if oracle.distance(child, 0) == distance - 1:
                actions.append(action_key(rules.PLACEMENTS[action][1]))
                state = child
                break
        else:
            raise ValueError('No certified shortest successor')
    return actions, states


def greedy_route(source, qmap, oracle, excluded):
    distance = oracle.distance(source, 0)
    state, ids, pairs = source, [], []
    for index in range(distance):
        if state in excluded:
            return None
        step = sorted_step(qmap, state, 0)
        if not step.map_minimal_candidates:
            return None
        chosen = step.map_minimal_candidates[0]
        _, child = step.execute(chosen)
        if oracle.distance(child, 0) != distance-index-1:
            return None
        ids.append(chosen)
        pairs.append((state, 0))
        state = child
    if state or not ids:
        return None
    return dict(start=str(source), goal='0', shortest_moves=distance,
                route_ids=ids, pairs=pairs)


def select_coverage(rows, count, seed):
    """Longest first; lazy greedy marginal coverage for distinct clearing roots."""
    if len({r['start'] for r in rows}) != len(rows):
        raise ValueError('Expected distinct initial boards')
    rng = random.Random(seed)
    pool = [(-r['shortest_moves'], -len(set(map(tuple, r['pairs']))),
             rng.random(), i) for i, r in enumerate(rows)]
    heapq.heapify(pool)
    selected, seen = [], set()
    while pool and len(selected) < count:
        length, previous, tie, i = heapq.heappop(pool)
        pairs = set(map(tuple, rows[i]['pairs']))
        current = -len(pairs-seen)
        if current != previous:
            heapq.heappush(pool, (length, current, tie, i))
            continue
        selected.append(rows[i])
        seen.update(pairs)
    return selected, seen
