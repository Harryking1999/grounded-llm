"""Bounded push-space A*: return a replayable walk/push witness, not minimum walks."""
from collections import deque
import heapq
from itertools import permutations
import time


def solve(env, max_expanded=200000, timeout=20):
    started = time.monotonic()
    goals = sorted(env.goals)
    distances = []
    for goal in goals:
        table, queue = {goal: 0}, deque([goal])
        while queue:
            target = queue.popleft()
            for dr, dc in env.deltas:
                box = env.adjacent(target, (-dr, -dc))
                behind = env.adjacent(box, (-dr, -dc)) if box >= 0 else -1
                if box in env.floor and behind in env.floor and box not in table:
                    table[box] = table[target] + 1
                    queue.append(box)
        distances.append(table)
    assignments = list(permutations(range(len(goals))))
    def heuristic(boxes):
        return min(sum(distances[g].get(b, float('inf')) for b, g in zip(boxes, order)) for order in assignments)
    def paths(player, boxes):
        occupied = set(boxes)
        reached, queue = {player: ''}, deque([player])
        while queue:
            cell = queue.popleft()
            for action, delta in zip(env.actions, env.deltas):
                child = env.adjacent(cell, delta)
                if child in env.floor and child not in occupied and child not in reached:
                    reached[child] = reached[cell] + action
                    queue.append(child)
        return reached
    player, boxes = env.start
    reachable = paths(player, boxes)
    key = boxes, min(reachable)
    initial_h = heuristic(boxes)
    if initial_h == float('inf'):
        return None, {'expanded': 0, 'reason': 'static_unreachable'}
    queue = [(initial_h, 0, 0, key, player)]
    best, parent, serial, expanded = {key: 0}, {}, 0, 0
    while queue:
        _, pushes, _, key, player = heapq.heappop(queue)
        if pushes != best[key]: continue
        boxes = key[0]
        if set(boxes) == env.goals:
            segments = []
            while key in parent:
                key, segment = parent[key]
                segments.append(segment)
            return list(''.join(reversed(segments))), {'expanded': expanded, 'pushes': pushes, 'seconds': time.monotonic() - started}
        expanded += 1
        if expanded > max_expanded or time.monotonic() - started > timeout:
            return None, {'expanded': expanded, 'reason': 'solver_budget'}
        reachable = paths(player, boxes)
        occupied = set(boxes)
        for box in boxes:
            for action, (dr, dc) in zip(env.actions, env.deltas):
                behind = env.adjacent(box, (-dr, -dc))
                destination = env.adjacent(box, (dr, dc))
                if behind not in reachable or destination not in env.floor or destination in occupied: continue
                moved = tuple(sorted((occupied - {box}) | {destination}))
                h = heuristic(moved)
                if h == float('inf'): continue
                child_key = moved, min(paths(box, moved))
                if pushes + 1 >= best.get(child_key, float('inf')): continue
                best[child_key] = pushes + 1
                parent[child_key] = key, reachable[behind] + action
                serial += 1
                heapq.heappush(queue, (pushes + 1 + h, pushes + 1, serial, child_key, box))
    return None, {'expanded': expanded, 'reason': 'search_exhausted'}
