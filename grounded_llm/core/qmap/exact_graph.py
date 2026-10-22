"""Exact cost-to-go for small, fully enumerated directed task graphs."""
import heapq
import numpy as np


def goal_distances(children, costs, terminal):
    reverse = [[] for _ in children]
    for source, row in enumerate(children):
        for action, target in enumerate(row):
            if target >= 0:
                cost = float(costs[source, action])
                if cost <= 0:
                    raise ValueError("Physical actions must have positive cost")
                reverse[target].append((source, cost))
    distances = np.full(len(children), np.inf, dtype=np.float32)
    queue = []
    for goal in np.flatnonzero(terminal):
        distances[goal] = 0
        heapq.heappush(queue, (0, int(goal)))
    while queue:
        distance, target = heapq.heappop(queue)
        if distance != distances[target]:
            continue
        for source, cost in reverse[target]:
            candidate = distance + cost
            if candidate < distances[source]:
                distances[source] = candidate
                heapq.heappush(queue, (candidate, source))
    return distances
