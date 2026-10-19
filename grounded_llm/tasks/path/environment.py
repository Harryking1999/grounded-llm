"""Graph action semantics and actual state transitions, separate from Q/V."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from collections import deque

def action_catalog(adj):
    if not np.array_equal(adj, adj.T) or np.any(np.diag(adj)):
        raise ValueError("Expected a simple undirected graph")
    edges = [(int(i), int(j)) for i, j in zip(*np.nonzero(np.triu(adj, 1)))]
    actions = np.array([edge for i, j in edges for edge in [(i, j), (j, i)]])
    outgoing = [np.flatnonzero(actions[:, 0] == i) for i in range(len(adj))]
    if any(len(x) == 0 for x in outgoing):
        raise ValueError("Isolated node")
    return actions, outgoing


def shortest_distances(adj):
    n = len(adj)
    distances = np.full((n, n), -1, dtype=np.int32)
    neighbors = [np.flatnonzero(row) for row in adj]
    for start in range(n):
        distances[start, start] = 0
        queue = deque([start])
        while queue:
            node = queue.popleft()
            for other in neighbors[node]:
                if distances[start, other] < 0:
                    distances[start, other] = distances[start, node] + 1
                    queue.append(other)
    if np.any(distances < 0):
        raise ValueError("Disconnected graph")
    return distances


def environment_from_suite(suite):
    """Read only topology; neither shortest paths nor reference lengths are used."""
    first = suite["cases"][0]
    adjacency = np.zeros((first["node_count"], first["node_count"]), dtype=bool)
    for source, neighbors in first["neighbors"].items():
        adjacency[int(source), neighbors] = True
    for case in suite["cases"]:
        if case["node_count"] != len(adjacency) or any(
            set(case["neighbors"][str(i)]) != set(np.flatnonzero(adjacency[i]))
            for i in range(len(adjacency))
        ):
            raise ValueError("All cases must use the same fixed graph")
    actions, _ = action_catalog(adjacency)
    return GraphEnvironment(adjacency, actions)


@dataclass(frozen=True)
class GraphEnvironment:
    adjacency: np.ndarray
    actions: np.ndarray

    @classmethod
    def load(cls, path: Path) -> "GraphEnvironment":
        # Deliberately read only these keys. graph_distances in Step 1 inputs
        # are evaluation truth and must never enter the planner interface.
        with np.load(path, allow_pickle=False) as saved:
            adjacency = saved["adjacency"].astype(bool)
            actions = saved["actions"].astype(np.int64)
        if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
            raise ValueError("Adjacency must be square")
        if actions.ndim != 2 or actions.shape[1] != 2:
            raise ValueError("Actions must be directed (source, destination) rows")
        n = len(adjacency)
        if np.any(actions < 0) or np.any(actions >= n):
            raise ValueError("Action endpoint outside graph")
        if not np.all(adjacency[actions[:, 0], actions[:, 1]]):
            raise ValueError("Action catalog disagrees with graph")
        if len(actions) != int(adjacency.sum()):
            raise ValueError("Action catalog does not cover all directed edges")
        if len({tuple(row) for row in actions.tolist()}) != len(actions):
            raise ValueError("Action catalog has duplicate edges")
        return cls(adjacency=adjacency, actions=actions)

    def legal_actions(self, current: int) -> list[int]:
        return np.flatnonzero(self.actions[:, 0] == current).tolist()

    def execute(self, current: int, action_id: int) -> int:
        if action_id not in self.legal_actions(current):
            raise ValueError("Illegal action ID for current node")
        return int(self.actions[action_id, 1])
