"""Graph action semantics and actual state transitions, separate from Q/V."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np


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
