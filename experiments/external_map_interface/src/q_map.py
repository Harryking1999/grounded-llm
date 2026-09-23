"""Read a learned graph Q/V map without consulting shortest-path labels."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class GraphQMap:
    q: np.ndarray
    v: np.ndarray

    @classmethod
    def load(cls, path: Path, node_count: int, action_count: int) -> "GraphQMap":
        with np.load(path, allow_pickle=False) as saved:
            q = saved["q"].astype(np.float64)
            v = saved["v"].astype(np.float64)
        if q.ndim != 2 or q.shape[0] != node_count:
            raise ValueError("Q rows must match graph nodes")
        if v.shape != (action_count, q.shape[1]):
            raise ValueError("V rows must match action IDs and Q dimension")
        if not (np.isfinite(q).all() and np.isfinite(v).all()):
            raise ValueError("Q/V contains non-finite values")
        return cls(q=q, v=v)

    def candidate_distance(self, current: int, action_id: int, goal: int) -> float:
        """Distance of predicted Q_next = Q_current + V_action to Q_goal."""
        predicted = self.q[current] + self.v[action_id]
        return float(np.linalg.norm(predicted - self.q[goal]))
