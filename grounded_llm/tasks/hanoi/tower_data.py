"""Data-only replay of complete, simultaneously peg-symmetric tower relations."""
import numpy as np
from grounded_llm.core.qmap.sampling import CoverageSampler


class TowerData:
    def __init__(self, env, data, rng):
        towers = np.array([0, (env.size - 1) // 2, env.size - 1])
        self.pairs = np.flatnonzero(np.isin(data["pairs_train"][:, :2], towers).any(-1))
        self.tasks = np.flatnonzero(np.isin(data["tasks_train"][:, 1], towers))
        self.ps = CoverageSampler(data["pairs_train"][self.pairs, 2], rng, len(self.pairs), 1.)
        rows = data["tasks_train"][self.tasks]
        self.rs = CoverageSampler(env.distance(rows[:, 0], rows[:, 1]), rng, len(self.tasks), 1.)

    def draw(self, count, ranking):
        return self.tasks[self.rs.draw(count)] if ranking else self.pairs[self.ps.draw(count)]

    def report(self):
        return {"tower_pair_passes": self.ps.passes, "tower_decision_passes": self.rs.passes}
