"""Training-only wrong forks, basin-weighted exits and exact multistep closure."""
import json
import numpy as np
import torch

from .graph_data import GraphData
from .checkpoint_audit import cycle_ranking


class FailureGraphData(GraphData):
    def __init__(self, env, data, options, rng, out, repair):
        super().__init__(env, data, options, rng, out, repair["structure_cache"])
        self.repair = repair
        self.out = out
        self.repair_restore = self.repair_middle = None
        self.restore_weights = None
        self.repair_seen = {key: set() for key in ("huber", "forks", "candidates")}
        self.repair_draws = {key: 0 for key in ("huber", "forks", "candidates", "optimal_transitions")}
        self.repair_distances = {key: np.zeros(env.diameter + 1, dtype=np.int64)
                                 for key in ("huber", "forks")}
        self.mine_round = 0

    @torch.no_grad()
    def mine(self, model, features, state):
        encoded = torch.cat([model.encode(part) for part in features.split(8192)])
        known = np.unique(self.data["tasks_train"][:, 1])
        fresh = self.rng.choice(self.sources, self.repair["fresh_goals_per_refresh"], replace=False)
        goals = np.setdiff1d(np.unique(np.r_[known, fresh, 0]), self.protected)
        middle = []
        legal = self.env.next >= 0
        children = np.maximum(self.env.next, 0)
        for goal in goals:
            height = model.distance(encoded, encoded[int(goal)]).cpu().numpy()
            chosen = self.env.next[np.arange(self.env.size),
                                   np.where(legal, height[children], np.inf).argmin(-1)]
            truth = self.env.distance(np.arange(self.env.size), int(goal))
            wrong = self.allowed & (truth > 0) & (self.env.distance(chosen, int(goal)) != truth - 1)
            if int(goal) == 0:
                parents = np.flatnonzero(wrong)
                if not len(parents):
                    parents = self.parents[self.parents != 0]
                self.repair_restore = np.column_stack((parents, np.zeros(len(parents), dtype=int)))
                weights = np.ones(self.env.size)
                for cycle in cycle_ranking(self.env, state):
                    for star in cycle["stars"]:
                        weights[star["parent"]] += np.log1p(cycle["basin_states"])
                self.restore_weights = weights[parents] / weights[parents].sum()
                continue
            # Equal quotas per distance and branch degree avoid overwhelming
            # the replay pool with the most common long or easy contexts.
            degree = legal.sum(-1)
            for distance in range(1, self.env.diameter + 1):
                for branches in np.unique(degree):
                    pool = np.flatnonzero(wrong & (truth == distance) & (degree == branches))
                    if len(pool):
                        take = self.rng.choice(pool, min(len(pool), self.repair["errors_per_distance_degree"]),
                                               replace=False)
                        middle.append(np.column_stack((take, np.full(len(take), goal))))
        self.repair_middle = (np.concatenate(middle) if middle else
                              np.column_stack((self.parents[:1], self.sources[-1:])))
        self.mine_round += 1
        result = {"mine_round": self.mine_round, "training_goals": goals.tolist(),
                  "protected_goals": self.protected.tolist(), "restoration_error_contexts": len(self.repair_restore),
                  "intermediate_error_contexts": len(self.repair_middle),
                  "scope": "Exact training-only wrong action rankings; full candidate sets remain unchanged; no test failures enter replay"}
        (self.out / f"failure_pool_{self.mine_round:04d}.json").write_text(json.dumps(result, indent=2) + "\n")

    def repair_rows(self, count, ranking):
        restore = round(count * self.repair["restoration_fraction"])
        a = self.repair_restore[self.rng.choice(len(self.repair_restore), restore, p=self.restore_weights)]
        b = self.repair_middle[self.rng.integers(len(self.repair_middle), size=count - restore)]
        rows = np.concatenate((a, b)).copy()
        levels = np.asarray(self.repair["descendant_steps"])
        steps = levels[self.rng.integers(len(levels), size=count)]
        for depth in range(int(steps.max(initial=0))):
            children = self.env.next[rows[:, 0]]
            distances = self.env.distance(rows[:, 0], rows[:, 1])
            labels = self.env.distance(np.maximum(children, 0), rows[:, 1, None])
            good = (children >= 0) & (labels == distances[:, None] - 1)
            good &= self.allowed[np.maximum(children, 0)] & (children != rows[:, 1, None])
            active = (depth < steps) & (distances > 1) & good.any(-1)
            random = self.rng.random(good.shape)
            random[~good] = -1
            rows[active, 0] = children[active, random[active].argmax(-1)]
            self.repair_draws["optimal_transitions"] += int(active.sum())
        key = "forks" if ranking else "huber"
        self.repair_draws[key] += count
        self.repair_seen[key].update((rows[:, 0].astype(np.int64) * self.env.size + rows[:, 1]).tolist())
        self.repair_distances[key] += np.bincount(self.env.distance(rows[:, 0], rows[:, 1]),
                                                minlength=self.env.diameter + 1)
        if ranking:
            children = self.env.next[rows[:, 0]]
            keys = children.astype(np.int64) * self.env.size + rows[:, 1, None]
            self.repair_seen["candidates"].update(keys[children >= 0].tolist())
            self.repair_draws["candidates"] += int((children >= 0).sum())
        return rows

    def graph(self, count, ranking):
        take = round(count * self.repair["graph_fraction"])
        return np.concatenate((super().graph(count - take, ranking), self.repair_rows(take, ranking)))

    def state_dict(self):
        result = super().state_dict()
        result["failure_repair"] = {"restore": self.repair_restore, "middle": self.repair_middle,
            "weights": self.restore_weights, "seen": self.repair_seen, "draws": self.repair_draws,
            "distances": self.repair_distances, "mine_round": self.mine_round}
        return result

    def load_state_dict(self, state):
        super().load_state_dict(state)
        if "failure_repair" in state:
            repair = state["failure_repair"]
            self.repair_restore, self.repair_middle = repair["restore"], repair["middle"]
            self.restore_weights, self.repair_seen = repair["weights"], repair["seen"]
            self.repair_draws, self.repair_distances = repair["draws"], repair["distances"]
            self.mine_round = repair["mine_round"]

    def report(self):
        result = super().report()
        result["failure_repair"] = {"draws": self.repair_draws,
            "distinct": {key: len(value) for key, value in self.repair_seen.items()},
            "distance_draws": {key: value.tolist() for key, value in self.repair_distances.items()},
            "mine_round": self.mine_round,
            "scope": "Global single sampler across replicas; exact distinct replay relations, overlapping original streams; candidate appearance is not nonzero score gradient"}
        return result
