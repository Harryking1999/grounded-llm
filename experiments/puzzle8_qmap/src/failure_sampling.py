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
        self.retention_restore = self.retention_middle = None
        self.retention_weights_restore = self.retention_weights_middle = None
        self.retention_draws = 0
        self._retention_encoded = {}

    def retained_states(self, model, features, goal, children, legal, truth):
        path = self.repair.get("retention_restore_checkpoint" if goal == 0 else "retention_checkpoint")
        if path is None:
            return None, None
        if path not in self._retention_encoded:
            from .checkpoint_audit import load_model
            reference = load_model(torch.load(path, map_location=features.device, weights_only=False), features.device)
            self._retention_encoded[path] = torch.cat([reference.encode(part) for part in features.split(8192)])
        encoded = self._retention_encoded[path]
        heights = model.distance(encoded, encoded[int(goal)]).cpu().numpy()
        pointer = self.env.next[np.arange(self.env.size), np.where(legal, heights[children], np.inf).argmin(-1)]
        pointer[goal] = goal
        jump = pointer.copy()
        length = np.ones(self.env.size, dtype=np.int64)
        length[goal] = 0
        for _ in range(self.env.size.bit_length()):
            length, jump = length + length[jump], jump[jump]
        within = (jump == goal) & (length <= 62)
        correct = self.env.distance(pointer, goal) == truth - 1
        return within, correct

    @torch.no_grad()
    def mine(self, model, features, state):
        encoded = torch.cat([model.encode(part) for part in features.split(8192)])
        known = np.unique(self.data["tasks_train"][:, 1])
        fresh = self.rng.choice(self.sources, self.repair["fresh_goals_per_refresh"], replace=False)
        goals = np.setdiff1d(np.unique(np.r_[known, fresh, 0]), self.protected)
        middle = []
        retention = {"restore": [], "middle": []}
        retained_weights = {"restore": [], "middle": []}
        legal = self.env.next >= 0
        children = np.maximum(self.env.next, 0)
        for goal in goals:
            height = model.distance(encoded, encoded[int(goal)]).cpu().numpy()
            chosen = self.env.next[np.arange(self.env.size),
                                   np.where(legal, height[children], np.inf).argmin(-1)]
            truth = self.env.distance(np.arange(self.env.size), int(goal))
            wrong = self.allowed & (truth > 0) & (self.env.distance(chosen, int(goal)) != truth - 1)
            within, correct = self.retained_states(model, features, int(goal), children, legal, truth)
            degree = legal.sum(-1)
            if within is not None:
                family = "restore" if int(goal) == 0 else "middle"
                for distance in range(1, self.env.diameter + 1):
                    for branches in np.unique(degree):
                        for lost in (False, True):
                            pool = np.flatnonzero(self.allowed & within & correct & (truth == distance) &
                                                  (degree == branches) & (wrong == lost))
                            if len(pool):
                                take = self.rng.choice(pool, min(len(pool), self.repair["errors_per_distance_degree"]),
                                                       replace=False)
                                retention[family].append(np.column_stack((take, np.full(len(take), goal))))
                                retained_weights[family].extend([4. if lost else 1.] * len(take))
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
            for distance in range(1, self.env.diameter + 1):
                for branches in np.unique(degree):
                    pool = np.flatnonzero(wrong & (truth == distance) & (degree == branches))
                    if len(pool):
                        take = self.rng.choice(pool, min(len(pool), self.repair["errors_per_distance_degree"]),
                                               replace=False)
                        middle.append(np.column_stack((take, np.full(len(take), goal))))
        for family in ("restore", "middle"):
            if retention[family]:
                setattr(self, f"retention_{family}", np.concatenate(retention[family]))
                weights = np.asarray(retained_weights[family])
                setattr(self, f"retention_weights_{family}", weights / weights.sum())
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
        retained = []
        counts = []
        for family, total in (("restore", restore), ("middle", count - restore)):
            pool = getattr(self, f"retention_{family}")
            keep = round(total * self.repair.get("retention_fraction", 0)) if pool is not None else 0
            if keep:
                weights = getattr(self, f"retention_weights_{family}")
                retained.append(pool[self.rng.choice(len(pool), keep, p=weights)])
            counts.append(total - keep)
            self.retention_draws += keep
        a = self.repair_restore[self.rng.choice(len(self.repair_restore), counts[0], p=self.restore_weights)]
        b = self.repair_middle[self.rng.integers(len(self.repair_middle), size=counts[1])]
        rows = np.concatenate((a, b, *retained)).copy()
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
        result["failure_repair"]["retention"] = {"restore": self.retention_restore,
            "middle": self.retention_middle, "weights_restore": self.retention_weights_restore,
            "weights_middle": self.retention_weights_middle, "draws": self.retention_draws}
        return result

    def load_state_dict(self, state):
        super().load_state_dict(state)
        if "failure_repair" in state:
            repair = state["failure_repair"]
            self.repair_restore, self.repair_middle = repair["restore"], repair["middle"]
            self.restore_weights, self.repair_seen = repair["weights"], repair["seen"]
            self.repair_draws, self.repair_distances = repair["draws"], repair["distances"]
            self.mine_round = repair["mine_round"]
            if "retention" in repair:
                kept = repair["retention"]
                self.retention_restore, self.retention_middle = kept["restore"], kept["middle"]
                self.retention_weights_restore, self.retention_weights_middle = kept["weights_restore"], kept["weights_middle"]
                self.retention_draws = kept["draws"]

    def report(self):
        result = super().report()
        result["failure_repair"] = {"draws": self.repair_draws,
            "distinct": {key: len(value) for key, value in self.repair_seen.items()},
            "distance_draws": {key: value.tolist() for key, value in self.repair_distances.items()},
            "mine_round": self.mine_round,
            "old_success_context_draws": self.retention_draws,
            "scope": "Global single sampler across replicas; exact distinct replay relations, overlapping original streams; candidate appearance is not nonzero score gradient"}
        return result
