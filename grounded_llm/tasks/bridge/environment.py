"""Capacity-two bridge and torch; cost is the slower traveller's time."""
from itertools import combinations
from functools import cached_property
import numpy as np
from grounded_llm.core.qmap.exact_graph import goal_distances


class Bridge:
    def __init__(self, times):
        if not times or any(t <= 0 for t in times):
            raise ValueError("Crossing times must be positive")
        self.times = tuple(times)
        self.n = len(times)
        self.goal = ((1 << self.n) - 1) * 2 + 1

    @cached_property
    def actions(self):
        return tuple(combinations(range(self.n), 1)) + tuple(combinations(range(self.n), 2))

    def apply(self, state, people):
        """Replay a supplied crossing without enumerating all O(n^2) candidates."""
        if (not 1 <= len(people) <= 2 or len(set(people)) != len(people)
                or any(type(p) is not int or not 0 <= p < self.n for p in people)):
            raise ValueError('Expected one or two distinct valid person IDs')
        mask, side = divmod(state, 2)
        if not all(((mask >> p) & 1) == side for p in people):
            raise ValueError('Travellers must be on the torch bank')
        for p in people:
            mask ^= 1 << p
        return mask * 2 + 1 - side, max(self.times[p] for p in people)

    def transitions(self, state):
        mask, side = divmod(state, 2)
        for a, people in enumerate(self.actions):
            if all(((mask >> p) & 1) == side for p in people):
                target, cost = self.apply(state, people)
                yield a, target, cost

    def features(self, states):
        return np.asarray([[*self.times, *[(s // 2 >> p) & 1 for p in range(self.n)], s % 2]
                           for s in states], dtype=np.float32)

    def graph(self):
        states = list(range(2 ** (self.n + 1)))
        children = np.full((len(states), len(self.actions)), -1, dtype=np.int32)
        costs = np.zeros_like(children, dtype=np.float32)
        for s in states:
            for a, t, c in self.transitions(s):
                children[s, a], costs[s, a] = t, c
        terminal = np.asarray(states) == self.goal
        return dict(features=self.features(states), children=children, costs=costs,
                    baseline=np.asarray([max([self.times[p] for p in range(self.n) if not ((s // 2 >> p) & 1)] or [0]) for s in states], dtype=np.float32),
                    distances=goal_distances(children, costs, terminal), terminal=terminal,
                    goal_features=self.features([self.goal])[0], starts=np.array([0]),
                    metadata={"times": self.times, "actions": self.actions})

    def judge(self, actions, start=0):
        state, total = start, 0
        for action in actions:
            choices = {self.actions[a]: (t, c) for a, t, c in self.transitions(state)}
            key = tuple(sorted(action))
            if key not in choices:
                return {"success": False, "reason": "illegal_action", "cost": total}
            state, cost = choices[key]
            total += cost
        optimum = float(self.graph()["distances"][start])
        return {"success": state == self.goal, "cost": total,
                "optimal": state == self.goal and total == optimum}
