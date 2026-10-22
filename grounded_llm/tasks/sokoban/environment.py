"""Sokoban physical moves, compatible with Boxoban ASCII and push semantics."""
from itertools import combinations, permutations
import numpy as np
from grounded_llm.core.qmap.exact_graph import goal_distances


class Sokoban:
    actions = ("U", "D", "L", "R")
    deltas = ((-1, 0), (1, 0), (0, -1), (0, 1))

    def __init__(self, lines):
        self.height, self.width = len(lines), len(lines[0])
        if any(len(row) != self.width for row in lines):
            raise ValueError("Ragged board")
        self.floor, self.goals, boxes, players = set(), set(), [], []
        for r, row in enumerate(lines):
            for c, char in enumerate(row):
                if char not in '# .@$*+':
                    raise ValueError("Unknown board symbol")
                cell = r * self.width + c
                if char != '#': self.floor.add(cell)
                if char in '.+*': self.goals.add(cell)
                if char in '$*': boxes.append(cell)
                if char in '@+': players.append(cell)
        if len(players) != 1 or not boxes or len(boxes) != len(self.goals):
            raise ValueError("Require one player and equal nonzero boxes/targets")
        self.start = players[0], tuple(sorted(boxes))

    def adjacent(self, cell, delta):
        r, c = divmod(cell, self.width)
        r, c = r + delta[0], c + delta[1]
        return r * self.width + c if 0 <= r < self.height and 0 <= c < self.width else -1

    def transitions(self, state):
        player, boxes = state
        for action, delta in enumerate(self.deltas):
            dest = self.adjacent(player, delta)
            if dest not in self.floor: continue
            moved = set(boxes)
            if dest in moved:
                beyond = self.adjacent(dest, delta)
                if beyond not in self.floor or beyond in moved: continue
                moved.remove(dest)
                moved.add(beyond)
            yield action, (dest, tuple(sorted(moved))), 1

    def terminal(self, state):
        return set(state[1]) == self.goals

    def validate(self, state):
        player, boxes = state
        if (type(player) is not int or player not in self.floor or player in boxes
                or len(boxes) != len(self.goals) or len(set(boxes)) != len(boxes)
                or any(type(b) is not int or b not in self.floor for b in boxes)
                or tuple(boxes) != tuple(sorted(boxes))):
            raise ValueError('Expected one player and a sorted box set on distinct floor cells')

    def features(self, states):
        area = self.height * self.width
        out = np.zeros((len(states), 4, area), dtype=np.float32)
        walls = list(set(range(area)) - self.floor)
        out[:, 0, walls] = 1
        out[:, 1, list(self.goals)] = 1
        if len(states):
            players=np.fromiter((s[0] for s in states),dtype=np.int64,count=len(states))
            boxes=np.asarray([s[1] for s in states],dtype=np.int64)
            out[np.arange(len(states))[:,None],2,boxes]=1
            valid=players>=0
            out[np.flatnonzero(valid),3,players[valid]]=1
        return out.reshape(len(states), -1)

    def graph(self):
        states = [(p, boxes) for boxes in combinations(sorted(self.floor), len(self.goals))
                  for p in sorted(self.floor) if p not in boxes]
        index = {s: i for i, s in enumerate(states)}
        children = np.full((len(states), 4), -1, dtype=np.int32)
        costs = np.ones_like(children, dtype=np.float32)
        for i, s in enumerate(states):
            for a, t, _ in self.transitions(s): children[i, a] = index[t]
        terminal = np.array([self.terminal(s) for s in states])
        distances = goal_distances(children, costs, terminal)
        def baseline(state):
            return min(sum(abs(b // self.width - g // self.width) + abs(b % self.width - g % self.width)
                           for b, g in zip(state[1], targets)) for targets in permutations(self.goals))
        return dict(features=self.features(states), children=children, costs=costs,
                    baseline=np.asarray([baseline(s) for s in states], dtype=np.float32),
                    distances=distances, terminal=terminal,
                    goal_features=self.features([(-1, tuple(sorted(self.goals)))])[0],
                    starts=np.flatnonzero(np.isfinite(distances) & (distances > 0)),
                    metadata={"floor": sorted(self.floor), "goals": sorted(self.goals),
                              "shape": [self.height, self.width], "goal_is_predicate_anchor": True})

    def judge(self, actions, start=None):
        state = self.start if start is None else start
        for action in actions:
            choices = {self.actions[a]: t for a, t, _ in self.transitions(state)}
            if action not in choices:
                return {"success": False, "reason": "illegal_action"}
            state = choices[action]
        return {"success": self.terminal(state), "cost": len(actions)}
