"""Three-disk Hanoi: rules, vectorized state interface, and exact BFS oracle."""
from collections import deque

import numpy as np


class Hanoi:
    name = "hanoi3"
    size = 27
    diameter = 7
    feature_dim = 9
    actions = ("A>B", "A>C", "B>A", "B>C", "C>A", "C>B")

    def __init__(self):
        self.positions = np.array([[s // 3**d % 3 for d in range(3)]
                                   for s in range(self.size)], dtype=np.int64)
        self.next = np.full((self.size, len(self.actions)), -1, dtype=np.int64)
        for state, positions in enumerate(self.positions):
            top = [next((d for d in range(3) if positions[d] == peg), 3)
                   for peg in range(3)]
            for action, (src, dst) in enumerate((i, j) for i in range(3)
                                                for j in range(3) if i != j):
                if top[src] < top[dst]:
                    self.next[state, action] = state + (dst - src) * 3**top[src]
        self.distances = np.full((self.size, self.size), -1, dtype=np.int8)
        for goal in range(self.size):
            self.distances[goal, goal] = 0
            queue = deque([goal])
            while queue:
                state = queue.popleft()
                for child in self.next[state]:
                    if child >= 0 and self.distances[child, goal] < 0:
                        self.distances[child, goal] = self.distances[state, goal] + 1
                        queue.append(int(child))

    def validate(self, state):
        if not isinstance(state, (int, np.integer)) or isinstance(state, (bool, np.bool_)) or not 0 <= state < self.size:
            raise ValueError("Hanoi state id must be an integer in [0, 27)")

    def state_id(self, positions):
        values = np.asarray(positions)
        if values.shape != (3,) or values.dtype.kind not in "iu" or np.any((values < 0) | (values > 2)):
            raise ValueError("Specify a peg 0, 1, 2 for each disk, smallest first")
        return int(values @ np.array([1, 3, 9]))

    def describe(self, state):
        self.validate(state)
        return {chr(65 + peg): [d + 1 for d in reversed(range(3))
                                if self.positions[state, d] == peg] for peg in range(3)}

    def successors(self, ids):
        return self.next[np.asarray(ids)]

    def step(self, state, action):
        self.validate(state)
        if action not in self.actions:
            raise ValueError(f"Unknown action: {action}")
        child = int(self.next[state, self.actions.index(action)])
        if child < 0:
            raise ValueError("Empty source or a larger disk on a smaller disk")
        return child

    def features(self, ids):
        return np.eye(3, dtype=np.float32)[self.positions[np.asarray(ids)]].reshape(-1, self.feature_dim)

    def distance(self, source, goal):
        return self.distances[np.asarray(source), np.asarray(goal)].astype(np.int64)

    def baseline(self, source, goal):
        return (self.positions[np.asarray(source)] != self.positions[np.asarray(goal)]).sum(axis=-1)

    def solve(self, source, goal=26):
        self.validate(source)
        self.validate(goal)
        result = []
        while source != goal:
            child_dist = self.distance(np.maximum(self.next[source], 0), goal)
            action = next(i for i, child in enumerate(self.next[source])
                          if child >= 0 and child_dist[i] == self.distances[source, goal] - 1)
            result.append(self.actions[action])
            source = int(self.next[source, action])
        return result
