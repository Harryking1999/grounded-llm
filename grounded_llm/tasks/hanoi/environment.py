"""Hanoi with 1--8 disks: shared rules, state interface and exact BFS oracle."""
from collections import deque
from itertools import permutations

import numpy as np


class Hanoi:
    family = "hanoi"
    actions = ("A>B", "A>C", "B>A", "B>C", "C>A", "C>B")
    symmetry_kind = "simultaneous_peg_permutation"
    symmetry_count = 6

    def __init__(self, disks=3):
        if not isinstance(disks, int) or isinstance(disks, bool) or not 1 <= disks <= 8:
            raise ValueError("This study supports 1--8 disks")
        self.disks, self.name = disks, f"hanoi{disks}"
        self.size, self.diameter, self.feature_dim = 3**disks, 2**disks - 1, 3 * disks
        self.powers = 3 ** np.arange(disks)
        self.positions = np.array([[s // 3**d % 3 for d in range(disks)]
                                   for s in range(self.size)], dtype=np.int64)
        self.peg_permutations = np.array(list(permutations(range(3))))
        self.renamed = (self.peg_permutations[:, self.positions] @ self.powers).T
        moves = [(i, j) for i in range(3) for j in range(3) if i != j]
        self.renamed_actions = np.array([[moves.index((int(p[i]), int(p[j]))) for i, j in moves]
                                        for p in self.peg_permutations]).T
        self.next = np.full((self.size, len(self.actions)), -1, dtype=np.int64)
        for state, positions in enumerate(self.positions):
            top = [next((d for d in range(disks) if positions[d] == peg), disks)
                   for peg in range(3)]
            for action, (src, dst) in enumerate((i, j) for i in range(3)
                                                for j in range(3) if i != j):
                if top[src] < top[dst]:
                    self.next[state, action] = state + (dst - src) * 3**top[src]
        if disks >= 7:
            # The same unweighted BFS oracle, compiled for the larger graph.
            from scipy.sparse import csr_matrix
            from scipy.sparse.csgraph import shortest_path
            rows, actions = np.where(self.next >= 0)
            graph = csr_matrix((np.ones(len(rows)), (rows, self.next[rows, actions])),
                               shape=(self.size, self.size))
            self.distances = shortest_path(graph, directed=False, unweighted=True).astype(np.int16)
            return
        self.distances = np.full((self.size, self.size), -1, dtype=np.int16)
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
            raise ValueError(f"Hanoi state id must be an integer in [0, {self.size})")

    def state_id(self, positions):
        values = np.asarray(positions)
        if values.shape != (self.disks,) or values.dtype.kind not in "iu" or np.any((values < 0) | (values > 2)):
            raise ValueError("Specify a peg 0, 1, 2 for each disk, smallest first")
        return int(values @ self.powers)

    def describe(self, state):
        self.validate(state)
        return {chr(65 + peg): [d + 1 for d in reversed(range(self.disks))
                                if self.positions[state, d] == peg] for peg in range(3)}

    def successors(self, ids):
        return self.next[np.asarray(ids)]

    def sample_symmetry(self, rng, count):
        return rng.integers(self.symmetry_count, size=count)

    def symmetry(self, ids, transform):
        """Rename all pegs together; caller broadcasts one transform per group."""
        return self.renamed[np.asarray(ids), np.asarray(transform)]

    def symmetry_actions(self, actions, transform):
        return self.renamed_actions[np.asarray(actions), np.asarray(transform)]

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

    def solve(self, source, goal=None):
        goal = self.size - 1 if goal is None else goal
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
