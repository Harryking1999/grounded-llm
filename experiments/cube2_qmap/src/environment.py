"""Pocket cube in the DBL-fixed frame, HTM, Kociemba corner convention.

No edges or 3x3 corner/edge parity restriction. The valid states are all 7!
corner permutations times 3**6 twists. Whole-cube rotations are quotiented out
by requiring corner 6 (DBL) at position 6, orientation zero.
"""
from itertools import permutations
from math import factorial
from pathlib import Path

import numpy as np


SLOTS = np.array([0, 1, 2, 3, 4, 5, 7])
POWERS = 3 ** np.arange(5, -1, -1)
FACTORS = np.array([factorial(6 - i) for i in range(7)])
CORNER_NAMES = ("URF", "UFL", "ULB", "UBR", "DFR", "DLF", "DBL", "DRB")
# "is replaced by" arrays. These are move definitions, not solver code.
QUARTERS = (
    ([3, 0, 1, 2, 4, 5, 6, 7], [0] * 8),
    ([4, 1, 2, 0, 7, 5, 6, 3], [2, 0, 0, 1, 1, 0, 0, 2]),
    ([1, 5, 2, 3, 0, 4, 6, 7], [1, 2, 0, 0, 2, 1, 0, 0]),
)


def permutation_rank(values):
    values = np.asarray(values)
    rank = np.zeros(values.shape[:-1], dtype=np.int64)
    for i in range(6):
        rank += (values[..., i + 1:] < values[..., i, None]).sum(axis=-1) * FACTORS[i]
    return rank


class Cube2:
    name = "cube2"
    size = 5040 * 729
    diameter = 11
    feature_dim = 8 * 8 * 3
    actions = ("U", "U2", "U'", "R", "R2", "R'", "F", "F2", "F'")
    symmetry_kind = "common_left_multiplication"

    def __init__(self, oracle_path=None):
        self.permutations = np.tile(np.arange(8), (5040, 1))
        self.permutations[:, SLOTS] = np.array(list(permutations(SLOTS)))
        self.twists = np.zeros((729, 8), dtype=np.int64)
        self.twists[:, SLOTS[:6]] = np.arange(729)[:, None] // POWERS % 3
        self.twists[:, 7] = -self.twists.sum(axis=1) % 3
        self.cp_next = np.empty((5040, 9), dtype=np.int64)
        self.co_next = np.empty((729, 9), dtype=np.int64)
        for face, (cp, co) in enumerate(QUARTERS):
            p, o = self.permutations.copy(), self.twists.copy()
            for power in range(3):
                p, o = p[:, cp], (o[:, cp] + co) % 3
                self.cp_next[:, 3 * face + power] = permutation_rank(p[:, SLOTS])
                self.co_next[:, 3 * face + power] = o[:, SLOTS[:6]] @ POWERS
        self.depth = None
        if oracle_path is not None and Path(oracle_path).exists():
            self.depth = np.load(oracle_path, allow_pickle=False)
            if self.depth.shape != (self.size,) or self.depth.dtype != np.uint8 or self.depth.max() != 11:
                raise ValueError("Invalid complete HTM oracle table")

    def validate(self, state):
        if not isinstance(state, (int, np.integer)) or isinstance(state, (bool, np.bool_)) or not 0 <= state < self.size:
            raise ValueError(f"Cube state id must be an integer in [0, {self.size})")

    def state_id(self, permutation, orientation):
        p, o = np.asarray(permutation), np.asarray(orientation)
        if p.shape != (8,) or p.dtype.kind not in "iu" or sorted(p.tolist()) != list(range(8)):
            raise ValueError("Corners must be a permutation of 0..7")
        if o.shape != (8,) or o.dtype.kind not in "iu" or np.any((o < 0) | (o > 2)) or o.sum() % 3:
            raise ValueError("Eight twists in 0..2 must sum to zero modulo 3")
        if p[6] != 6 or o[6] != 0:
            raise ValueError("Normalize the cube frame first: DBL corner must be fixed and untwisted")
        return int(permutation_rank(p[SLOTS]) * 729 + o[SLOTS[:6]] @ POWERS)

    def describe(self, state):
        self.validate(state)
        return {"slots": list(CORNER_NAMES), "permutation": self.permutations[state // 729].tolist(),
                "orientation": self.twists[state % 729].tolist()}

    def successors(self, ids):
        ids = np.asarray(ids, dtype=np.int64)
        return self.cp_next[ids // 729] * 729 + self.co_next[ids % 729]

    def step(self, state, action):
        self.validate(state)
        if action not in self.actions:
            raise ValueError(f"Unknown or out-of-frame action: {action}")
        return int(self.successors(state)[self.actions.index(action)])

    def features(self, ids):
        ids = np.asarray(ids, dtype=np.int64).reshape(-1)
        p, o = self.permutations[ids // 729], self.twists[ids % 729]
        return np.eye(24, dtype=np.float32)[p * 3 + o].reshape(-1, self.feature_dim)

    def baseline(self, source, goal):
        source, goal = np.asarray(source), np.asarray(goal)
        return ((self.permutations[source // 729] != self.permutations[goal // 729]) |
                (self.twists[source % 729] != self.twists[goal % 729])).sum(axis=-1)

    def compose(self, left, right):
        """Apply right after left; inputs broadcast. Useful for exact pair sampling."""
        left, right = np.broadcast_arrays(left, right)
        p, q = self.permutations[left // 729], self.permutations[right // 729]
        o, r = self.twists[left % 729], self.twists[right % 729]
        cp = np.take_along_axis(p, q, axis=-1)
        co = (np.take_along_axis(o, q, axis=-1) + r) % 3
        return permutation_rank(cp[..., SLOTS]) * 729 + co[..., SLOTS[:6]] @ POWERS

    def sample_symmetry(self, rng, count):
        return rng.integers(self.size, size=count)

    def symmetry(self, ids, transform):
        """Common left multiplication preserves right-action edges and distance.

        This changes corner identities/reference twists in the DBL-fixed group;
        it is not the operation of applying the same scramble to both inputs.
        """
        return self.compose(np.asarray(transform), np.asarray(ids))

    def symmetry_actions(self, actions, transform):
        return np.broadcast_arrays(actions, transform)[0]

    def relative(self, source, goal):
        """source^-1 * goal: its solved distance is d(source, goal)."""
        source, goal = np.broadcast_arrays(source, goal)
        p, q = self.permutations[source // 729], self.permutations[goal // 729]
        o, r = self.twists[source % 729], self.twists[goal % 729]
        inv = np.argsort(p, axis=-1)
        cp = np.take_along_axis(inv, q, axis=-1)
        co = (r - np.take_along_axis(o, cp, axis=-1)) % 3
        return permutation_rank(cp[..., SLOTS]) * 729 + co[..., SLOTS[:6]] @ POWERS

    def build_oracle(self, path):
        """Complete BFS using factored transitions; ~4 MB final distance table."""
        path = Path(path)
        if self.depth is not None:
            return
        depth = np.full(self.size, 255, dtype=np.uint8)
        depth[0] = 0
        frontier, level = np.array([0], dtype=np.int64), 0
        while len(frontier):
            chunks = []
            for offset in range(0, len(frontier), 50000):
                children = self.successors(frontier[offset:offset + 50000]).ravel()
                unseen = np.unique(children[depth[children] == 255])
                depth[unseen] = level + 1
                chunks.append(unseen)
            print(f"oracle depth={level} states={len(frontier)}", flush=True)
            frontier = np.concatenate(chunks)
            level += 1
        if np.any(depth == 255) or int(depth.max()) != self.diameter:
            raise RuntimeError("BFS failed to cover the defined pocket-cube group")
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, depth)
        self.depth = depth

    def distance(self, source, goal):
        if self.depth is None:
            raise RuntimeError("Build/load the exact oracle before requesting labels")
        return self.depth[self.relative(source, goal)].astype(np.int64)

    def solve(self, source, goal=0):
        self.validate(source)
        self.validate(goal)
        result = []
        while source != goal:
            children = self.successors(source)
            distances = self.distance(children, goal)
            action = int(np.flatnonzero(distances == self.distance(source, goal) - 1)[0])
            result.append(self.actions[action])
            source = int(children[action])
        return result
