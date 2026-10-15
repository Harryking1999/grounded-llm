from collections import deque
import tempfile
from pathlib import Path
import unittest

import numpy as np

from experiments.cube2_qmap.src.environment import Cube2


# Independent sticker geometry: ordered normals of each corner's three stickers.
NORMALS = np.array([
    [(0, 1, 0), (1, 0, 0), (0, 0, 1)],
    [(0, 1, 0), (0, 0, 1), (-1, 0, 0)],
    [(0, 1, 0), (-1, 0, 0), (0, 0, -1)],
    [(0, 1, 0), (0, 0, -1), (1, 0, 0)],
    [(0, -1, 0), (0, 0, 1), (1, 0, 0)],
    [(0, -1, 0), (-1, 0, 0), (0, 0, 1)],
    [(0, -1, 0), (0, 0, -1), (-1, 0, 0)],
    [(0, -1, 0), (1, 0, 0), (0, 0, -1)],
])


class CubeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = Cube2()

    def test_moves_against_independent_spatial_rotation(self):
        env = self.env
        coordinates = NORMALS.sum(axis=1)
        positions = coordinates.copy()
        reference_normals = NORMALS[:, 0].copy()
        # Minus 90 degrees about the outward normal: clockwise seen from outside.
        matrices = {"R": np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]]),
                    "U": np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]]),
                    "F": np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])}
        axes = {"R": 0, "U": 1, "F": 2}
        rng, state = np.random.default_rng(42), 0
        for _ in range(100):
            action = rng.choice(env.actions)
            face = action[0]
            for _ in range(2 if action.endswith("2") else 3 if action.endswith("'") else 1):
                mask = positions[:, axes[face]] == 1
                positions[mask] = positions[mask] @ matrices[face].T
                reference_normals[mask] = reference_normals[mask] @ matrices[face].T
            state = env.step(state, action)
            cp, co = [], []
            for slot, coordinate in enumerate(coordinates):
                cubie = int(np.flatnonzero((positions == coordinate).all(axis=1))[0])
                cp.append(cubie)
                co.append(int(np.flatnonzero((NORMALS[slot] == reference_normals[cubie]).all(axis=1))[0]))
            self.assertEqual(env.state_id(cp, co), state)

    def test_encoding_inverse_and_relative_group(self):
        env = self.env
        rng = np.random.default_rng(123)
        ids = rng.integers(env.size, size=100)
        for state in ids:
            self.assertEqual(env.state_id(env.permutations[state // 729], env.twists[state % 729]), state)
        left, right = ids[:50], ids[50:]
        np.testing.assert_array_equal(env.compose(left, env.relative(left, right)), right)
        np.testing.assert_array_equal(env.relative(left, left), np.zeros(50))
        for face in ("U", "R", "F"):
            state = 0
            for _ in range(4):
                state = env.step(state, face)
            self.assertEqual(state, 0)
            self.assertEqual(env.step(env.step(12345, face), face + "'"), 12345)

    def test_invalid_and_odd_permutation(self):
        env = self.env
        # A 2x2 has no edge parity constraint: a two-corner swap is legal.
        self.assertGreater(env.state_id([1, 0, 2, 3, 4, 5, 6, 7], [0] * 8), 0)
        for p, o in (([0] * 8, [0] * 8), (list(range(8)), [1] + [0] * 7),
                     ([0, 1, 2, 3, 4, 6, 5, 7], [0] * 8)):
            with self.assertRaises(ValueError):
                env.state_id(p, o)
        with self.assertRaises(ValueError):
            env.step(-1, "U")
        with self.assertRaises(ValueError):
            env.step(0, "D")

    def test_exact_oracle_full_coverage_and_arbitrary_goals(self):
        env = Cube2()
        with tempfile.TemporaryDirectory() as tmp:
            env.build_oracle(Path(tmp) / "oracle.npy")
        self.assertEqual(np.bincount(env.depth).tolist(),
                         [1, 9, 54, 321, 1847, 9992, 50136, 227536, 870072, 1887748, 623800, 2644])
        rng = np.random.default_rng(321)
        for _ in range(30):
            source, goal = rng.integers(env.size, size=2)
            self.assertEqual(env.distance(source, goal), env.distance(goal, source))
            state = source
            actions = env.solve(source, goal)
            self.assertEqual(len(actions), env.distance(source, goal))
            for action in actions:
                state = env.step(state, action)
            self.assertEqual(state, goal)
        # Independent short BFS from an arbitrary goal tests the relative formula.
        goal, queue, seen = 123456, deque([(123456, 0)]), {123456}
        while queue:
            state, d = queue.popleft()
            self.assertEqual(env.distance(state, goal), d)
            if d < 3:
                for child in env.successors(state):
                    if int(child) not in seen:
                        seen.add(int(child))
                        queue.append((int(child), d + 1))


if __name__ == "__main__":
    unittest.main()
