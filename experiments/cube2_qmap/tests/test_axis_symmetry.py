import unittest

import numpy as np

from experiments.cube2_qmap.src.environment import Cube2


class AxisSymmetryTests(unittest.TestCase):
    def test_all_six_maps_preserve_solved_and_relabel_every_action_transition(self):
        for metric in ("QTM_URF_6", "HTM_URF_9"):
            env = Cube2(metric=metric)
            rng = np.random.default_rng(30)
            states = rng.integers(env.size, size=3000)
            for transform in range(6):
                self.assertEqual(env.axis_symmetry(0, transform), 0)
                changed = env.axis_symmetry(states, transform)
                self.assertTrue(np.all((changed >= 0) & (changed < env.size)))
                expected = env.successors(changed)[:, env.axis_action_map[transform]]
                actual = env.axis_symmetry(env.successors(states), transform)
                np.testing.assert_array_equal(actual, expected)
                inverse = next(t for t in range(6) if np.array_equal(
                    env.axis_symmetry(changed[:20], t), states[:20]))
                np.testing.assert_array_equal(env.axis_symmetry(changed, inverse), states)

    def test_relative_relations_and_broadcasted_packs_transform_together(self):
        env = Cube2(metric="QTM_URF_6")
        rng = np.random.default_rng(31)
        source, goal = rng.integers(env.size, size=(2, 1000))
        changed = env.axis_symmetry(np.stack((source, goal), -1)[:, None, :],
                                    np.arange(6)[None, :, None])
        expected = env.axis_symmetry(env.relative(source, goal)[:, None], np.arange(6))
        actual = env.relative(changed[..., 0], changed[..., 1])
        np.testing.assert_array_equal(actual, expected)


if __name__ == "__main__":
    unittest.main()
