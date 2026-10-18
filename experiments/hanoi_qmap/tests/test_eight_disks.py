import unittest
import numpy as np
from experiments.hanoi_qmap.src.environment import Hanoi
from experiments.task_text_baselines.src.hanoi import distances


class EightDiskTest(unittest.TestCase):
    def test_eight_disk_oracle_and_joint_peg_symmetry(self):
        env = Hanoi(8)
        self.assertEqual(env.size, 6561)
        self.assertEqual(env.distance(0, env.size - 1), 255)
        rng = np.random.default_rng(18)
        for goal in (0, 1234, env.size - 1):
            table = distances(tuple(map(int, env.positions[goal])))
            np.testing.assert_array_equal(env.distance(np.arange(env.size), goal),
                [table[tuple(map(int, state))] for state in env.positions])
        sources, goals = rng.integers(env.size, size=(2, 1000))
        for transform in range(6):
            np.testing.assert_array_equal(env.distance(sources, goals),
                env.distance(env.symmetry(sources, transform), env.symmetry(goals, transform)))
            for state in sources[:20]:
                for action, child in enumerate(env.next[state]):
                    expected = -1 if child < 0 else env.symmetry(child, transform)
                    self.assertEqual(env.next[env.symmetry(state, transform), env.symmetry_actions(action, transform)], expected)


if __name__ == "__main__":
    unittest.main()
