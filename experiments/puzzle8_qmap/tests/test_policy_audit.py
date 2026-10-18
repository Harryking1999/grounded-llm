import unittest
import numpy as np
from experiments.puzzle8_qmap.src.policy_audit import resolve, cycle_nodes


class PolicyAuditTest(unittest.TestCase):
    def test_long_paths_cycles_and_budget(self):
        pointer = np.array([0, 0, 1, 2, 5, 4, 5, 8, 9, 7])
        solved, within, length = resolve(pointer, 2)
        np.testing.assert_array_equal(solved, [1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
        np.testing.assert_array_equal(within, [1, 1, 1, 0, 0, 0, 0, 0, 0, 0])
        np.testing.assert_array_equal(length, [0, 1, 2, 3, -1, -1, -1, -1, -1, -1])
        np.testing.assert_array_equal(cycle_nodes(pointer), [0, 0, 0, 0, 1, 1, 0, 1, 1, 1])
        with self.assertRaises(ValueError):
            resolve([1, 0], 2)


if __name__ == "__main__":
    unittest.main()
