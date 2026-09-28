"""The hard-branch stratum defeats target retention, local support and area."""
import unittest

import numpy as np

from experiments.blocks_distance_map.src.hardbranch_data import find_cases
from experiments.blocks_distance_map.src.oracle import DistanceOracle, successors
from experiments.gcml_counterexamples.src import blocks


class HardBranchTests(unittest.TestCase):
    def test_equal_area_locally_plausible_dead_branch(self):
        parent = 31912711224343
        goal = 8209
        oracle = DistanceOracle()
        parent_distance = oracle.distance(parent, goal)
        self.assertGreaterEqual(parent_distance, 2)
        data = {"states": np.asarray([str(parent), str(goal)]),
                "train": np.asarray([[0, 1, parent_distance, 0]], dtype=np.int32)}
        cases, counts = find_cases(data, "train", 1, 1, 1, oracle)
        self.assertEqual(counts["cases"], 1)
        _, good, bad, target, good_distance = cases[0]
        children = {mask for _, mask in successors(parent)}
        self.assertIn(good, children)
        self.assertIn(bad, children)
        self.assertEqual(target, goal)
        self.assertEqual(good.bit_count(), bad.bit_count())
        self.assertTrue(blocks.locally_supported(good ^ goal))
        self.assertTrue(blocks.locally_supported(bad ^ goal))
        self.assertEqual(good & goal, goal)
        self.assertEqual(bad & goal, goal)
        self.assertEqual(oracle.distance(good, goal), good_distance)
        self.assertEqual(oracle.distance(bad, goal), -1)


if __name__ == "__main__":
    unittest.main()
