"""Checks for goal-dependent labels and grouped supervision."""
import unittest

import numpy as np

from experiments.blocks_distance_map.src.multiboard_data import candidate_pairs, choose_landmarks, contrast_cases
from experiments.blocks_distance_map.src.multiboard_eval import listwise_groups
from experiments.blocks_distance_map.src.oracle import DistanceOracle


class MultiboardTests(unittest.TestCase):
    def test_dead_state_can_reach_a_nonempty_dead_goal(self):
        oracle = DistanceOracle()
        source, goal = 0b100011, 0b100000
        self.assertEqual(oracle.distance(source, 0), -1)
        self.assertEqual(oracle.distance(goal, 0), -1)
        self.assertEqual(oracle.distance(source, goal), 1)
        pairs = candidate_pairs([[source, goal]], [goal], 2, np.random.default_rng(1))
        self.assertIn((source, goal), pairs)

    def test_isolated_landmark_is_selected_explicitly(self):
        isolated = 0b100000
        self.assertEqual(choose_landmarks([[0b100011, isolated]], 1,
                                          np.random.default_rng(1)), [isolated])

    def test_contrast_reverses_with_goal(self):
        oracle = DistanceOracle()
        cases = contrast_cases([[0b11111111, 0b111111, 0]], oracle, 1, np.random.default_rng(2))
        self.assertTrue(cases)
        _, a, b, ga, gb = cases[0]
        self.assertEqual(oracle.distance(a, ga), 1)
        self.assertEqual(oracle.distance(b, gb), 1)
        self.assertEqual(oracle.distance(b, ga), -1)
        self.assertEqual(oracle.distance(a, gb), -1)

    def test_listwise_groups_share_an_anchor_and_different_labels(self):
        pairs = np.asarray([[1, 3, 1, 0], [2, 3, 2, 0], [4, 3, -1, 2],
                            [1, 5, 1, 0], [1, 6, -1, 1]], dtype=np.int32)
        groups = listwise_groups(pairs, 3, per_anchor=2)
        self.assertGreater(len(groups), 0)
        for group in groups:
            selected = pairs[group]
            self.assertTrue(len(set(selected[:, 0])) == 1 or len(set(selected[:, 1])) == 1)
            self.assertGreater(len(set(selected[:, 2])), 1)


if __name__ == "__main__":
    unittest.main()
