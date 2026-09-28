"""Checks for goal-dependent labels and grouped supervision."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from experiments.blocks_distance_map.src.multiboard_data import (
    candidate_pairs, choose_landmarks, contrast_cases, fresh_official_boards)
from experiments.blocks_distance_map.src.multiboard_eval import listwise_groups
from experiments.blocks_distance_map.src.oracle import DistanceOracle


class MultiboardTests(unittest.TestCase):
    def test_fresh_official_boards_uses_frozen_rows_after_historical_split(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "base.npz"
            config = root / "config.json"
            comparison = root / "comparison.json"
            np.savez_compressed(data, train_board_rows=[1, 2], ood_board_rows=[10, 11])
            config.write_text(json.dumps({"board_seed": 7}), encoding="utf-8")
            comparison.write_text(json.dumps({"skip_fresh_boards": 1,
                                              "board_rows": [20, 21]}), encoding="utf-8")
            ood = [(row, 0, []) for row in (10, 11, 99, 20, 21)]
            with patch("experiments.blocks_distance_map.src.multiboard_data.official_boards",
                       return_value=([], ood)) as source:
                fresh, frozen = fresh_official_boards(data, config, "official.h5", comparison)
            source.assert_called_once_with("official.h5", 7, 2, 5)
            self.assertEqual([row for row, _, _ in fresh], [20, 21])
            self.assertEqual(frozen["board_rows"], [20, 21])
            comparison.write_text(json.dumps({"skip_fresh_boards": 1,
                                              "board_rows": [20, 22]}), encoding="utf-8")
            with patch("experiments.blocks_distance_map.src.multiboard_data.official_boards",
                       return_value=([], ood)):
                with self.assertRaises(ValueError):
                    fresh_official_boards(data, config, "official.h5", comparison)

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

    def test_heldout_landmarks_include_isolated_and_supported(self):
        isolated, supported = 0b100000, 0b111000
        landmarks = choose_landmarks([[0b111111, supported, 0], [0b100011, isolated]],
                                     2, np.random.default_rng(1), heldout_count=2)
        self.assertEqual(set(landmarks), {isolated, supported})

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
