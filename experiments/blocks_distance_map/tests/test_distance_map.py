"""Correctness checks for the current shared-board Q pipeline."""
from collections import deque
import time
import unittest

import numpy as np
import torch

from experiments.gcml_counterexamples.src import blocks
from experiments.blocks_distance_map.src.model import BoardEncoder, board_bits, distance
from experiments.blocks_distance_map.src.oracle import (
    DistanceOracle, OracleBudgetExceeded, successors,
)
from experiments.blocks_distance_map.src.rollout_eval import summarize, trace


class DistanceMapTests(unittest.TestCase):
    def test_exact_distance_matches_directed_bfs_on_small_states(self):
        oracle = DistanceOracle()
        for source in range(64):
            distances, queue = {source: 0}, deque([source])
            while queue:
                state = queue.popleft()
                for _, following in successors(state):
                    if following not in distances:
                        distances[following] = distances[state] + 1
                        queue.append(following)
            for target in range(64):
                self.assertEqual(oracle.distance(source, target),
                                 distances.get(target, -1), (source, target))

    def test_legal_successor_can_be_unreachable_for_a_specific_goal(self):
        oracle = DistanceOracle()
        good, bad = 0b111000, 0b110001
        self.assertEqual(oracle.distance(0b111111, good), 1)
        self.assertEqual(oracle.distance(0b111111, bad), 1)
        self.assertEqual(oracle.distance(good, 0), 1)
        self.assertEqual(oracle.distance(bad, 0), -1)
        retained = 0b100011
        self.assertEqual(oracle.distance(0b111111, retained), 1)
        self.assertEqual(oracle.distance(retained, 0), -1)
        self.assertEqual(oracle.distance(retained, 0b100000), 1)

    def test_oracle_budget_does_not_create_false_negative(self):
        oracle = DistanceOracle()
        oracle.deadline = time.monotonic() - 1
        with self.assertRaises(OracleBudgetExceeded):
            oracle.distance(0b111111, 0)
        self.assertNotIn(0b111111, oracle.cache)

    def test_board_encoder_handles_unseen_states_and_directed_scores(self):
        bits = board_bits([0, 1, 1 << 10, (1 << 99) | 1])
        self.assertEqual(bits.shape, (4, 100))
        self.assertEqual(bits.sum(dim=1).tolist(), [0, 1, 1, 2])
        encoder = BoardEncoder(8, 16, 5, "directed_sum")
        score = encoder(bits, torch.tensor([[3, 0], [0, 3]]))
        self.assertEqual(score.shape, (2,))
        self.assertTrue(torch.isfinite(score).all())
        points = torch.tensor([[0., 2., 1.], [3., 0., 1.]])
        self.assertNotEqual(distance(points[:1], points[1:], "directed_sum").item(),
                            distance(points[1:], points[:1], "directed_sum").item())

    def test_rollout_reports_a_goal_specific_first_error(self):
        class PreferBad:
            def scores(self, masks, goal):
                return np.asarray([0 if mask == 0b110001 else 1 for mask in masks])

        oracle = DistanceOracle()
        source, goal = 0b111111, 0
        result = trace(PreferBad(), source, goal, oracle.distance(source, goal), oracle)
        self.assertEqual(result["step"], 1)
        self.assertEqual(result["outcome"], "local_coverage_failure")
        self.assertGreater(result["reachable_count"], 0)
        summary = summarize([result])
        self.assertEqual(summary["outcomes"]["local_coverage_failure"], 1)

    def test_official_actions_are_only_two_or_three_cells(self):
        self.assertEqual(len(blocks.PLACEMENTS), 664)
        self.assertEqual({mask.bit_count() for mask, _ in blocks.PLACEMENTS}, {2, 3})


if __name__ == "__main__":
    unittest.main()
