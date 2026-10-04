from collections import deque
import json
from pathlib import Path
import tempfile
import time
import unittest

import numpy as np
import torch

from experiments.gcml_counterexamples.src import blocks
from experiments.blocks_distance_map.src.data import comparisons, pair_splits, prepare
from experiments.blocks_distance_map.src.evaluate import decisions_metrics
from experiments.blocks_distance_map.src.model import BoardEncoder, board_bits, distance
from experiments.blocks_distance_map.src.oracle import DistanceOracle, OracleBudgetExceeded, successors


class DistanceMapTests(unittest.TestCase):
    def test_official_rules(self):
        self.assertEqual(len(blocks.PLACEMENTS), 664)
        self.assertEqual({t.bit_count() for t, _ in blocks.PLACEMENTS}, {2, 3})

    def test_difference_cover_matches_exhaustive_directed_bfs(self):
        oracle = DistanceOracle()
        # All states on a six-cell strip, including states unreachable from
        # the full strip. Compare every ordered pair, not just positive paths.
        for source in range(64):
            distances, queue = {source: 0}, deque([source])
            while queue:
                state = queue.popleft()
                for _, following in successors(state):
                    if following not in distances:
                        distances[following] = distances[state] + 1
                        queue.append(following)
            for target in range(64):
                self.assertEqual(oracle.distance(source, target), distances.get(target, -1),
                                 (source, target))

    def test_bad_branch_still_positive_and_deep_dead_is_certified(self):
        oracle = DistanceOracle()
        good, bad = 0b111000, 0b110001
        self.assertEqual(oracle.distance(0b111111, good), 1)
        self.assertEqual(oracle.distance(0b111111, bad), 1)
        self.assertEqual(oracle.distance(good, 0), 1)
        self.assertEqual(oracle.distance(bad, 0), -1)
        self.assertTrue(successors(bad))  # A domino is removable, but the singleton remains.
        self.assertEqual(good.bit_count(), bad.bit_count())
        self.assertEqual(oracle.distance(good, 0b111111), -1)

    def test_budget_does_not_create_false_negative(self):
        oracle = DistanceOracle()
        oracle.deadline = time.monotonic() - 1
        with self.assertRaises(OracleBudgetExceeded):
            oracle.distance(0b111111, 0)
        self.assertNotIn(0b111111, oracle.cache)

    def test_directed_landmark_embedding_recovers_capped_distance(self):
        oracle, cap = DistanceOracle(), 4
        matrix = torch.tensor([[cap if (d := oracle.distance(s, t)) < 0 else d
                                for t in range(64)] for s in range(64)], dtype=torch.float32)
        reconstructed = distance(matrix[:, None, :], matrix[None, :, :], "directed_max")
        torch.testing.assert_close(reconstructed, matrix)

    def test_directed_sum_is_asymmetric_and_obeys_triangle(self):
        points = torch.tensor([[0., 2., 1.], [2., 0., 1.], [1., 2., 2.]])
        d = distance(points[:, None, :], points[None, :, :], "directed_sum")
        self.assertNotEqual(d[0, 2].item(), d[2, 0].item())
        for i in range(len(points)):
            for j in range(len(points)):
                for k in range(len(points)):
                    self.assertLessEqual(d[i, k].item(), d[i, j].item() + d[j, k].item() + 1e-6)

    def test_board_encoder_scores_unseen_masks_without_state_ids(self):
        bits = board_bits([0, 1, 1 << 10, (1 << 99) | 1])
        self.assertEqual(bits.shape, (4, 100))
        self.assertEqual(bits[1, 0].item(), 1)
        self.assertEqual(bits[2, 10].item(), 1)
        self.assertEqual(bits[3, 99].item(), 1)
        self.assertEqual(bits.sum(dim=1).tolist(), [0, 1, 1, 2])
        encoder = BoardEncoder(8, 16, 5, "directed_sum")
        score = encoder(bits, torch.tensor([[3, 0], [0, 3]]))
        self.assertEqual(score.shape, (2,))
        self.assertTrue(torch.isfinite(score).all())
        self.assertTrue(((encoder.encode(bits) >= 0) & (encoder.encode(bits) <= 5)).all())

    def test_split_groups_reverse_and_reserves_goal_labels(self):
        pairs = np.array([[1, 0, 1, 0], [0, 1, -1, 1], [1, 2, 1, 0],
                          [2, 1, -1, 1], [2, 0, 2, 0], [0, 2, -1, 1]])
        split = pair_splits(pairs, {(0, 1)}, [.8, .1, .1], 1, {(1, 2)})
        self.assertEqual(split[0], 2)
        self.assertEqual(split[1], 2)
        self.assertEqual(split[2], split[3])
        self.assertEqual(split[2], 0)
        with self.assertRaises(ValueError):
            pair_splits(pairs, {(0, 1)}, [.8, .1, .1], 1, {(0, 1)})
        for sid in range(3):
            ids = np.flatnonzero(split == sid)
            for rows in comparisons(pairs, ids, 3, np.random.default_rng(1)):
                for near, far in rows:
                    self.assertIn(near, ids)
                    self.assertIn(far, ids)
                    d1 = pairs[near, 2] if pairs[near, 2] >= 0 else 100
                    d2 = pairs[far, 2] if pairs[far, 2] >= 0 else 100
                    self.assertLess(d1, d2)

    def test_action_ties_not_resolved_in_favor_of_oracle(self):
        data = {"states": np.array(["0", "7", "11", "15"]),
                "decisions": np.array([[3, 0, 1, 1], [3, 1, 2, -1]]),
                "immediate_dead": np.array([False, False, False, False])}
        result = decisions_metrics(data, np.array([0., 3., 3., 4.]))
        self.assertEqual(result["solvable_action_rate"], .5)
        self.assertEqual(result["optimal_action_rate"], .5)
        self.assertEqual(result["deep_dead_pairs"], 1)

    def test_landmark_supervision_adds_non_goal_hard_relations_without_goal_leak(self):
        config = json.loads((Path(__file__).parents[1] / "configs/landmark_probe.json").read_text())
        config["data"].update(episodes=30, target_pairs=3000, random_pair_attempts=3000,
                              subset_pair_attempts=3000, decision_parents=8)
        with tempfile.TemporaryDirectory() as work:
            summary = prepare(config, Path(work) / "data")
            data = np.load(Path(work) / "data/data.npz")
            pairs, split = data["pairs"], data["split"]
            goal = int(data["goal_id"])
            self.assertGreater(summary["landmark_training_groups"], 0)
            self.assertGreater(int(((pairs[:, 3] == 2) & (pairs[:, 1] != goal) & (split == 0)).sum()), 0)
            successors = set(data["decisions"][:, 2]) - {goal}
            for row, sid in zip(pairs, split):
                if row[1] == goal and row[0] in successors:
                    self.assertEqual(int(sid), 2)


if __name__ == "__main__":
    unittest.main()
