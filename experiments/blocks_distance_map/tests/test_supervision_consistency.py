import unittest
import numpy as np
import torch

from experiments.blocks_distance_map.src.model import distance
from experiments.blocks_distance_map.src.supervision_consistency import SIGNS, lower_bounds


class MetricConsistencyTest(unittest.TestCase):
    def test_identity_is_architectural_and_targets_violate_it(self):
        torch.manual_seed(19)
        states = torch.randn(3, 128, dtype=torch.float64)
        pairs = [(0, 1), (1, 0), (1, 2), (2, 1), (0, 2), (2, 0)]
        scores = np.asarray([float(distance(states[a], states[b], "directed_sum")) for a, b in pairs])
        self.assertAlmostEqual(float(scores @ SIGNS), 0, places=12)
        truth = np.asarray([1, -1, 1, -1, 2, -1])
        for cap in (13, 19):
            target = np.where(truth < 0, cap, truth)
            self.assertEqual(int(target @ SIGNS), -cap)
            self.assertGreaterEqual(float(np.abs(scores - target).sum()), cap)
        self.assertAlmostEqual(lower_bounds(19)["six_pair_equal_weight_mean_huber_at_least"], 19 / 6 - .5)


if __name__ == "__main__":
    unittest.main()
