import unittest

import numpy as np
import torch

from experiments.cube2_qmap.src.landmarks import landmark_batch, greedy_leg
from experiments.cube2_qmap.tests.test_full_state_joint import Ring


class LandmarkTests(unittest.TestCase):
    def setUp(self):
        self.env = Ring()
        self.env.depth = self.env.distance(np.arange(self.env.size), 0)

    def test_all_competitors_receive_true_distance_and_rank_toward_landmark(self):
        heldout = np.zeros(self.env.size, dtype=bool)
        batch = landmark_batch(self.env, [5], heldout, np.random.default_rng(3), goal_depth=2)
        states = batch["states"]
        pairs = states[batch["pairs"]]
        np.testing.assert_array_equal(batch["labels"], self.env.distance(*pairs.T))
        goals = states[batch["goals"]]
        self.assertTrue(np.all(self.env.depth[goals] == 2))
        children = states[batch["children"]]
        np.testing.assert_array_equal(batch["truth"], self.env.distance(children, goals[:, None]))
        self.assertTrue(np.any(batch["truth"][:, 0] != batch["truth"][:, 1]))
        heldout[children[0, 0]] = True
        extra = landmark_batch(self.env, [5], heldout, np.random.default_rng(3), goal_depth=2)
        if extra is not None:
            self.assertFalse(heldout[extra["states"][extra["pairs"]]].any())
            self.assertFalse(heldout[extra["states"][extra["children"]]].any())
            self.assertFalse(heldout[extra["states"][extra["goals"]]].any())

    def test_greedy_leg_never_uses_oracle_to_correct_a_tied_action(self):
        class ConstantQ:
            def encode(self, features):
                return torch.zeros((len(features), 2))

            def distance(self, a, b):
                return (a - b).abs().amax(-1)
        result = greedy_leg(ConstantQ(), self.env, [5], [2], budget=3, entry_depth=2)
        self.assertFalse(result["reached"][0])
        self.assertEqual(result["steps"][0], 3)


if __name__ == "__main__":
    unittest.main()
