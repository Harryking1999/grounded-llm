"""Fixed-goal supervision must not leak heldout labels via candidates."""
import unittest
from types import SimpleNamespace

import numpy as np
import torch

from experiments.cube2_qmap.src.fixed_goal import balanced_parents, exhaustive_policy, make_batch, path_relations, prepare, reach_within_budget
from experiments.cube2_qmap.src.full_state_joint import batch_losses
from experiments.puzzle_qmap.src.model import QMap
from experiments.cube2_qmap.tests.test_full_state_joint import Ring


class FixedGoalTests(unittest.TestCase):
    def setUp(self):
        self.env = Ring()
        self.env.depth = self.env.distance(np.arange(self.env.size), 0).astype(np.uint8)

    def test_heldout_labels_are_absent_from_both_loss_families(self):
        heldout = np.zeros(self.env.size, dtype=bool)
        heldout[[5, 10]] = True
        parents = np.arange(self.env.size)
        batch = make_batch(self.env, parents, heldout)
        pairs = batch["states"][batch["pairs"]]
        self.assertFalse(heldout[pairs[:, 0]].any())
        np.testing.assert_array_equal(pairs[:, 1], np.zeros(len(pairs)))
        np.testing.assert_array_equal(batch["labels"], self.env.depth[pairs[:, 0]])
        children = batch["states"][batch["children"]]
        expected = self.env.successors(parents[(~heldout[np.column_stack((parents, self.env.successors(parents)))]).all(-1) & (parents != 0)])
        np.testing.assert_array_equal(children, expected)
        np.testing.assert_array_equal(batch["truth"], self.env.depth[children])
        self.assertFalse(heldout[children].any())

    def test_split_and_balanced_replay_keep_every_layer_and_goal_trainable(self):
        env = SimpleNamespace(size=201, diameter=5, depth=np.r_[0, np.repeat(np.arange(1, 6), 40)])
        config = {"data": {"seed": 7, "first_heldout_depth": 4, "heldout_fraction_per_split": .1,
                           "minimum_heldout_per_layer": 0}, "evaluation": {"states_per_layer": 2}}
        mask, layers, cohorts = prepare(env, config)
        self.assertEqual(mask[0], 0)
        self.assertTrue(all(len(layer) for layer in layers))
        self.assertEqual(int((mask == 1).sum()), 8)
        self.assertEqual(int((mask == 2).sum()), 8)
        self.assertTrue((mask[cohorts["validation"]] == 1).all())
        self.assertTrue((mask[cohorts["test"]] == 2).all())
        replay = balanced_parents(layers, 60, 0, np.random.default_rng(5))
        self.assertFalse(mask[replay].any())
        np.testing.assert_array_equal(np.bincount(env.depth[replay])[1:], np.full(5, 12))

    def test_policy_scoring_counts_cycles_and_budget_limits_without_correcting_actions(self):
        policy = np.array([0, 0, 1, 2, 5, 4, 5])
        np.testing.assert_array_equal(reach_within_budget(policy, 0, 2), [0, 1, 2, -1, -1, -1, -1])
        np.testing.assert_array_equal(reach_within_budget(policy, 0, 3), [0, 1, 2, 3, -1, -1, -1])

    def test_intermediate_nonadjacent_pairs_are_exact_and_heldout_endpoints_excluded(self):
        heldout = np.zeros(self.env.size, dtype=bool)
        heldout[3] = True
        pairs, labels = path_relations(self.env, np.array([5]), heldout, np.random.default_rng(7))
        self.assertTrue(np.any((pairs[:, 0] == 5) & (pairs[:, 1] == 2)))
        self.assertFalse(heldout[pairs].any())
        np.testing.assert_array_equal(labels, self.env.distance(*pairs.T))
        batch = make_batch(self.env, np.arange(self.env.size), heldout, np.array([5, 10]), np.random.default_rng(8))
        physical = batch["states"][batch["pairs"]]
        self.assertGreater(batch["path_pair_count"], 0)
        self.assertFalse(heldout[physical].any())
        np.testing.assert_array_equal(batch["labels"], self.env.distance(*physical.T))
        model = QMap(self.env.size, dimension=8, hidden=16, metric="linf", cap=self.env.diameter)
        h, r = batch_losses(model, torch.as_tensor(self.env.features(batch["states"])), batch,
                            {"training": {"temperature": 1.0}})
        (h + r).backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_exhaustive_policy_chooses_from_network_without_oracle_correction(self):
        class ConstantQ:
            def encode(self, features):
                return torch.zeros((len(features), 2))

            def distance(self, a, b):
                return (a - b).abs().amax(-1)
        # Every action ties, so choose the first (+) and follow it unchanged.
        result = exhaustive_policy(ConstantQ(), self.env, np.zeros(self.env.size), {"evaluation": {"horizon": 3}}, "cpu")
        self.assertEqual(result["cohorts"]["all_states"]["solved"], 4)


if __name__ == "__main__":
    unittest.main()
