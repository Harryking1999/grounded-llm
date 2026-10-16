import unittest

import numpy as np

from experiments.cube2_qmap.src.policy_audit import cycle_diagnostics, policy, summarize_policy, two_cycle_basins
from experiments.cube2_qmap.tests.test_full_state_joint import Ring


class PolicyAuditTests(unittest.TestCase):
    def test_exact_basins_distinguish_long_tails_and_other_cycles(self):
        # A long solved tail must not be mistaken for a cycle after 28 steps.
        pointer = np.arange(70, dtype=np.int32)
        pointer[1:61] = np.arange(60)
        pointer[61:65] = [62, 61, 64, 61]
        pointer[65:68] = [66, 67, 65]
        pointer[68:] = [68, 65]
        labels = two_cycle_basins(pointer)
        np.testing.assert_array_equal(labels[:61], -np.ones(61))
        np.testing.assert_array_equal(labels[61:65], np.full(4, 61))
        np.testing.assert_array_equal(labels[65:], np.full(5, -2))

    def test_cycle_report_exposes_complete_candidates_and_keeps_oracle_separate(self):
        class EvenRing(Ring):
            size = 40
        env = EvenRing()
        env.depth = env.distance(np.arange(env.size), 0)
        height = -env.depth.astype(float)
        pointer, oracle, _, _ = policy(env, height)
        report = cycle_diagnostics(env, height, pointer, oracle)
        self.assertEqual(report["two_cycles"], 1)
        self.assertEqual(report["two_cycle_basin_states"], 39)
        trap = report["largest_traps"][0]
        self.assertEqual(trap["true_goal_distances"], [19, 20])
        self.assertEqual(trap["exit_true_goal_distance"], 18)
        self.assertEqual(trap["exit_disadvantage"], 2.)
        self.assertEqual(len(trap["both_complete_candidate_sets"]), 2)
        self.assertEqual(summarize_policy(env, pointer)["solved"], 1)

    def test_oracle_height_completes_and_bad_height_is_not_corrected(self):
        env = Ring()
        env.depth = env.distance(np.arange(env.size), 0)
        exact, oracle, optimal, margins = policy(env, env.depth.astype(float))
        self.assertEqual(summarize_policy(env, exact)["success_rate"], 1.)
        self.assertTrue(optimal[1:].all())
        self.assertTrue((margins[np.isfinite(margins)] > 0).all())
        wrong, same_oracle, _, _ = policy(env, -env.depth.astype(float))
        np.testing.assert_array_equal(same_oracle, oracle)
        self.assertLess(summarize_policy(env, wrong)["success_rate"], .1)
        self.assertGreater(summarize_policy(env, wrong)["two_cycles"], 0)


if __name__ == "__main__":
    unittest.main()
