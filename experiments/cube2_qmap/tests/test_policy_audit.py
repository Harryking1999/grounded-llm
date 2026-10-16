import unittest

import numpy as np

from experiments.cube2_qmap.src.policy_audit import policy, summarize_policy
from experiments.cube2_qmap.tests.test_full_state_joint import Ring


class PolicyAuditTests(unittest.TestCase):
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
