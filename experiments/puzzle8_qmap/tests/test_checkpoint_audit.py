import unittest
from types import SimpleNamespace

import numpy as np

from experiments.puzzle8_qmap.src.checkpoint_audit import cycle_ranking, exposure, same_tasks


class CheckpointAuditTest(unittest.TestCase):
    def test_matched_outcomes_keep_regressions(self):
        self.assertEqual(same_tasks([1, 1, 0, 0], [1, 0, 1, 1]),
                         {"count": 4, "baseline_solved": 2, "candidate_solved": 3,
                          "previously_solved_lost": 1, "previously_failed_gained": 2, "net_gain": 1})

    def test_basin_and_complete_candidates(self):
        env = SimpleNamespace(size=7, actions=("a", "b"),
                              next=np.array([[0, -1], [0, -1], [3, 1], [2, 4], [2, -1], [6, 0], [5, -1]]),
                              distance=lambda ids, goal: np.array([0, 1, 2, 3, 3, 1, 2])[ids])
        state = {"pointer": np.array([0, 0, 3, 2, 2, 6, 5]),
                 "height": np.arange(7) * .5, "within": np.array([1, 1, 0, 0, 0, 0, 0], dtype=bool)}
        rows = cycle_ranking(env, state)
        self.assertEqual([row["basin_states"] for row in rows], [3, 2])
        self.assertEqual({star["parent"] for star in rows[0]["stars"]}, {2, 3})
        self.assertEqual(rows[0]["stars"][0]["candidates"][1]["true_distance"], 1)

    def test_distinct_exposure_does_not_exceed_pool(self):
        values = exposure({"all_pairs_draws": 15, "all_decision_draws": 3},
                          {"available_joint_pairs": 10, "available_joint_decisions": 10})
        self.assertEqual(values["broad_pair_distinct"], 10)
        self.assertEqual(values["broad_decision_distinct_fraction"], .3)


if __name__ == "__main__":
    unittest.main()
