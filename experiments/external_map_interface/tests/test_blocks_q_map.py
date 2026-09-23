import unittest

import numpy as np

from experiments.external_map_interface.src.blocks_q_map import (
    collect_transitions, coverage_diagnostic, train_map)
from experiments.sol_dag_blocks.src.tasks import TASKS


class BlocksMapTest(unittest.TestCase):
    def test_coverage_separates_trained_action_from_unseen_successor(self):
        class TwoTiles:
            placements = [(1, {}), (2, {})]

        coverage = coverage_diagnostic(TwoTiles(), [0, 3],
                                       np.array([[1, 0, 0]], dtype=np.int32))
        self.assertEqual(coverage["trained_action_rows"], 1)
        self.assertEqual(coverage["legal_candidates_from_seen_states"], 2)
        self.assertEqual(coverage["legal_candidates_with_trained_V"], 1)
        self.assertEqual(coverage["legal_successors_with_tabular_Q"], 0)

    def test_real_actions_share_v_across_board_states(self):
        task = TASKS["blocks8"]
        first_tile, first_action = task.placements[0]
        second_tile, second_action = next(
            (tile, action) for tile, action in task.placements if tile & first_tile == 0)
        initial = first_tile | second_tile
        case = {"grid": "/".join(task.to_rows(initial)),
                "construction_reference": [first_action, second_action]}
        states, actions, transitions = collect_transitions(task, case, 50, 1)
        self.assertIn(0, states)
        self.assertGreaterEqual(len(transitions), 3)
        self.assertLess(len(set(transitions[:, 1])), len(transitions))
        config = {"seed": 1, "state_dim": 8, "q_init_std": 1.0,
                  "v_init_std": 0.1, "epochs": 2, "eta_q": 0.1, "eta_v": 0.01}
        q, v, mse = train_map(states, actions, transitions, config)
        self.assertEqual(q.shape, (len(states), 8))
        self.assertEqual(v.shape, (len(actions), 8))
        self.assertGreaterEqual(mse, 0)


if __name__ == "__main__":
    unittest.main()
