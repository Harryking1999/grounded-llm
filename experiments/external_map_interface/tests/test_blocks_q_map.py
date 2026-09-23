import unittest

from experiments.external_map_interface.src.blocks_q_map import collect_transitions, train_map
from experiments.sol_dag_blocks.src.tasks import TASKS


class BlocksMapTest(unittest.TestCase):
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
