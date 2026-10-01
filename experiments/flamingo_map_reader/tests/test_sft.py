"""Targeted checks for map-distance labels and terminal trajectories."""

import unittest

import numpy as np

from experiments.external_map_interface.src.q_map import GraphQMap
from experiments.flamingo_map_reader.src.sft import greedy_demonstration
from test_graph import line_graph


class SFTDataTest(unittest.TestCase):
    def test_greedy_trajectory_has_action_and_terminal_map(self):
        environment, qmap = line_graph()
        trajectory = greedy_demonstration(environment, qmap, 1, 2,
                                          rng=np.random.default_rng(7))
        self.assertTrue(trajectory.success)
        self.assertEqual(trajectory.executed_path, (1, 2))
        self.assertEqual(len(trajectory.turns), 2)
        action_turn, final_turn = trajectory.turns
        self.assertNotIn("1.0000", action_turn.answer_text)
        self.assertIn("closest to farthest: 2 < current < 1.", action_turn.answer_text)
        self.assertTrue(action_turn.answer_text.endswith("<action>2</action>"))
        self.assertTrue(final_turn.step.done)
        self.assertEqual(final_turn.step.current, 2)
        self.assertIn("Executed actions: move(1,2).", final_turn.answer_text)
        self.assertTrue(final_turn.answer_text.endswith("<done/>"))
        self.assertEqual(final_turn.user_text.splitlines()[0], "[Environment update]")

    def test_start_at_goal_still_has_model_terminal_turn(self):
        environment, qmap = line_graph()
        trajectory = greedy_demonstration(environment, qmap, 1, 1,
                                          rng=np.random.default_rng(4))
        self.assertTrue(trajectory.success)
        self.assertEqual(len(trajectory.turns), 1)
        self.assertIsNone(trajectory.turns[0].chosen_id)
        self.assertIn("Executed actions: (none).", trajectory.turns[0].answer_text)
        self.assertIn("Summary: Reached the goal after 0 executed moves.",
                      trajectory.turns[0].answer_text)

    def test_bad_greedy_choice_can_end_at_dead_end(self):
        environment, qmap = line_graph()
        wrong_v = qmap.v.copy()
        wrong_v[1] = [1., 0.]
        wrong_v[2] = [-1., 0.]
        wrong_map = GraphQMap(qmap.q, wrong_v)
        trajectory = greedy_demonstration(environment, wrong_map, 1, 2,
                                          rng=np.random.default_rng(7))
        self.assertFalse(trajectory.success)
        self.assertEqual(trajectory.executed_path, (1, 0))
        self.assertEqual(len(trajectory.turns), 1)


if __name__ == "__main__":
    unittest.main()
