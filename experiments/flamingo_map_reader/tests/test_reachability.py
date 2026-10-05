"""The reachability metric must be wider than the shortest-path one, never narrower."""

import unittest

import numpy as np

from experiments.flamingo_map_reader.src.graph import graph_step
from experiments.flamingo_map_reader.src.trajectory_eval import TaskEnvironment
from experiments.flamingo_map_reader.src.trajectory_metrics import summarize_turns
from experiments.external_map_interface.src.q_map import GraphQMap
from experiments.external_map_interface.src.transitions import GraphEnvironment
from test_graph import line_graph


def task_environment(environment, qmap):
    task = object.__new__(TaskEnvironment)
    task.config = {"task": "graph"}
    task.task = "graph"
    task.env = environment
    task.qmap = qmap
    task.distances = None
    return task


def detour_graph():
    """0->1->2 is length 2, 0->3->4->2 is length 3: one good move is not a shortest one."""
    adjacency = np.array([[0, 1, 0, 1, 0],
                          [1, 0, 1, 0, 0],
                          [0, 1, 0, 0, 1],
                          [1, 0, 0, 0, 1],
                          [0, 0, 1, 1, 0]], dtype=bool)
    actions = np.array([[0, 1], [1, 0], [1, 2], [2, 1], [0, 3],
                        [3, 0], [3, 4], [4, 3], [4, 2], [2, 4]])
    qmap = GraphQMap(q=np.zeros((5, 2)), v=np.zeros((len(actions), 2)))
    return GraphEnvironment(adjacency, actions), qmap


def chosen_for(step, destination):
    return step.candidate_destinations.index(destination) + 1


class ReachabilityTest(unittest.TestCase):
    def test_shortest_move_also_keeps_the_goal_reachable(self):
        environment, qmap = line_graph()
        task = task_environment(environment, qmap)
        step = graph_step(environment, qmap, 1, 2, executed_path=[1],
                          rng=np.random.default_rng(7))
        remaining = task.remaining(1, 2, [1])
        chosen = chosen_for(step, 2)
        self.assertTrue(task.chosen_is_shortest(step, chosen, [1], remaining))
        self.assertTrue(task.chosen_keeps_reachable(step, chosen, [1], remaining))

    def test_a_longer_move_counts_as_reachable_but_not_as_shortest(self):
        environment, qmap = detour_graph()
        task = task_environment(environment, qmap)
        step = graph_step(environment, qmap, 0, 2, executed_path=[0],
                          rng=np.random.default_rng(7))
        remaining = task.remaining(0, 2, [0])
        self.assertEqual(remaining, 2)
        detour = chosen_for(step, 3)
        self.assertFalse(task.chosen_is_shortest(step, detour, [0], remaining))
        self.assertTrue(task.chosen_keeps_reachable(step, detour, [0], remaining))
        self.assertEqual(task.reachable_candidates(step, [0], remaining), 2)

    def test_a_move_that_strands_the_goal_counts_as_neither(self):
        environment, qmap = line_graph()
        task = task_environment(environment, qmap)
        step = graph_step(environment, qmap, 1, 2, executed_path=[1],
                          rng=np.random.default_rng(7))
        remaining = task.remaining(1, 2, [1])
        stranded = chosen_for(step, 0)
        self.assertFalse(task.chosen_is_shortest(step, stranded, [1], remaining))
        self.assertFalse(task.chosen_keeps_reachable(step, stranded, [1], remaining))
        # Only the move to 2 survives; the move to 0 cannot come back through 1.
        self.assertEqual(task.reachable_candidates(step, [1], remaining), 1)

    def test_illegal_and_absent_choices_are_not_reachable(self):
        environment, qmap = line_graph()
        task = task_environment(environment, qmap)
        step = graph_step(environment, qmap, 1, 2, executed_path=[1],
                          rng=np.random.default_rng(7))
        for chosen in (None, 0, len(step.candidate_actions) + 1):
            self.assertFalse(task.chosen_keeps_reachable(step, chosen, [1], 1))
        self.assertEqual(task.reachable_candidates(step, [1], 0), 0)

    @staticmethod
    def row(keeps, reachable, slots, remaining, done=False):
        return dict(done=done, action_keeps_goal_reachable=keeps, reachable_candidates=reachable,
                    candidate_slots=slots, remaining_shortest=remaining)

    def test_the_floor_is_the_share_of_legal_candidates_that_survive(self):
        summary = summarize_turns([self.row(True, 1, 2, 3), self.row(False, 1, 2, 3)])
        self.assertEqual(summary["decision_turns"], 2)
        self.assertEqual(summary["solvable_decisions"], 2)
        self.assertEqual(summary["action_keeps_goal_reachable_rate"], 0.5)
        self.assertEqual(summary["reachable_candidate_rate"], 0.5)

    def test_a_board_already_made_dead_is_not_charged_to_the_model(self):
        # The first turn was still solvable and kept the goal reachable. The second
        # is played on a board the loop already killed (remaining < 0) and the third
        # was never scored at all, so neither belongs in the rate or in its floor.
        summary = summarize_turns([self.row(True, 1, 2, 3), self.row(False, 1, 2, -1),
                                   self.row(False, 1, 2, None)])
        self.assertEqual(summary["decision_turns"], 3)
        self.assertEqual(summary["solvable_decisions"], 1)
        self.assertEqual(summary["action_keeps_goal_reachable_rate"], 1.0)
        self.assertEqual(summary["reachable_candidate_rate"], 0.5)

    def test_the_stopping_turn_is_not_a_decision_however_the_field_is_carried(self):
        # Reference shards attach the metric to the stopping turn as well, where
        # remaining is 0 and it is always false. Counting that row as solvable
        # would deflate the rate while its zero slots left the floor untouched,
        # so the two halves of one table row would be read over different rows.
        summary = summarize_turns([self.row(True, 1, 2, 3), self.row(False, 0, 0, 0, done=True)])
        self.assertEqual(summary["decision_turns"], 1)
        self.assertEqual(summary["solvable_decisions"], 1)
        self.assertEqual(summary["action_keeps_goal_reachable_rate"], 1.0)
        self.assertEqual(summary["reachable_candidate_rate"], 0.5)


if __name__ == "__main__":
    unittest.main()
