"""The shared prompt must match the environment and reveal no map distances."""

import unittest

import numpy as np

from experiments.flamingo_map_reader.src.graph import graph_step
from experiments.flamingo_map_reader.src.prompt import initial_prompt, turn_prompt
from test_graph import line_graph


class PromptTest(unittest.TestCase):
    def test_initial_and_turn_follow_shuffled_action_mapping(self):
        environment, qmap = line_graph()
        initial = initial_prompt(environment.adjacency, 1, 2,
                                 node_order=[2, 0, 1],
                                 neighbor_order={0: [1], 1: [2, 0], 2: [1]})
        self.assertEqual(initial,
            "Find a valid path from node 1 to node 2 in this undirected graph, as short as you can.\n"
            "Reaching the goal is the first priority; among valid solutions, prefer fewer moves.\n"
            "Use the listed edges and visit each node at most once.\n\n"
            "Neighbors:\n2: 1\n0: 1\n1: 2, 0")
        self.assertNotIn("<action>", initial)
        step = graph_step(environment, qmap, 1, 2, executed_path=[1],
                          rng=np.random.default_rng(7))
        turn = turn_prompt(step, [1])
        for index, destination in enumerate(step.candidate_destinations, 1):
            self.assertIn(f"{index}: 1 -> {destination}", turn)
        self.assertNotIn("map distance", turn.lower())
        self.assertNotIn("Goal node:", turn)
        self.assertNotIn("<done/>", turn)
        self.assertNotIn("<action>", turn)

    def test_rejects_path_and_graph_mismatch(self):
        environment, qmap = line_graph()
        step = graph_step(environment, qmap, 1, 2, executed_path=[1])
        with self.assertRaisesRegex(ValueError, "actual current node"):
            turn_prompt(step, [0])
        with self.assertRaisesRegex(ValueError, "disagree"):
            initial_prompt(environment.adjacency, 1, 2,
                           neighbor_order={0: [1], 1: [0], 2: [1]})


if __name__ == "__main__":
    unittest.main()
