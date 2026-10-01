"""Counterfactual map slots preserve physical candidates and true Q labels."""

import unittest

import numpy as np
import torch

from experiments.flamingo_map_reader.src.blocks import blocks_step
from experiments.flamingo_map_reader.src.blocks_prompt import (
    initial_prompt as blocks_initial_prompt, turn_prompt as blocks_turn_prompt)
from experiments.flamingo_map_reader.src.evaluate_kv_pilot import (
    physical_action, renumbered_prompt, reorder_candidates, swap_best_worst_q)
from experiments.flamingo_map_reader.src.graph import graph_step
from experiments.flamingo_map_reader.src.prompt import (
    initial_prompt as graph_initial_prompt, turn_prompt as graph_turn_prompt)
from test_blocks import CountingMap
from test_graph import line_graph


class PilotCounterfactualTest(unittest.TestCase):
    def check_step(self, step, prompt, task):
        count = len(step.candidate_actions)
        self.assertGreater(count, 1)
        order = list(reversed(range(count)))
        relabeled = reorder_candidates(step, order)
        self.assertEqual(relabeled.candidate_actions,
                         tuple(step.candidate_actions[i] for i in order))
        for new_id, old_index in enumerate(order, 1):
            self.assertEqual(physical_action(relabeled, new_id),
                             step.candidate_actions[old_index])
            torch.testing.assert_close(relabeled.map_batch.vectors[0, new_id + 1],
                                       step.map_batch.vectors[0, old_index + 2])
        self.assertEqual(relabeled.map_batch.candidate_ids[0].tolist(),
                         [0, 0, *range(1, count + 1)])
        self.assertIn("[Environment update]",
                      renumbered_prompt(task, relabeled, prompt))
        swapped = swap_best_worst_q(step)
        if swapped is None:
            self.assertTrue(np.isclose(min(step.candidate_map_distances),
                                       max(step.candidate_map_distances)))
            return
        torch.testing.assert_close(swapped.map_batch.vectors[:, :2],
                                   step.map_batch.vectors[:, :2])
        torch.testing.assert_close(swapped.map_batch.candidate_ids,
                                   step.map_batch.candidate_ids)
        self.assertEqual(swapped.candidate_actions, step.candidate_actions)
        self.assertNotEqual(swapped.candidate_map_distances,
                            step.candidate_map_distances)

    def test_graph_reindex_and_q_swap(self):
        environment, qmap = line_graph()
        step = graph_step(environment, qmap, 1, 2, executed_path=[1],
                          rng=np.random.default_rng(7))
        prompt = graph_initial_prompt(environment.adjacency, 1, 2)
        prompt += "\n\n" + graph_turn_prompt(step, [1])
        self.check_step(step, prompt, "graph")

    def test_blocks_reindex_and_q_swap(self):
        start = 3 | (3 << 90)
        step = blocks_step(CountingMap(), start, 3, rng=np.random.default_rng(7))
        prompt = blocks_initial_prompt(start, 3)
        prompt += "\n\n" + blocks_turn_prompt(step, [])
        self.check_step(step, prompt, "blocks")


if __name__ == "__main__":
    unittest.main()
