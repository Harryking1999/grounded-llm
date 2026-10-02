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
from experiments.flamingo_map_reader.src.readout_aux import paired_turns
from experiments.flamingo_map_reader.src.relations import score_relationships
from experiments.flamingo_map_reader.src.sft import SupervisedTurn, decision_text
from experiments.flamingo_map_reader.src.evaluate_short_readout import (
    correct_ids, diagnostic_prompt, parse_short_answer)
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
            self.assertTrue(len(step.map_minimal_candidates) != 1 or
                np.isclose(min(step.candidate_map_distances),
                           max(step.candidate_map_distances)))
            return
        torch.testing.assert_close(swapped.map_batch.vectors[:, :2],
                                   step.map_batch.vectors[:, :2])
        torch.testing.assert_close(swapped.map_batch.candidate_ids,
                                   step.map_batch.candidate_ids)
        self.assertEqual(swapped.candidate_actions, step.candidate_actions)
        self.assertNotEqual(swapped.candidate_map_distances,
                            step.candidate_map_distances)
        turn = SupervisedTurn(prompt, decision_text(step, step.map_minimal_candidates[0]),
                              step, (step.current,), step.map_minimal_candidates[0])
        paired = paired_turns(turn)
        self.assertEqual(len(paired), 2)
        self.assertEqual(paired[0].user_text, paired[1].user_text)
        self.assertNotEqual(paired[0].step.candidate_map_distances,
                            paired[1].step.candidate_map_distances)
        self.assertTrue(all(score_relationships(item.step, item.answer_text)["exact_ranking"]
                            for item in paired))

    def test_graph_reindex_and_q_swap(self):
        environment, qmap = line_graph()
        step = graph_step(environment, qmap, 1, 2, executed_path=[1],
                          rng=np.random.default_rng(7))
        prompt = graph_initial_prompt(environment.adjacency, 1, 2)
        prompt += "\n\n" + graph_turn_prompt(step, [1])
        self.check_step(step, prompt, "graph")

    def test_blocks_reindex_and_q_swap(self):
        class UnequalMap(CountingMap):
            def encode(self, masks):
                return torch.tensor([[-float(mask.bit_count() + (mask & 1))]
                                     for mask in masks])

        start = 3 | (3 << 90)
        step = blocks_step(UnequalMap(), start, 3, rng=np.random.default_rng(7))
        prompt = blocks_initial_prompt(start, 3)
        prompt += "\n\n" + blocks_turn_prompt(step, [])
        self.check_step(step, prompt, "blocks")
        swapped = swap_best_worst_q(step)
        pair = (step.map_minimal_candidates[0],
                int(np.argmax(step.candidate_map_distances)) + 1)
        question, tag = diagnostic_prompt(prompt, "pairwise", pair)
        self.assertEqual(tag, "closer")
        self.assertIn(f"candidate {pair[0]}", question)
        self.assertNotEqual(correct_ids(step, "pairwise", pair),
                            correct_ids(swapped, "pairwise", pair))
        self.assertEqual(parse_short_answer(f"<closer>{pair[0]}</closer>", tag,
                                            len(step.candidate_actions)), pair[0])
        self.assertIsNone(parse_short_answer("The closer one is candidate 1", tag,
                                              len(step.candidate_actions)))


if __name__ == "__main__":
    unittest.main()
