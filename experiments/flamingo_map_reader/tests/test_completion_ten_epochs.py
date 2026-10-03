"""Check that the two ten-epoch conditions keep Q, IDs, and labels aligned."""

from dataclasses import replace
from unittest import TestCase, mock

import numpy as np
import torch

from experiments.external_map_interface.src.q_map import GraphQMap
from experiments.external_map_interface.src.transitions import GraphEnvironment
from experiments.flamingo_map_reader.src.completion_diagnostics import ordered_pair_correct
from experiments.flamingo_map_reader.src.data import first_turn_from_record
from experiments.flamingo_map_reader.src.readout_aux import CounterfactualFirstTurnDataset, paired_turns
from experiments.flamingo_map_reader.src.relations import score_relationships
from test_focus import ByteFocusTokenizer


class ByteTokenizer(ByteFocusTokenizer):
    def decode(self, ids, **kwargs):
        return bytes(ids).decode("utf-8")


def star_graph():
    adjacency = np.zeros((4, 4), dtype=bool)
    for neighbor in (1, 2, 3):
        adjacency[0, neighbor] = adjacency[neighbor, 0] = True
    actions = np.argwhere(adjacency)
    environment = GraphEnvironment(adjacency, actions)
    q = np.array([[1., 0.], [-1., 0.], [2., 0.], [4., 0.]])
    return environment, GraphQMap(q, q[actions[:, 1]] - q[actions[:, 0]])


class CompletionTenEpochsTest(TestCase):
    def setUp(self):
        self.environment, self.qmap = star_graph()
        self.record = dict(start=0, goal=3, sample_seed=39, graph_id="star", split="train")
        turn = first_turn_from_record(self.environment, self.qmap, self.record)
        nonbest = [i for i in range(1, 4) if i not in turn.step.map_minimal_candidates]
        self.record["candidate_pair"] = nonbest
        self.base = paired_turns(turn, "full_ranking", pair=nonbest)

    def dataset(self, weight=1):
        config = {"task": "graph", "seed": 123, "maximum_sequence_tokens": 18000,
                  "training": {"readout_style": "full_ranking",
                               "candidate_order_augmentation": "per_epoch",
                               "augmentation_seed": 987,
                               "decision_focus_weight": weight}}
        with mock.patch("experiments.flamingo_map_reader.src.readout_aux.load_graph",
                        return_value=(self.environment, self.qmap, {})):
            return CounterfactualFirstTurnDataset([self.record], None, ByteTokenizer(), config)

    def test_swap_and_per_epoch_renumbering_preserve_gold_relationships(self):
        dataset = self.dataset()
        self.assertEqual(self.base[0].chosen_id, self.base[1].chosen_id)
        self.assertTrue(ordered_pair_correct(self.base[0].step, self.base[0].answer_text,
                                             self.record["candidate_pair"]))
        self.assertTrue(ordered_pair_correct(self.base[1].step, self.base[1].answer_text,
                                             self.record["candidate_pair"]))
        seen = set()
        for epoch in range(6):
            dataset.set_epoch(epoch)
            original, swapped = dataset.turns
            self.assertEqual(original.user_text, swapped.user_text)
            self.assertEqual(original.step.candidate_actions, swapped.step.candidate_actions)
            self.assertEqual([int(x) for x in original.step.map_batch.candidate_ids[0, 2:]], [1, 2, 3])
            self.assertTrue(score_relationships(original.step, original.answer_text)["exact_ranking"])
            self.assertTrue(score_relationships(swapped.step, swapped.answer_text)["exact_ranking"])
            self.assertEqual(original.step.candidate_destinations[original.chosen_id - 1], 3)
            for idx, action in enumerate(original.step.candidate_actions):
                source_idx = self.base[0].step.candidate_actions.index(action)
                torch.testing.assert_close(original.step.map_batch.vectors[0, idx + 2],
                                           self.base[0].step.map_batch.vectors[0, source_idx + 2])
            seen.add(original.step.candidate_actions)
            snapshot = original
            dataset.set_epoch(epoch)
            self.assertEqual(dataset.turns[0].answer_text, snapshot.answer_text)
            self.assertEqual(dataset.turns[0].step.candidate_actions, snapshot.step.candidate_actions)
        self.assertGreater(len(seen), 1)

    def test_weight_five_marks_ranking_only(self):
        ordinary = self.dataset()
        weighted = self.dataset(5)
        self.assertIsNone(ordinary[0][0].focus_mask)
        encoded, _ = weighted[0]
        rendered = bytes(encoded.input_ids).decode("utf-8")
        ranking = rendered.index("Map-distance ranking to the goal, closest to farthest: ")
        ranking_content = ranking + len("Map-distance ranking to the goal, closest to farthest: ")
        action = rendered.index("<action>")
        self.assertTrue(encoded.focus_mask[ranking_content])
        self.assertFalse(encoded.focus_mask[ranking])
        self.assertFalse(any(encoded.focus_mask[action:]))
