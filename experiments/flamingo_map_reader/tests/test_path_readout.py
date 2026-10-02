"""Check arbitrary-pair labels, map binding and held-out goal boundaries."""

from dataclasses import replace
import unittest

import numpy as np
import torch

from experiments.external_map_interface.src.q_map import GraphQMap
from experiments.external_map_interface.src.transitions import GraphEnvironment
from experiments.flamingo_map_reader.src.counterfactual import swap_candidate_q
from experiments.flamingo_map_reader.src.data import first_turn_from_record, demonstration_from_record
from experiments.flamingo_map_reader.src.prepare_path_readout import select_records
from experiments.flamingo_map_reader.src.readout_aux import paired_turns
from test_graph import line_graph


class ExpandedReadoutTest(unittest.TestCase):
    def test_first_turn_reproduces_existing_prompt_ids_and_vectors(self):
        environment, qmap = line_graph()
        record = dict(start=0, goal=2, sample_seed=39)
        first = first_turn_from_record(environment, qmap, record)
        original = demonstration_from_record(environment, qmap, record).turns[0]
        self.assertEqual(first.user_text, original.user_text)
        self.assertEqual(first.answer_text, original.answer_text)
        torch.testing.assert_close(first.step.map_batch.vectors, original.step.map_batch.vectors)

    def test_arbitrary_pair_swap_reverses_label_without_changing_question(self):
        environment, qmap = line_graph()
        turn = first_turn_from_record(environment, qmap, dict(start=1, goal=2, sample_seed=39))
        pair = (2, 1)
        original, swapped = paired_turns(turn, "pairwise", pair=pair)
        self.assertEqual(original.user_text, swapped.user_text)
        self.assertNotEqual(original.chosen_id, swapped.chosen_id)
        self.assertEqual(original.step.candidate_actions, swapped.step.candidate_actions)
        torch.testing.assert_close(original.step.map_batch.vectors[:, :2], swapped.step.map_batch.vectors[:, :2])
        restored = swap_candidate_q(swapped.step, pair)
        torch.testing.assert_close(restored.map_batch.vectors, original.step.map_batch.vectors)
        tied = replace(turn.step, candidate_map_distances=(1., 1.))
        self.assertIsNone(swap_candidate_q(tied, pair))
        with self.assertRaises(ValueError):
            swap_candidate_q(turn.step, (1, 1))

    def test_goal_split_precedes_augmentation_and_reserves_old_pairs(self):
        # A cycle offers multiple starts and goals at a fixed distance.
        count = 12
        adjacency = np.zeros((count, count), dtype=bool)
        for i in range(count):
            adjacency[i, (i+1) % count] = adjacency[(i+1) % count, i] = True
        actions = np.argwhere(adjacency)
        environment = GraphEnvironment(adjacency, actions)
        q = np.arange(count, dtype=float)[:, None]
        qmap = GraphQMap(q, q[actions[:, 1]] - q[actions[:, 0]])
        config = {"unseen_test_graph": "other", "readout_data": {
            "held_out_goals_per_graph": 4,
            "train_per_length_per_graph": {"2": 4},
            "validation_per_length_per_graph": {"2": 3},
            "training_evaluation_per_length_per_graph": {"2": 2}}}
        prior = [dict(start=0, goal=2, split="train")]
        records, partitions = select_records(environment, qmap, {"cases": []}, "g", config, prior, 13)
        self.assertNotIn(2, partitions["validation"])
        self.assertFalse(set(partitions["train"]) & set(partitions["validation"]))
        self.assertEqual(len(records), 7)
        self.assertFalse({(0, 2), (2, 0)} & {(r["start"], r["goal"]) for r in records})
        self.assertEqual(sum(r["evaluate"] for r in records), 5)
        repeated, _ = select_records(environment, qmap, {"cases": []}, "g", config, prior, 13)
        self.assertEqual(records, repeated)


if __name__ == "__main__":
    unittest.main()
