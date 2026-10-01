"""Blocks map reading: shared encoder, directed order, legal execution, OOD."""

from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from experiments.blocks_distance_map.src.model import BoardEncoder
from experiments.gcml_counterexamples.src import blocks as rules
from experiments.flamingo_map_reader.src.blocks import FrozenBoardMap, blocks_step
from experiments.flamingo_map_reader.src.blocks_data import coverage_group, visible_states, record_q_training_coverage
from experiments.flamingo_map_reader.src.evaluate_blocks import parse_control
from experiments.flamingo_map_reader.src.blocks_sft import greedy_demonstration
from experiments.flamingo_map_reader.src.sft import decision_text
from experiments.flamingo_map_reader.src.transcript import encode_trajectory
from test_timeline import ByteChatTemplate


class CountingMap:
    """Test-only explicit representation; production always loads BoardEncoder."""
    metric = "directed_sum"

    def encode(self, masks):
        return torch.tensor([[float(mask.bit_count())] for mask in masks])


class BlocksReaderTest(unittest.TestCase):
    def test_encoder_is_frozen_and_accepts_unseen_board(self):
        qmap = FrozenBoardMap(BoardEncoder(128, 256, 13, "directed_sum"))
        result = qmap.encode([0, 3, (1 << 99)])
        self.assertEqual(tuple(result.shape), (3, 128))
        self.assertFalse(result.requires_grad)
        self.assertTrue(all(not p.requires_grad for p in qmap.encoder.parameters()))

    def test_nonempty_goal_execution_and_explicit_terminal_turn(self):
        # Two isolated horizontal dominoes; either can legally be removed.
        start, goal = 3 | (3 << 90), 3
        qmap = FrozenBoardMap(BoardEncoder(128, 16, 13, "directed_sum"))
        step = blocks_step(qmap, start, goal, rng=np.random.default_rng(3))
        choice = step.candidate_destinations.index(goal) + 1
        action, after = step.execute(choice)
        self.assertEqual(after, goal)
        self.assertTrue(torch.equal(step.map_batch.vectors[0, choice + 1], qmap.encode([goal])[0]))
        final = blocks_step(qmap, after, goal)
        self.assertTrue(final.done)
        self.assertGreater(len(final.candidate_actions), 0)
        with self.assertRaises(ValueError):
            step.execute(0)
        with self.assertRaises(ValueError):
            step.execute(True)
        self.assertEqual(after, rules.apply(start, rules.PLACEMENTS[action][1]))

    def test_complete_trajectory_and_state_visibility(self):
        demo = greedy_demonstration(CountingMap(), 3, 0, rng=np.random.default_rng(7))
        self.assertTrue(demo.success)
        self.assertEqual(demo.executed_path, (3, 0))
        self.assertEqual(visible_states(demo), {3, 0})
        self.assertNotIn("rank", demo.turns[0].user_text.lower())
        self.assertIn("1 < current", demo.turns[0].answer_text)
        self.assertIn("remove(5,0,0)", demo.turns[-1].answer_text)
        encoded = encode_trajectory(demo, ByteChatTemplate(), 16384)
        self.assertEqual(set(encoded.token_map_ids), {0, 1})
        self.assertTrue(demo.turns[-1].answer_text.endswith("<done/>"))

    def test_rank_includes_current_and_keeps_ties_without_numbers(self):
        step = blocks_step(CountingMap(), 3 | (3 << 90), 0)
        step = replace(step, current_map_distance=2.0, candidate_map_distances=(2.0, 2.0),
                       map_minimal_candidates=(1, 2))
        answer = decision_text(step, 2)
        self.assertIn("current = 1 = 2", answer)
        self.assertNotIn("2.0", answer)
        self.assertIn("same map distance", answer)

    def test_ood_uses_candidate_states_too(self):
        seen = {0, 3, 12}
        self.assertIsNone(coverage_group(3, 12, seen))
        self.assertEqual(coverage_group(15, 1, seen), "unseen_start_unseen_goal")
        self.assertEqual(coverage_group(12, 1, seen), "seen_start_unseen_goal")
        self.assertEqual(coverage_group(15, 0, seen), "unseen_start_seen_goal")

    def test_q_coverage_is_separate_from_sft_coverage(self):
        with tempfile.TemporaryDirectory() as directory:
            archive = Path(directory) / "data.npz"
            np.savez(archive, states=np.array(["0", "3", "12", "15"]),
                     train=np.array([[1, 0, 1, 0]]),
                     contrast_train=np.array([[1, 2, 0, 2, 1]]))
            manifest = {"records": [{"start": "15", "goal": "12"}]}
            record_q_training_coverage(manifest, archive)
            record = manifest["records"][0]
            self.assertFalse(record["q_training_start_seen"])
            self.assertTrue(record["q_training_goal_seen"])

    def test_control_rejects_multiple_actions_or_false_mixed_termination(self):
        self.assertEqual(parse_control("Reason. <action>2</action>"), 2)
        self.assertIsNone(parse_control("Finished. <done/>"))
        for answer in ("<action>1</action><done/>", "<action>1</action><action>2</action>", "done"):
            with self.assertRaises(ValueError):
                parse_control(answer)

if __name__ == "__main__":
    unittest.main()
