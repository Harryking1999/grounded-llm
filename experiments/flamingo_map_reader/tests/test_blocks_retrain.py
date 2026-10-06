"""Failure labels, renumbering, FFN gradients and no-solution evaluation."""

import unittest
from dataclasses import replace
import torch

from experiments.blocks_distance_map.src.oracle import DistanceOracle
from experiments.flamingo_map_reader.src.blocks_failure import (
    failure_demonstration, declares_no_solution, no_solution_terminal)
from experiments.flamingo_map_reader.src.trajectory_dataset import pack_demo, unpack_demo
from experiments.flamingo_map_reader.src.trajectory_protocol import numbering_plans, renumber_demonstration
from experiments.flamingo_map_reader.src.transcript import encode_trajectory
from experiments.flamingo_map_reader.src.memory import AddressedMapMemoryEncoder, JointFeatureMapMemoryEncoder
from experiments.flamingo_map_reader.src.trajectory_metrics import score_turn, summarize_turns
from experiments.flamingo_map_reader.src.evaluate_blocks import parse_control
from test_blocks import CountingMap
from test_timeline import ByteChatTemplate


class RetrainTest(unittest.TestCase):
    def failure(self):
        oracle = DistanceOracle()
        for seed in range(20):
            demo = failure_demonstration(CountingMap(), dict(start=3 | (3 << 90), goal=3,
                sample_seed=seed))
            if demo is not None:
                self.assertGreater(oracle.distance(demo.turns[0].step.current, 3), 0)
                self.assertEqual(oracle.distance(demo.turns[-1].step.current, 3), -1)
                self.assertFalse(demo.turns[-1].step.candidate_actions)
                self.assertFalse(demo.turns[-1].step.done)
                # A wrong move already destroys reachability, but legal moves
                # remain: the new target must wait until those are exhausted.
                intermediate = demo.turns[1].step
                self.assertEqual(oracle.distance(intermediate.current, 3), -1)
                self.assertTrue(intermediate.candidate_actions)
                self.assertFalse(no_solution_terminal(intermediate))
                return demo
        self.fail('Expected one deliberately bad greedy tie choice')

    def test_goal_and_action_budget_are_not_no_solution(self):
        self.assertIsNone(failure_demonstration(CountingMap(),
            dict(start=3, goal=0, sample_seed=0)))
        self.assertIsNone(failure_demonstration(CountingMap(),
            dict(start=3 | (3 << 90), goal=3, sample_seed=0), max_actions=1))

    def test_failure_actions_masked_after_save_and_renumber(self):
        demo = unpack_demo(pack_demo(self.failure()), 'blocks')
        for plan in numbering_plans(demo, 6, 1):
            numbered = renumber_demonstration(demo, plan, 'blocks', 10)
            self.assertFalse(numbered.success)
            self.assertTrue(numbered.no_solution)
            encoded = encode_trajectory(numbered, ByteChatTemplate(), 32768)
            for label, turn in zip(encoded.labels, encoded.token_map_ids):
                if turn < len(numbered.turns)-1:
                    self.assertEqual(label, -100)
            supervised = bytes(x for x in encoded.labels if x >= 0).decode()
            self.assertIn('<action>none</action>\n<done/>', supervised)
            self.assertNotRegex(supervised, r'<action>[0-9]+</action>')
            self.assertEqual(encoded.answer_tokens, sum(x >= 0 for x in encoded.labels))

    def test_feature_ffn_changes_values_only_and_receives_gradients(self):
        step = self.failure().turns[0].step
        encoder = AddressedMapMemoryEncoder(1, 8, 10, 4, 2, feature_ffn_hidden_dim=16)
        memory = encoder(step.map_batch)
        old_keys = memory.keys.detach().clone()
        memory.values.square().sum().backward()
        self.assertGreater(encoder.feature_ffn[0].weight.grad.abs().sum().item(), 0)
        with torch.no_grad():
            encoder.feature_ffn[0].weight.add_(1)
        changed = encoder(step.map_batch)
        torch.testing.assert_close(old_keys, changed.keys)
        self.assertFalse(torch.equal(memory.values, changed.values))

    def test_joint_ffn_makes_both_kv_depend_on_state_and_identity(self):
        batch = self.failure().turns[0].step.map_batch
        encoder = JointFeatureMapMemoryEncoder(1, 8, 10, 4, 2, feature_ffn_hidden_dim=16)
        memory = encoder(batch)
        state_changed = encoder(replace(batch, vectors=batch.vectors + 3))
        torch.testing.assert_close(memory.keys, memory.values)
        self.assertFalse(torch.equal(memory.keys, state_changed.keys))
        self.assertFalse(torch.equal(memory.values, state_changed.values))
        numbered = batch.candidate_ids.clone()
        numbered[:, 2:] = numbered[:, 2:].flip(1)
        id_changed = encoder(replace(batch, candidate_ids=numbered))
        self.assertFalse(torch.equal(memory.values, id_changed.values))
        self.assertFalse(torch.equal(memory.keys, id_changed.keys))
        (memory.keys.square().sum() + memory.values.square().sum()).backward()
        self.assertGreater(encoder.feature_ffn[0].weight.grad.abs().sum().item(), 0)

    def test_no_solution_control_and_metrics(self):
        demo = self.failure()
        row = score_turn(demo.turns[-1].step, demo.turns[-1].answer_text)
        self.assertTrue(row['valid_control'])
        self.assertFalse(row['legal_action'])
        self.assertFalse(row['premature_done'])
        self.assertFalse(declares_no_solution('<action>1</action><done/>'))
        self.assertIsNone(parse_control(demo.turns[-1].answer_text))
        with self.assertRaises(ValueError):
            parse_control('<action>none</action>')
        row.update(no_solution_expected=True, no_solution_correct=True)
        self.assertEqual(summarize_turns([row])['no_solution_recall'], 1)


if __name__ == '__main__':
    unittest.main()
