"""Checks for full-trajectory supervision and Flamingo-style map masks."""

import unittest

import numpy as np
import torch

from experiments.flamingo_map_reader.src.fusion import GatedMapCrossAttention
from experiments.flamingo_map_reader.src.graph import timeline_maps
from experiments.flamingo_map_reader.src.sft import greedy_demonstration
from experiments.flamingo_map_reader.src.transcript import encode_trajectory
from test_graph import line_graph


class ByteChatTemplate:
    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        assert tokenize
        rendered = "".join(
            f"<{message['role']}>{message['content']}</{message['role']}>"
            for message in messages
        )
        if add_generation_prompt:
            rendered += "<assistant>"
        return list(rendered.encode("utf-8"))


class TimelineTest(unittest.TestCase):
    def test_complete_trajectory_labels_answers_and_binds_maps(self):
        environment, qmap = line_graph()
        demonstration = greedy_demonstration(
            environment, qmap, 1, 2, rng=np.random.default_rng(7),
        )
        encoded = encode_trajectory(demonstration, ByteChatTemplate(), 16384)
        rendered = bytes(encoded.input_ids).decode("utf-8")
        self.assertEqual(len(encoded.input_ids), len(encoded.labels))
        self.assertEqual(len(encoded.input_ids), len(encoded.token_map_ids))
        boundary = rendered.index("<user>[Environment update]", 1)
        self.assertTrue(all(index == 0 for index in encoded.token_map_ids[:boundary]))
        self.assertTrue(all(index == 1 for index in encoded.token_map_ids[boundary:]))
        self.assertEqual(sum(label != -100 for label in encoded.labels),
                         encoded.answer_tokens)
        self.assertEqual(encoded.labels[rendered.rindex("<done/>")], ord("<"))
        timeline = timeline_maps([turn.step for turn in demonstration.turns],
                                 encoded.token_map_ids)
        self.assertEqual(tuple(timeline.snapshots.vectors.shape[:2]), (2, 4))
        self.assertEqual(timeline.token_map_ids.shape[1], len(encoded.input_ids))

    def test_later_map_cannot_change_earlier_text_positions(self):
        torch.manual_seed(11)
        attention = GatedMapCrossAttention(language_dim=8, heads=2, head_dim=4)
        attention.gate.data.fill_(1)
        hidden = torch.randn(1, 4, 8)
        memory = torch.randn(1, 2, 3, 8)
        changed = memory.clone()
        changed[:, 1] = torch.randn(1, 3, 8) * 5
        valid = torch.ones((1, 2, 3), dtype=torch.bool)
        turn_ids = torch.tensor([[0, 0, 1, 1]])
        original = attention(hidden, memory, valid, turn_ids)
        modified = attention(hidden, changed, valid, turn_ids)
        self.assertTrue(torch.equal(original[:, :2], modified[:, :2]))
        self.assertFalse(torch.allclose(original[:, 2:], modified[:, 2:]))


if __name__ == "__main__":
    unittest.main()
