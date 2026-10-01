"""Decision-token weighting keeps the full trajectory and exact map timeline."""

import re
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
from torch.nn import functional as F
from transformers import Qwen2Config, Qwen2ForCausalLM, TrainingArguments

from experiments.flamingo_map_reader.src.fusion import MapReader
from experiments.flamingo_map_reader.src.graph import graph_step
from experiments.flamingo_map_reader.src.memory import MapMemoryEncoder
from experiments.flamingo_map_reader.src.sft import greedy_demonstration
from experiments.flamingo_map_reader.src.train import (MapCollator, MapSFTTrainer,
    collate_examples, decision_weighted_loss)
from experiments.flamingo_map_reader.src.transcript import EncodedTrajectory, encode_trajectory
from test_graph import line_graph
from test_timeline import ByteChatTemplate


class ByteFocusTokenizer(ByteChatTemplate):
    def __call__(self, text, *, add_special_tokens, return_offsets_mapping=False):
        assert not add_special_tokens
        raw = text.encode("utf-8")
        result = {"input_ids": list(raw)}
        if return_offsets_mapping:
            result["offset_mapping"] = [(i, i + 1) for i in range(len(raw))]
        return result


class FocusTest(unittest.TestCase):
    def test_only_variable_decision_spans_are_focused(self):
        environment, qmap = line_graph()
        demo = greedy_demonstration(environment, qmap, 1, 2,
                                    rng=np.random.default_rng(7))
        encoded = encode_trajectory(demo, ByteFocusTokenizer(), 16384,
                                    focus_decisions=True)
        text = bytes(encoded.input_ids).decode("utf-8")
        ranking = re.search(r"Map-distance ranking to the goal, closest to farthest: (.+)\.", text)
        action = re.search(r"<action>(\d+)</action>", text)
        self.assertEqual(len(encoded.focus_mask), len(encoded.input_ids))
        self.assertTrue(all(encoded.focus_mask[ranking.start(1):ranking.end(1)]))
        self.assertTrue(encoded.focus_mask[action.start(1)])
        self.assertFalse(encoded.focus_mask[ranking.start()])
        self.assertFalse(encoded.focus_mask[action.start()])
        self.assertFalse(any(encoded.focus_mask[text.index("<done/>"):]))
        self.assertTrue(all(not focus or label != -100
                            for focus, label in zip(encoded.focus_mask, encoded.labels)))
        _, batch = collate_examples([(encoded, [turn.step for turn in demo.turns])], 0)
        self.assertEqual(batch["focus_mask"].shape, batch["labels"].shape)

    def test_weighted_loss_matches_explicit_per_token_calculation(self):
        logits = torch.tensor([[[0.2, 1.0, -0.2], [0.4, 0.1, 1.2],
                                [1.3, -0.5, 0.1], [0.0, 0.0, 0.0]]],
                              requires_grad=True)
        labels = torch.tensor([[-100, 1, 2, 0]])
        focused = torch.tensor([[False, False, True, False]])
        per_token = F.cross_entropy(logits[0, :-1], labels[0, 1:], reduction="none")
        ordinary = per_token.mean()
        weighted = decision_weighted_loss(ordinary, logits, labels, focused, 5)
        torch.testing.assert_close(weighted,
                                   (per_token[0] + 5 * per_token[1] + per_token[2]) / 7)
        weighted.backward()
        self.assertIsNotNone(logits.grad)

    def test_trainer_uses_focused_batch_without_changing_map_timeline(self):
        environment, qmap = line_graph()
        step = graph_step(environment, qmap, 1, 2, executed_path=[1])
        encoded = EncodedTrajectory([2, 3, 4, 5], [-100, -100, 4, 5],
                                    [0] * 4, 2, [False, False, True, False])
        base = Qwen2ForCausalLM(Qwen2Config(vocab_size=32, hidden_size=32,
            intermediate_size=64, num_hidden_layers=1, num_attention_heads=2,
            num_key_value_heads=1, max_position_embeddings=32, pad_token_id=0))
        base.config.use_cache = False
        reader = MapReader(base, MapMemoryEncoder(2, 32, 2, 8, 4), 2, head_dim=8)
        with tempfile.TemporaryDirectory() as directory:
            args = TrainingArguments(output_dir=str(Path(directory) / "run"), use_cpu=True,
                per_device_train_batch_size=1, max_steps=1, save_strategy="no",
                remove_unused_columns=False, label_names=["labels"],
                report_to=[], disable_tqdm=True, dataloader_pin_memory=False)
            trainer = MapSFTTrainer(model=reader, args=args,
                train_dataset=[(encoded, [step])], data_collator=MapCollator(0),
                contract={"config": {"training": {"decision_focus_weight": 5}}})
            result = trainer.train()
            self.assertEqual(result.global_step, 1)
            self.assertTrue(torch.isfinite(torch.tensor(result.training_loss)))


if __name__ == "__main__":
    unittest.main()
