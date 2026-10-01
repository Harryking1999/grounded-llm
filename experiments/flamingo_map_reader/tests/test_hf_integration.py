"""Smoke the adapter against Hugging Face Qwen2 and Qwen3 layer signatures."""

import unittest

import torch

from experiments.flamingo_map_reader.src.fusion import MapReader
from experiments.flamingo_map_reader.src.memory import MapBatch, MapMemoryEncoder, MapTimeline
from experiments.flamingo_map_reader.src.train import collate_examples
from experiments.flamingo_map_reader.src.transcript import EncodedTrajectory
from experiments.flamingo_map_reader.src.graph import graph_step
from test_graph import line_graph


class QwenIntegrationTest(unittest.TestCase):
    def test_real_batch_isolates_maps_and_updates_only_new_parameters(self):
        from transformers import Qwen2Config, Qwen2ForCausalLM
        torch.manual_seed(17)
        base = Qwen2ForCausalLM(Qwen2Config(vocab_size=32, hidden_size=32,
            intermediate_size=64, num_hidden_layers=2, num_attention_heads=2,
            num_key_value_heads=1, max_position_embeddings=32, pad_token_id=0))
        reader = MapReader(base, MapMemoryEncoder(2, 32, 2, 8, 4), 2, head_dim=8).eval()
        for layer in reader.conditioned_layers:
            layer.map_attention.gate.data.fill_(0.5)
        environment, qmap = line_graph()
        first = graph_step(environment, qmap, 1, 2, executed_path=[1])
        second = graph_step(environment, qmap, 2, 2, executed_path=[1, 2])
        examples = [
            (EncodedTrajectory([2, 3, 4, 5], [-100, 3, -100, 5], [0, 0, 1, 1], 2), [first, second]),
            (EncodedTrajectory([6, 7], [-100, 7], [0, 0], 1), [second]),
        ]
        timeline, inputs = collate_examples(examples, 0)
        labels = inputs.pop("labels")
        logits = reader(timeline, **inputs, use_cache=False).logits
        for index, example in enumerate(examples):
            single, tokens = collate_examples([example], 0)
            tokens.pop("labels")
            expected = reader(single, **tokens, use_cache=False).logits
            torch.testing.assert_close(logits[index, :expected.shape[1]], expected[0], atol=1e-5, rtol=1e-5)
        self.assertEqual(labels[1].tolist(), [-100, 7, -100, -100])
        loss = reader(timeline, **inputs, labels=labels, use_cache=False).loss
        loss.backward()
        self.assertIsNotNone(reader.memory_encoder.project.weight.grad)
        self.assertTrue(all(p.grad is None for p in reader.base_model.parameters() if not p.requires_grad))

    def check_model(self, base):
        torch.manual_seed(5)
        base = base.eval()
        tokens = torch.tensor([[2, 3, 4]])
        original = base(input_ids=tokens, use_cache=False).logits.detach().clone()
        reader = MapReader(base, MapMemoryEncoder(2, 64, 2, 16, 8), heads=2,
                           head_dim=16).eval()
        map_batch = MapBatch(
            vectors=torch.tensor([[[0., 0.], [1., 0.], [1., 0.]]]),
            roles=torch.tensor([[0, 1, 2]]),
            candidate_ids=torch.tensor([[0, 0, 1]]),
            valid=torch.tensor([[True, True, True]]),
        )
        adapted = reader(map_batch, input_ids=tokens, use_cache=False).logits
        self.assertTrue(torch.equal(original, adapted))
        for layer in reader.conditioned_layers:
            layer.map_attention.gate.data.fill_(0.5)
        changed = reader(map_batch, input_ids=tokens, use_cache=False).logits
        self.assertFalse(torch.equal(original, changed))
        generated = reader.generate(map_batch, input_ids=tokens, max_new_tokens=2,
                                    do_sample=False)
        self.assertEqual(generated.shape, (1, 5))

    def test_tiny_qwen2_forward_and_generation(self):
        try:
            from transformers import Qwen2Config, Qwen2ForCausalLM
        except ImportError:
            self.skipTest("transformers with Qwen2 is not installed")
        config = Qwen2Config(
            vocab_size=128, hidden_size=64, intermediate_size=128,
            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
            max_position_embeddings=64, pad_token_id=0, eos_token_id=1,
        )
        self.check_model(Qwen2ForCausalLM(config))

    def test_tiny_qwen3_forward_and_generation(self):
        try:
            from transformers import Qwen3Config, Qwen3ForCausalLM
        except ImportError:
            self.skipTest("transformers with Qwen3 is not installed")
        config = Qwen3Config(
            vocab_size=128, hidden_size=64, intermediate_size=128,
            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
            head_dim=16, max_position_embeddings=64, pad_token_id=0, eos_token_id=1,
        )
        self.check_model(Qwen3ForCausalLM(config))

    def test_tiny_qwen2_full_timeline_forward_and_generation(self):
        try:
            from transformers import Qwen2Config, Qwen2ForCausalLM
        except ImportError:
            self.skipTest("transformers with Qwen2 is not installed")
        base = Qwen2ForCausalLM(Qwen2Config(
            vocab_size=128, hidden_size=64, intermediate_size=128,
            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
            max_position_embeddings=64, pad_token_id=0, eos_token_id=1,
        )).eval()
        reader = MapReader(base, MapMemoryEncoder(2, 64, 2, 16, 8),
                           heads=2, head_dim=16).eval()
        timeline = MapTimeline(
            snapshots=MapBatch(
                vectors=torch.tensor([
                    [[0., 0.], [1., 0.], [0., 1.]],
                    [[1., 0.], [0., 1.], [1., 1.]],
                ]),
                roles=torch.tensor([[0, 1, 2], [0, 1, 2]]),
                candidate_ids=torch.tensor([[0, 0, 1], [0, 0, 1]]),
                valid=torch.ones((2, 3), dtype=torch.bool),
            ),
            token_map_ids=torch.tensor([[0, 0, 1, 1]]),
        )
        tokens = torch.tensor([[2, 3, 4, 5]])
        logits = reader(timeline, input_ids=tokens, use_cache=False).logits
        self.assertEqual(tuple(logits.shape), (1, 4, 128))
        generated = reader.generate(timeline, input_ids=tokens,
                                    max_new_tokens=2, do_sample=False)
        self.assertEqual(tuple(generated.shape), (1, 6))


if __name__ == "__main__":
    unittest.main()
