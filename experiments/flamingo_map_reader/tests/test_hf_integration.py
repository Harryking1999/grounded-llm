"""Smoke the adapter against Hugging Face Qwen2 and Qwen3 layer signatures."""

import unittest

import torch

from experiments.flamingo_map_reader.src.fusion import MapReader
from experiments.flamingo_map_reader.src.memory import MapBatch, MapMemoryEncoder


class QwenIntegrationTest(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
