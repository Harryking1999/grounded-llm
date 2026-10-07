"""Map projections remain fixed within one generate call and reset between calls."""

import unittest
from unittest import mock

import torch
from transformers import Qwen2Config, Qwen2ForCausalLM, set_seed

from experiments.flamingo_map_reader.src.fusion import MapReader
from experiments.flamingo_map_reader.src.graph import graph_step
from experiments.flamingo_map_reader.src.memory import JointFeatureMapMemoryEncoder
from experiments.flamingo_map_reader.src.train import collate_examples
from experiments.flamingo_map_reader.src.transcript import EncodedTrajectory
from test_graph import line_graph


class GenerationMapCacheTest(unittest.TestCase):
    def make(self, cached):
        set_seed(7)
        base = Qwen2ForCausalLM(Qwen2Config(vocab_size=32, hidden_size=32,
            intermediate_size=64, num_hidden_layers=2, num_attention_heads=2,
            num_key_value_heads=1, max_position_embeddings=32, pad_token_id=0))
        reader = MapReader(base, JointFeatureMapMemoryEncoder(2, 32, 2, 8, 4, 16),
            2, head_dim=8, cache_generation=cached).eval()
        for layer in reader.conditioned_layers:
            layer.map_attention.gate.data.fill_(0.3)
        return reader

    def inputs(self, current):
        environment, qmap = line_graph()
        first = graph_step(environment, qmap, 1, 2, executed_path=[1])
        last = graph_step(environment, qmap, current, 2, executed_path=[1] if current == 1 else [1, current])
        maps, inputs = collate_examples([(EncodedTrajectory([2, 3, 4, 5],
            [-100, 3, -100, 5], [0, 0, 1, 1], 2), [first, last])], 0)
        inputs.pop('labels')
        return maps, inputs

    def test_greedy_generation_matches_and_projections_reset_per_call(self):
        ordinary, cached = self.make(False), self.make(True)
        layer = cached.conditioned_layers[0]
        for current in (1, 2):
            maps, inputs = self.inputs(current)
            with torch.inference_mode(), mock.patch.object(layer.map_attention.to_key,
                    'forward', wraps=layer.map_attention.to_key.forward) as project:
                expected = ordinary.generate(maps, **inputs, max_new_tokens=5, do_sample=False,
                    return_dict_in_generate=True, output_scores=True)
                actual = cached.generate(maps, **inputs, max_new_tokens=5, do_sample=False,
                    return_dict_in_generate=True, output_scores=True)
                torch.testing.assert_close(expected.sequences, actual.sequences, atol=0, rtol=0)
                for left, right in zip(expected.scores, actual.scores):
                    torch.testing.assert_close(left, right, atol=0, rtol=0)
                self.assertEqual(project.call_count, 1)
            self.assertIsNone(layer._projected_key_value)
            self.assertIsNone(layer._memory)

    def test_training_forward_does_not_use_generation_cache(self):
        reader = self.make(True).train()
        maps, inputs = self.inputs(1)
        inputs['labels'] = inputs['input_ids'].clone()
        loss = reader(maps, **inputs, use_cache=False).loss
        loss.backward()
        for layer in reader.conditioned_layers:
            self.assertIsNone(layer._projected_key_value)
            self.assertGreater(layer.map_attention.to_key.weight.grad.norm().item(), 0)


if __name__ == '__main__':
    unittest.main()
