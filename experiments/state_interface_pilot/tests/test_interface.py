"""Cheap tensor checks using a tiny random Qwen; no downloaded weights."""
import unittest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM
from grounded_llm.blocks_interface import BlocksInterface, CompleteObject
from grounded_llm.interface import replace_vectors
from grounded_llm.training import answer_losses


class TensorTests(unittest.TestCase):
    def test_multiple_state_slots_gradients_and_answer_mask(self):
        model = Qwen3ForCausalLM(Qwen3Config(vocab_size=32, hidden_size=16, intermediate_size=32,
                    num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1, head_dim=8))
        model.eval().requires_grad_(False)
        interface = object.__new__(BlocksInterface)
        interface.config = {'generation': {'context_limit': 64}}
        interface.device, interface.dtype = torch.device('cpu'), torch.float32
        interface.model, interface.pad_id, interface.slot_id = model, 0, 99
        adapter = torch.nn.Linear(4, 16)
        batch = interface.tensor_batch([[1, 99, 2, 99, 3, 4], [1, 99, 3, 4]],
                 [[['01', '10'], ['00', '10']], [['11', '00']]], adapter, [[3, 4], [3, 4]])
        self.assertTrue(torch.all(batch['labels'][batch['input_ids'] == 99] == -100))
        self.assertTrue(torch.all(batch['labels'][batch['attention_mask'] == 0] == -100))
        output = model.model(inputs_embeds=batch['inputs_embeds'], attention_mask=batch['attention_mask'],
                             position_ids=batch['position_ids'], use_cache=False)
        loss = answer_losses(output.last_hidden_state, batch['labels'], model.lm_head).mean()
        loss.backward()
        self.assertGreater(float(adapter.weight.grad.abs().sum()), 0)
        self.assertTrue(all(p.grad is None for p in model.parameters()))
        with self.assertRaises(ValueError):
            replace_vectors(torch.zeros(1, 2, 3), torch.tensor([[True, True]]), torch.zeros(1, 3))

    def test_json_boundary_does_not_consult_legality(self):
        class Tokenizer:
            value = '{"shape_id":999,"board_after":["}"]}'
            def decode(self, *args, **kwargs):
                return self.value
        tokenizer = Tokenizer()
        stop = CompleteObject(tokenizer)
        self.assertTrue(stop(torch.tensor([[1]]), None))
        tokenizer.value = '{"shape_id":'
        self.assertFalse(stop(torch.tensor([[1]]), None))


if __name__ == '__main__':
    unittest.main()
