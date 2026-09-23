import copy
import unittest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM
from grounded_llm.parallel_readout import ParallelReadout, model_readout


class ParallelTests(unittest.TestCase):
    def check_equivalence(self, devices):
        torch.manual_seed(7)
        model = Qwen3ForCausalLM(Qwen3Config(vocab_size=32, hidden_size=16, intermediate_size=32,
                  num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1, head_dim=8))
        model.eval().requires_grad_(False).to(devices[0])
        replica = copy.deepcopy(model).to(devices[1])
        adapter = torch.nn.Linear(4, 16).to(devices[0])
        parallel_adapter = copy.deepcopy(adapter)
        pixels = torch.randn(5, 4, device=devices[0])
        attention = torch.tensor([[0,1,1,1], [1,1,1,1], [0,0,1,1], [1,1,1,1], [0,1,1,1]], device=devices[0])
        positions = (attention.cumsum(-1) - 1).masked_fill(attention==0, 0)
        labels = torch.tensor([[-100,-100,-100,3], [-100,-100,4,5], [-100,-100,-100,2],
                               [-100,-100,3,6], [-100,-100,-100,4]], device=devices[0])
        fixed = torch.randn(5, 4, 16, device=devices[0])
        def batch(layer):
            return dict(inputs_embeds=torch.cat((fixed[:, :2], layer(pixels)[:, None], fixed[:, 3:]), dim=1),
                        labels=labels, attention_mask=attention, position_ids=positions)
        reference = model_readout(model, batch(adapter), True)
        executor = ParallelReadout([model, replica])
        actual = executor(batch(parallel_adapter), True)
        torch.testing.assert_close(reference, actual, rtol=1e-5, atol=1e-6)
        reference.mean().backward()
        actual.mean().backward()
        for left, right in zip(adapter.parameters(), parallel_adapter.parameters()):
            torch.testing.assert_close(left.grad, right.grad, rtol=1e-4, atol=1e-6)
        with torch.no_grad():
            torch.testing.assert_close(model_readout(model, batch(adapter), False), executor(batch(adapter), False), rtol=1e-5, atol=1e-6)
        self.assertTrue(all(p.grad is None for m in (model,replica) for p in m.parameters()))

    def test_uneven_shards_preserve_per_answer_losses_and_adapter_gradient(self):
        self.check_equivalence(['cpu', 'cpu'])

    @unittest.skipUnless(torch.cuda.device_count() >= 2, 'Requires two GPUs')
    def test_two_cuda_devices_preserve_adapter_gradient(self):
        self.check_equivalence(['cuda:0', 'cuda:1'])


if __name__ == '__main__':
    unittest.main()
