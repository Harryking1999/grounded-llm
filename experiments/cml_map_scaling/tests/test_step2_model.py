"""Small numerical tests; actual Qwen gradient/load checks live in --mode smoke."""
import unittest
import json
from pathlib import Path

import numpy as np

import torch
from torch import nn
from torch.nn import functional as F

from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM

from experiments.cml_map_scaling.src.step2_model import CachedStateInterface, NodeIdGrammar, StateInterface, answer_losses, make_adapter


class Step2LossTest(unittest.TestCase):
    def test_shift_mask_multitoken_and_equal_sample_weight(self):
        torch.manual_seed(42)
        hidden = torch.randn(2, 6, 5, requires_grad=True)
        head = nn.Linear(5, 7, bias=False).requires_grad_(False)
        labels = torch.tensor([[-100, -100, -100, -100, 3, 6], [-100, -100, 1, 2, 3, 6]])
        losses = answer_losses(hidden, labels, head)
        logits = head(hidden).float()
        first = F.cross_entropy(logits[0, 3:5], torch.tensor([3, 6]))
        second = F.cross_entropy(logits[1, 1:5], torch.tensor([1, 2, 3, 6]))
        torch.testing.assert_close(losses, torch.stack((first, second)))
        losses.mean().backward()
        self.assertIsNone(head.weight.grad)
        self.assertEqual(hidden.grad[0, :3].abs().sum().item(), 0)
        self.assertGreater(hidden.grad[0, 3].abs().sum().item(), 0)
        self.assertEqual(hidden.grad[:, -1].abs().sum().item(), 0)

    def test_partial_microbatch_weight_matches_full_batch(self):
        torch.manual_seed(9)
        x = torch.randn(5, 4, 3)
        labels = torch.tensor([[-100, -100, 1, 6]] * 5)
        adapter = nn.Linear(3, 3)
        head = nn.Linear(3, 7, bias=False).requires_grad_(False)
        answer_losses(adapter(x), labels, head).mean().backward()
        full = adapter.weight.grad.clone()
        adapter.zero_grad(set_to_none=True)
        for start in range(0, 5, 2):
            batch = x[start:start + 2]
            (answer_losses(adapter(batch), labels[start:start + 2], head).mean() * len(batch) / 5).backward()
        torch.testing.assert_close(adapter.weight.grad, full)

    def test_continuous_slots_backpropagate_through_frozen_qwen_and_generate(self):
        config = json.loads(Path('experiments/cml_map_scaling/configs/step2.json').read_text())
        config['assets']['state_dim'] = 4
        config['generation']['max_new_tokens'] = 3
        vocab = {'[UNK]': 0, '[PAD]': 1, '<|im_end|>': 2, '<|im_start|>': 3}
        vocab.update({str(i): i + 4 for i in range(32)})
        backend = Tokenizer(models.WordLevel(vocab, unk_token='[UNK]'))
        backend.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
        tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token='[UNK]',
                                            pad_token='[PAD]', eos_token='<|im_end|>', bos_token='<|im_start|>')
        tokenizer.chat_template = ("{% for m in messages %}{{ bos_token + m['role'] + '\\n' + m['content'] + eos_token + '\\n' }}"
                                   "{% endfor %}{% if add_generation_prompt %}{{ bos_token + 'assistant\\n' }}{% endif %}")
        model = Qwen3ForCausalLM(Qwen3Config(vocab_size=40, hidden_size=32, intermediate_size=64,
                                            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                                            head_dim=8, max_position_embeddings=2048, pad_token_id=1, eos_token_id=2))
        interface = StateInterface(config, model, tokenizer, np.eye(4, dtype=np.float32), .5, '0: 1\n1: 0')
        adapter = make_adapter(config, 'linear', 32, 'cpu')
        items = [dict(u=0, g=1, task='report_current', template='canonical'),
                 dict(u=2, g=3, task='report_goal', template='heldout')]
        batch = interface.batch(items, adapter, supervised=True)
        for role, slot in zip(('u', 'g'), interface.slot_ids):
            mask = batch['input_ids'] == slot
            self.assertEqual(mask.sum().item(), 2)
            self.assertTrue(torch.all(batch['labels'][mask] == -100))
            expected = adapter(interface.q[[item[role] for item in items]])
            torch.testing.assert_close(batch['inputs_embeds'][mask], expected)
        loss = interface.losses(items, adapter).mean()
        loss.backward()
        self.assertGreater(adapter.weight.grad.norm().item(), 0)
        self.assertTrue(all(p.grad is None for p in model.parameters()))
        self.assertFalse(model.training)
        self.assertIsNone(interface.q.grad)
        generated = interface.generate(items, adapter)
        self.assertEqual(len(generated), 2)
        self.assertTrue(all(1 <= len(row['generated_ids']) <= 3 for row in generated))
        constrained = interface.generate(items, adapter, prefix_allowed_tokens_fn=NodeIdGrammar(tokenizer, 32))
        self.assertTrue(all(row['raw_output'] in {str(i) for i in range(32)} for row in constrained))
        self.assertTrue(all(row['native_end_seen'] for row in constrained))
        action = dict(u=0, g=3, target=1, task='action', template='canonical')
        with self.assertRaises(ValueError):
            interface.batch([action], adapter, supervised=True)
        config['training']['tasks'].append('action')
        action_batch = interface.batch([action], adapter, supervised=True)
        self.assertEqual(action_batch['labels'][action_batch['labels'] != -100].tolist(),
                         tokenizer.encode('1', add_special_tokens=False) + [tokenizer.eos_token_id])
        cached = CachedStateInterface(config, model, tokenizer, np.eye(4, dtype=np.float32), .5, '0: 1\n1: 0')
        torch.testing.assert_close(cached.losses([action], adapter), interface.losses([action], adapter), rtol=1e-4, atol=1e-4)

    def test_node_id_grammar_handles_shared_digit_prefix_without_state_access(self):
        class Digits:
            eos_token_id = 10
            pad_token_id = 11

            def encode(self, text, **kwargs):
                return [int(character) for character in text]

        grammar = NodeIdGrammar(Digits(), 32)
        self.assertEqual(grammar(0, torch.tensor([], dtype=torch.long)), list(range(10)))
        self.assertEqual(grammar(0, torch.tensor([3])), [0, 1, 10])
        self.assertEqual(grammar(0, torch.tensor([0])), [10])
        self.assertEqual(grammar(0, torch.tensor([1, 2])), [10])
        self.assertEqual(grammar(0, torch.tensor([1, 2, 10])), [11])


if __name__ == '__main__':
    unittest.main()
