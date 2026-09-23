import json
from pathlib import Path
import unittest

import numpy as np
import torch
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM

from grounded_llm.graph_data import relabel_item, relabel_matrix, validate_node_labels
from grounded_llm.graph_prompts import adjacency_text
from experiments.cml_map_scaling.src.core import shortest_distances, structural_colors
from experiments.cml_map_scaling.src.step2_model import StateInterface, make_adapter
from experiments.cml_map_scaling.src.step2_permutation_report import (
    RelabeledStateInterface, equivariance, permutation_bank, score_report, training_plan,
)


class RelabelingTest(unittest.TestCase):
    def setUp(self):
        self.adj = np.array([[0, 1, 0, 0], [1, 0, 1, 1], [0, 1, 0, 1], [0, 1, 1, 0]], bool)
        self.labels = [2, 0, 3, 1]

    def test_matrix_and_judge_follow_display_ids_without_changing_physical_items(self):
        display = relabel_matrix(self.adj, self.labels)
        for u in range(4):
            for v in range(4):
                self.assertEqual(display[self.labels[u], self.labels[v]], self.adj[u, v])
        item = dict(u=0, g=3, task='report_current', template='canonical')
        generation = dict(raw_output='2', hit_token_limit=False)
        row = score_report(item, generation, self.labels, self.adj, shortest_distances(self.adj))
        self.assertTrue(row['correct']); self.assertEqual(row['physical_prediction'], 0)
        self.assertEqual(row['u'], 2); self.assertEqual(item['u'], 0)
        self.assertEqual(relabel_item(dict(item, target=1), self.labels)['target'], 0)
        with self.assertRaises(ValueError):
            validate_node_labels([0, 0, 2, 3])

    def test_namespace_splits_and_report_budget_are_independent_of_physical_pairs(self):
        identity = list(range(8)); train = permutation_bank(8, 20, 17)
        test = permutation_bank(8, 4, 19, train)
        self.assertFalse(set(map(tuple, train)) & set(map(tuple, test)))
        self.assertNotIn(identity, train + test)
        spec = dict(report_repeats=2, epochs=3, report_batch_size=4, shuffle_seed=22)
        items = [dict(u=0, g=i) for i in range(1, 4)]
        plan = training_plan(items, spec)
        self.assertEqual(sum(len(batch['indices']) for batch in plan), 18)
        self.assertEqual([batch['permutation_index'] for batch in plan], list(range(6)))
        for epoch in range(1, 4):
            self.assertEqual(sorted(i for b in plan if b['epoch'] == epoch for i in b['indices']), [0, 0, 1, 1, 2, 2])
        # Color refinement is a sufficient uniqueness check, not a symmetry oracle.
        permuted_colors = structural_colors(relabel_matrix(self.adj, self.labels))
        colors = structural_colors(self.adj)
        self.assertEqual([permuted_colors[x] for x in self.labels], colors)

    def test_equivariance_does_not_count_consistently_wrong_outputs_as_correct(self):
        common = dict(physical_u=0, physical_g=3, task='report_current', decoding='free')
        rows = [dict(common, group='identity_test_pairs', physical_prediction=1, correct=False),
                dict(common, group='unseen_permutation_test_pairs', physical_prediction=1, correct=False)]
        summary = equivariance(rows)
        self.assertEqual(summary['inverse_label_consistency']['successes'], 1)
        self.assertEqual(summary['both_correct']['successes'], 0)

    def test_cached_prefix_refresh_keeps_q_lookup_and_remapped_supervision_aligned(self):
        config = json.loads(Path('experiments/cml_map_scaling/configs/step2.json').read_text(encoding='utf-8'))
        config['assets'].update(state_dim=4, node_count=4)
        vocab = {'[UNK]': 0, '[PAD]': 1, '<|im_end|>': 2, '<|im_start|>': 3}
        words = [str(i) for i in range(4)] + [f'{i}:' for i in range(4)] + [f'{i},' for i in range(4)]
        vocab.update({word: i + 4 for i, word in enumerate(words)})
        backend = Tokenizer(models.WordLevel(vocab, unk_token='[UNK]'))
        backend.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
        tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token='[UNK]', pad_token='[PAD]', eos_token='<|im_end|>', bos_token='<|im_start|>')
        tokenizer.chat_template = "{% for m in messages %}{{ bos_token + m['role'] + '\n' + m['content'] + eos_token + '\n' }}{% endfor %}{% if add_generation_prompt %}{{ bos_token + 'assistant\n' }}{% endif %}"
        model = Qwen3ForCausalLM(Qwen3Config(vocab_size=40, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                num_attention_heads=4, num_key_value_heads=2, head_dim=8, max_position_embeddings=2048, pad_token_id=1, eos_token_id=2))
        interface = RelabeledStateInterface(config, model, tokenizer, np.eye(4, dtype=np.float32), 1, self.adj, list(range(4)))
        adapter = make_adapter(config, 'linear', 32, 'cpu')
        old_prefix = interface.prefix_ids.copy()
        interface.set_labels(self.labels)
        self.assertNotEqual(interface.prefix_ids, old_prefix)
        self.assertEqual(interface.adjacency, adjacency_text(relabel_matrix(self.adj, self.labels)))
        items = [dict(u=0, g=3, task='report_current', template='canonical'), dict(u=0, g=3, task='report_goal', template='canonical')]
        batch = interface.batch(items, adapter, supervised=True)
        for row, target in enumerate(('2', '1')):
            self.assertEqual(batch['labels'][row][batch['labels'][row] != -100].tolist(), tokenizer.encode(target, add_special_tokens=False) + [tokenizer.eos_token_id])
        for role, slot in zip(('u', 'g'), interface.slot_ids):
            expected = adapter(interface.q[[item[role] for item in items]])
            torch.testing.assert_close(batch['inputs_embeds'][batch['input_ids'] == slot], expected)
        full = StateInterface.losses(interface, items, adapter).mean()
        full.backward(); full_gradient = adapter.weight.grad.clone(); adapter.zero_grad(set_to_none=True)
        cached = interface.losses(items, adapter).mean()
        cached.backward()
        torch.testing.assert_close(cached, full, rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(adapter.weight.grad, full_gradient, rtol=1e-4, atol=1e-4)
        self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))


if __name__ == '__main__':
    unittest.main()
