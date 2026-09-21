"""Contract tests: no recovery, exact reporting and strictly matched continuations."""
import copy
import json
import tempfile
from pathlib import Path
import unittest

from grounded_llm.artifacts import ROOT, write_json, append_json
from grounded_llm.blocks import (build_blocks_dataset, planning_prompt, resolve_blocks_config,
                                score_step, strict_json, summarize_blocks, collect_blocks_runs)
from grounded_llm.blocks_run import episode
from grounded_llm.config import load_config


class BlocksTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = resolve_blocks_config(load_config('experiments/state_interface_pilot/configs/blocks_smoke.json'), ROOT)
        cls.task, cls.dataset = build_blocks_dataset(cls.config)

    def test_witnesses_solve_and_splits_do_not_leak(self):
        owners = {}
        for split, cases in self.dataset.items():
            for case in cases:
                mask = self.task.from_grid(case['grid'])
                for action in case['construction_reference']:
                    mask = self.task.apply(mask, action)
                self.assertEqual(mask, 0)
                for rows in case['report_grids']:
                    key = self.task.normalized_key(self.task.from_grid(rows))
                    self.assertEqual(owners.setdefault(key, split), split)

    def test_exact_report_and_illegal_action_are_distinct(self):
        case = self.dataset['test'][0]
        mask = self.task.from_grid(case['grid'])
        action = dict(case['construction_reference'][0])
        after = self.task.apply(mask, action)
        action['board_after'] = self.task.to_rows(after)
        self.assertTrue(score_step(self.task, mask, json.dumps(action))['accepted'])
        repeated = score_step(self.task, after, json.dumps(action))
        self.assertEqual(repeated['failure'], 'illegal_action')
        action['board_after'] = self.task.to_rows(mask)
        wrong = score_step(self.task, mask, json.dumps(action))
        self.assertTrue(wrong['action_legal'])
        self.assertFalse(wrong['accepted'])
        self.assertEqual(wrong['failure'], 'state_report_error')
        action['shape_id'] = True
        self.assertEqual(score_step(self.task, mask, json.dumps(action))['failure'], 'illegal_action')
        self.assertEqual(score_step(self.task, mask, '{"done":true}')['failure'], 'premature_done')
        with self.assertRaises(ValueError):
            strict_json('{"row":1,"row":2}')

    def test_inherited_numeric_contract_and_shard_validation(self):
        self.assertIsInstance(self.config['training']['epsilon'], float)
        self.assertEqual(self.config['assets'], {'state_dim': 100})
        raw = load_config('experiments/state_interface_pilot/configs/blocks_smoke.json')
        raw['execution'] = dict(phase='eval', shard_index=2, num_shards=2)
        with self.assertRaises(ValueError):
            resolve_blocks_config(raw, ROOT)

    def test_rollout_injects_only_after_correct_step(self):
        case = self.dataset['test'][0]
        task = self.task
        mask = task.from_grid(case['grid'])
        outputs = []
        for action in case['construction_reference']:
            mask = task.apply(mask, action)
            outputs.append(json.dumps(dict(action, board_after=task.to_rows(mask))))

        class FakeInterface:
            end_id = -1
            def __init__(self, values):
                self.values, self.calls = iter(values), []
            def chat(self, text):
                return [1]
            def state_suffix(self, enabled):
                return [3, 4] if enabled else [3]
            def next_step(self, ids, states, adapter, budget):
                self.calls.append(copy.deepcopy(states))
                return [5], next(self.values)

        for enabled in (False, True):
            interface = FakeInterface(outputs)
            row = episode(interface, object() if enabled else None, task, case, self.config)
            self.assertTrue(row['solved'])
            self.assertEqual(len(interface.calls), len(outputs))
            self.assertEqual([len(s) for s in interface.calls], list(range(1, len(outputs) + 1)) if enabled else [0] * len(outputs))
            broken = json.loads(outputs[0])
            broken['board_after'] = task.to_rows(task.from_grid(case['grid']))
            interface = FakeInterface([json.dumps(broken)])
            row = episode(interface, object() if enabled else None, task, case, self.config)
            self.assertEqual(row['failure'], 'state_report_error')
            self.assertEqual(len(interface.calls), 1)
            self.assertEqual(row['accepted_steps'], 0)
            self.assertEqual(row['legal_actions'], 1)

    def test_prompt_has_no_witness_or_old_full_plan_protocol(self):
        prompt = planning_prompt(self.task, self.dataset['test'][0], self.config)
        self.assertNotIn('Plan the COMPLETE', prompt)
        self.assertNotIn('construction_reference', prompt)
        self.assertIn('board_after', prompt)
        self.assertIn('0-cells are holes', prompt)

    def test_collector_requires_complete_unique_pairs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            training, shard = root / 'train', root / 'shard'
            training.mkdir()
            shard.mkdir()
            for directory in (training, shard):
                write_json(directory / 'config.json', self.config)
            write_json(training / 'dataset.json', {'test': [{'id': 'a'}]})
            for name in ('readout.json', 'training_cost.json'):
                write_json(training / name, {})
            (training / 'adapter').mkdir()
            write_json(training / 'adapter/training_summary.json', {})
            row = dict(id='a', difficulty=2, solved=False, failure='illegal_action', steps=[],
                       legal_actions=0, accepted_steps=0, generated_tokens=1, prefill_tokens=2, seconds=1)
            append_json(shard / 'episodes.jsonl', dict(row, condition='text'))
            with self.assertRaises(ValueError):
                collect_blocks_runs(training, [shard], root / 'missing')
            append_json(shard / 'episodes.jsonl', dict(row, condition='text_token'))
            self.assertEqual(collect_blocks_runs(training, [shard], root / 'complete')['episodes'], 2)
            with self.assertRaises(ValueError):
                collect_blocks_runs(training, [shard, shard], root / 'duplicate')


if __name__ == '__main__':
    unittest.main()
