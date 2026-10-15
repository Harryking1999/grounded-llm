import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from experiments.hanoi_qmap.src.environment import Hanoi
from experiments.puzzle_qmap.src.data import build
from experiments.puzzle_qmap.src.run import train


class ContinuationTest(unittest.TestCase):
    def test_frozen_start_keeps_adam_and_allows_initial_model_to_win(self):
        config = json.loads((Path(__file__).parents[2] / 'hanoi_qmap/configs/study.json').read_text())
        config['training'].update(steps=2, eval_every=1, batch_size=12, decision_batch_size=12,
                                  min_steps=0, patience=None)
        config['evaluation'].update(decision_count=12, rollout_count=12)
        env = Hanoi()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first, second = root/'first', root/'second'
            first.mkdir()
            second.mkdir()
            data, _ = build(env, config, first)
            train(env, config, data, first, 'cpu', 'parent')
            parent = torch.load(first/'best.pt', weights_only=False)
            continued = copy.deepcopy(config)
            continued['training'].update(learning_rate=0., minimum_learning_rate=0., retain_initial=True,
                                          save_last=True, evaluate_initial_test=False, monitor_training=True)
            continued['sampling'] = {'kind':'stratified_epochs','minimum_full_passes':0.01}
            result = train(env, continued, data, second, 'cpu', 'child', first/'best.pt')
            best = torch.load(second/'best.pt', weights_only=False)
            last = torch.load(second/'last.pt', weights_only=False)
            self.assertEqual(result['best_step'], 0)
            self.assertEqual(result['stop_reason'], 'step_budget_exhausted')
            self.assertIsNone(result['initial'])
            self.assertEqual(best['parent']['source_commit'], 'parent')
            self.assertTrue(all(key.startswith('network.') for key in best['model']))
            for key in parent['model']:
                torch.testing.assert_close(parent['model'][key], best['model'][key], rtol=0, atol=0)
                torch.testing.assert_close(parent['model'][key], last['model'][key], rtol=0, atol=0)
            old_state = next(iter(parent['optimizer']['state'].values()))
            new_state = next(iter(last['optimizer']['state'].values()))
            self.assertEqual(float(new_state['step']), float(old_state['step']) + 2)
            self.assertIn('decision_layers', result['last_model_training_coverage'])
            records = [json.loads(line) for line in (second/'progress.jsonl').read_text().splitlines()]
            self.assertIn('training_monitor', records[-1])
            self.assertIn('validation_decisions_by_depth', records[-1])


if __name__ == '__main__':
    unittest.main()
