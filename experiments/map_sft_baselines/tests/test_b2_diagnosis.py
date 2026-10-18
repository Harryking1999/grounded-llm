import json
from pathlib import Path
import unittest

from experiments.map_sft_baselines.src.contract import training_steps, checkpoint_steps
from experiments.map_sft_baselines.src.train import learning_rate_factor
from experiments.map_sft_baselines.src.coverage_subset import select_coverage, decision_pairs
from experiments.map_sft_baselines.src.metrics import summarize


class DiagnosisTests(unittest.TestCase):
    def test_probe_denominators_exclude_missing_donors(self):
        turns=[dict(live=True,rank_eligible=True,text_swap_available=False,
                    training_probe=dict(safe=True,optimal=False)),
               dict(live=True,rank_eligible=True,text_swap_available=True,text_swap_action_changed=True)]
        result=summarize([dict(turns=turns,arrived=False,arrived_and_stopped=False)])
        self.assertEqual(result['text_swap_action_changed']['denominator'],1)
        self.assertEqual(result['text_swap_available']['value'],.5)
        self.assertEqual(result['training_probe_safe']['value'],1.)

    def test_prefix_schedule_is_identical_to_original_50k(self):
        root = Path(__file__).parents[1] / 'configs'
        old = json.loads((root / 'study.json').read_text())
        new = json.loads((root / 'b2_lr_match.json').read_text())
        for step in (0, 1, 2500, 12500, 25000, 49999, 50000):
            self.assertEqual(learning_rate_factor(step, 250000, old['training']),
                             learning_rate_factor(step, 50000, new['training']))

    def test_partial_epoch_stops_at_matched_update_budget(self):
        cfg = dict(data={'samples_per_epoch':30000}, training=dict(
            global_batch=4, epochs=7, maximum_updates=50000, save_steps=12500))
        self.assertEqual(training_steps(cfg), 50000)
        self.assertEqual(list(checkpoint_steps(cfg)), [12500, 25000, 37500, 50000])
        cfg['training']['maximum_updates'] = 53000
        with self.assertRaises(ValueError):
            training_steps(cfg)

    def test_coverage_avoids_duplicates_without_changing_length_or_board_quota(self):
        rows = []
        for board in range(1000):
            for index in range(50):
                # Many duplicate decisions, but enough distinct ones to fill quota.
                state = board * 100 + (index if index < 15 else 1)
                rows.append(dict(board_row=board, trajectory_id=f'{board}_{index}',
                    shortest_moves=6 if index else 0,
                    pairs=[(state, 0)] if index else []))
        chosen = select_coverage(rows, 10000, 42)
        self.assertEqual(chosen, select_coverage(rows, 10000, 42))
        self.assertEqual(len(chosen), 10000)
        self.assertEqual(sum(not row['shortest_moves'] for row in chosen), 200)
        self.assertEqual(len({row['trajectory_id'] for row in chosen}), 10000)
        self.assertEqual(len(set().union(*(decision_pairs(row) for row in chosen))), 9800)


if __name__ == '__main__':
    unittest.main()
