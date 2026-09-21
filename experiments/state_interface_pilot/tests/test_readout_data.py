"""Report training spans non-solvable pixel states without leaking board families."""
import unittest
from grounded_llm.artifacts import ROOT
from grounded_llm.blocks import build_blocks_dataset, resolve_blocks_config
from grounded_llm.blocks_readout import (build_readout_dataset, state_key, coverage_summary,
                                        report_training_items, report_question, report_target, spatial_probe_items)
from grounded_llm.config import load_config


class ReadoutDataTests(unittest.TestCase):
    def test_row_target_and_matched_training_presentations(self):
        board = dict(id='a', task='report_board', rows=['001', '110', '010'])
        full = report_training_items([board], dict(seed=19, full_board_repeats=3, rows_per_board=0))
        mixed = report_training_items([board], dict(seed=19, full_board_repeats=1, rows_per_board=2))
        self.assertEqual(len(full), len(mixed))
        self.assertEqual(len({x['report_row'] for x in mixed if x['task'] == 'report_row'}), 2)
        config = {'blocks': {'report_prompt': 'Board', 'row_report_prompt': 'Row {row}'}}
        for item in mixed[1:]:
            self.assertEqual(report_question(config, item), f"Row {item['report_row']}")
            self.assertEqual(report_target(item), '"' + board['rows'][item['report_row']] + '"')
            self.assertEqual(item['rows'], board['rows'])

    def test_spatial_intervention_changes_exactly_one_queried_pixel(self):
        board = dict(id='a', task='report_board', rows=['001', '110', '010'], category='arbitrary')
        first, second = spatial_probe_items([board], dict(seed=29, boards=1))
        self.assertEqual(sum(a != b for a, b in zip(''.join(first['rows']), ''.join(second['rows']))), 1)
        self.assertNotEqual(report_target(first), report_target(second))
        self.assertEqual(board['rows'], ['001', '110', '010'])

    def test_coverage_isolation_and_no_cross_split_duplicates(self):
        raw = load_config('experiments/state_interface_pilot/configs/blocks_smoke.json')
        task, planning = build_blocks_dataset(resolve_blocks_config(raw, ROOT))
        spec = load_config('experiments/state_interface_pilot/configs/blocks_readout.json')['readout_data']
        for split in spec['per_split']:
            spec['per_split'][split] = {k: 4 for k in spec['per_split'][split]}
        dataset = build_readout_dataset(task, planning, spec)
        owners = {state_key(task, task.from_grid(rows)): split for split, cases in planning.items()
                  for c in cases for rows in c['report_grids']}
        for split, items in dataset.items():
            self.assertEqual(len(items), len({tuple(x['rows']) for x in items}))
            for item in items:
                mask = task.from_grid(item['rows'])
                key = state_key(task, mask)
                self.assertEqual(owners.setdefault(key, split), split)
                if item['category'] == 'isolated':
                    self.assertEqual(item['occupied_cells'], item['isolated_cells'])
                if item['category'].startswith('coverage_'):
                    low, high = map(int, item['category'].split('_')[1:])
                    self.assertTrue(low <= mask.bit_count() <= high)
        summary = coverage_summary(dataset)
        self.assertEqual(summary['train']['categories']['empty'], 1)
        self.assertEqual(summary['train']['categories']['full'], 1)
        self.assertNotIn('empty', summary['validation']['categories'])
        self.assertEqual(summary['validation']['categories']['damaged_intermediate'], 4)
        self.assertEqual(dataset, build_readout_dataset(task, planning, spec))


if __name__ == '__main__':
    unittest.main()
