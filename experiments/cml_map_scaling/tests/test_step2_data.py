import json
import unittest
from pathlib import Path

import numpy as np

from experiments.cml_map_scaling.src.core import shortest_distances
from experiments.cml_map_scaling.src.step2_data import (
    examples, judge, make_split, messages, parse_answer, summarize, swap_diagnostics,
)
from grounded_llm.graph_data import make_split as canonical_make_split
from grounded_llm.graph_scoring import judge as canonical_judge


class Step2DataTest(unittest.TestCase):
    def setUp(self):
        self.config = json.loads(Path('experiments/cml_map_scaling/configs/step2.json').read_text())
        self.adj = np.array([[0, 1, 1, 0], [1, 0, 0, 1], [1, 0, 0, 1], [0, 1, 1, 0]], bool)
        self.distances = shortest_distances(self.adj)

    def test_split_covers_nodes_without_reverse_pair_leakage(self):
        self.assertIs(make_split, canonical_make_split)
        self.assertIs(judge, canonical_judge)
        split = make_split(self.config)
        self.assertEqual(split, make_split(self.config))
        self.assertEqual([len(examples(split[s])) for s in ('train', 'validation', 'test')], [1584, 200, 200])
        ordered_sets = [{(r['u'], r['g']) for r in examples(split[s])} for s in split]
        self.assertEqual(sum(map(len, ordered_sets)), len(set.union(*ordered_sets)))

    def test_parser_rejects_repairs_and_noncanonical_ids(self):
        for raw in ('01', '+1', '-1', '1 2', '{"node":1}', 'Node 1', '１', '1.0', ''):
            self.assertFalse(parse_answer(raw, 32)['format_valid'], raw)
        self.assertTrue(parse_answer('  31\n', 32)['node_id_valid'])
        self.assertTrue(parse_answer('32', 32)['format_valid'])
        self.assertFalse(parse_answer('32', 32)['node_id_valid'])

    def test_all_shortest_successors_and_legality_are_distinct(self):
        item = dict(u=0, g=3, task='action', template='canonical')
        for successor in ('1', '2'):
            self.assertTrue(judge(item, successor, self.adj, self.distances)['correct'])
        self.assertFalse(judge(item, '3', self.adj, self.distances)['action_legal'])
        away = judge(dict(item, g=1), '2', self.adj, self.distances)
        self.assertTrue(away['action_legal'])
        self.assertFalse(away['correct'])

    def test_slot_order_changes_without_changing_semantics(self):
        item = dict(u=4, g=19, task='report_current', template='heldout')
        content = messages(self.config, '0: 1', item)[1]['content']
        self.assertLess(content.index('<|cml_goal_state|>'), content.index('<|cml_current_state|>'))
        text = messages(self.config, '0: 1', item, True)[1]['content']
        self.assertIn('Goal node: 19\nCurrent node: 4', text)

    def test_error_denominators_conditional_actions_and_swap_labels(self):
        items = examples([[0, 3]], tasks=('report_current', 'report_goal', 'action'))
        outputs = ['0', '3', '1', 'wrong', '0', '2']
        rows = [judge(i, s, self.adj, self.distances) for i, s in zip(items, outputs)]
        metrics = summarize(rows)
        self.assertEqual(metrics['current_report_exact_accuracy']['total'], 2)
        self.assertEqual(metrics['both_reports_correct_rate']['successes'], 1)
        self.assertEqual(metrics['action_success_when_both_reports_correct']['total'], 1)
        swapped = swap_diagnostics(rows, self.adj, self.distances)['records']
        goal = next(r for r in swapped if r['u'] == 0 and r['task'] == 'report_goal')
        self.assertFalse(goal['correct_against_original_labels'])
        self.assertTrue(goal['correct_against_swapped_labels'])


if __name__ == '__main__':
    unittest.main()
