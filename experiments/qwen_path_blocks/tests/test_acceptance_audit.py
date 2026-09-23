"""Protect error attribution from incomplete numbers and exploratory thinking."""
import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from acceptance_audit import final_prefix, audit_records


class PrefixTests(unittest.TestCase):
    def test_truncated_board_keeps_complete_action(self):
        moves, _ = final_prefix('{"actions":[{"shape_id":5,"row":2,"col":3,"board_after":["00', 'blocks8')
        self.assertEqual(moves, [{'shape_id': 5, 'row': 2, 'col': 3}])

    def test_incomplete_number_is_not_a_coordinate(self):
        moves, _ = final_prefix('[{"from":4,"to":25},{"from":25,"to":2', 'path_undirected_256')
        self.assertEqual(moves, [{'from': 4, 'to': 25}])

    def test_does_not_skip_unreadable_element(self):
        moves, _ = final_prefix('[{"from":1,"to":2},garbage,{"from":2,"to":3}]', 'path_undirected_256')
        self.assertEqual(moves, [{'from': 1, 'to': 2}])

    def test_thinking_candidate_does_not_become_final_error(self):
        r = {'case_id': 'x', 'replicate': 1, 'response_status': 'incomplete',
             'raw_output': '', 'reasoning_text': '{"actions":[{"shape_id":5,"row":0,"col":0}]}',
             'verdict': {'pass': False, 'failure_type': 'budget_truncated'}}
        rows = audit_records([r], {'x': {'id': 'x', 'condition': 'blocks8'}})
        self.assertEqual(rows[0]['category'], 'truncated_unresolved')

    def test_illegal_final_prefix_has_priority(self):
        r = {'case_id': 'x', 'replicate': 1, 'response_status': 'incomplete',
             'raw_output': '{"actions":[{"shape_id":5,"row":0,"col":0,"board_after":["0',
             'verdict': {'pass': False, 'failure_type': 'budget_truncated'}}
        case = {'id': 'x', 'condition': 'blocks8', 'grid': '/'.join(['0000000000'] * 10)}
        row = audit_records([r], {'x': case})[0]
        self.assertEqual(row['category'], 'illegal_before_truncation')
        self.assertTrue(row['budget_hit'])
        self.assertFalse(row['pass'])

    def test_legal_detour_can_prove_nonshortest_before_completion(self):
        case = {'id': 'x', 'condition': 'path_undirected_256', 'start': 0, 'goal': 2,
                'neighbors': {'0': [1, 3], '1': [0, 2], '2': [1, 4], '3': [0, 4], '4': [3, 2]},
                'reference': {'length': 2}}
        r = {'case_id': 'x', 'replicate': 1, 'response_status': 'incomplete',
             'raw_output': '[{"from":0,"to":3},{"from":3,"to":',
             'verdict': {'pass': False, 'failure_type': 'budget_truncated'}}
        row = audit_records([r], {'x': case})[0]
        self.assertEqual(row['category'], 'non_shortest_before_truncation')
        self.assertEqual(row['prefix_suboptimal']['shortest_possible_total'], 3)


if __name__ == '__main__':
    unittest.main()
