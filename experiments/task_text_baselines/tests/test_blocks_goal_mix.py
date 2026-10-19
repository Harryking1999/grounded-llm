import json
import unittest

from experiments.gcml_counterexamples.src import blocks
from experiments.task_text_baselines.src.blocks_goal_mix import certify, isolated_cells
from experiments.task_text_baselines.src.protocol import judge


class GoalMixTests(unittest.TestCase):
    def test_isolation_uses_four_neighbors_without_row_wrapping(self):
        self.assertEqual(isolated_cells((1 << 9) | (1 << 10)), [9, 10])
        self.assertEqual(isolated_cells((1 << 0) | (1 << 1)), [])

    def test_exact_difference_certificate_preserves_isolated_target(self):
        goal = 1 << 99
        source = goal | blocks.PLACEMENTS[blocks.BY_ACTION[(5, 0, 0)]][0]
        case = certify(source, goal, 'nonempty_isolated')
        self.assertEqual(case['reference']['length'], 1)
        self.assertEqual(case['goal_isolated_cells'], [99])
        self.assertIn('Goal board:', case['prompt'])
        wrong = dict(case, goal=0)
        answer = json.dumps({'actions': case['reference']['actions'], 'final_status': 'solved'})
        self.assertTrue(judge(case, answer)['contract_pass'])
        self.assertFalse(judge(wrong, answer)['solved'])

    def test_unreachable_difference_or_added_goal_cell_is_rejected(self):
        with self.assertRaises(ValueError):
            certify(3, 1, 'nonempty_isolated')
        with self.assertRaises(ValueError):
            certify(3, 4, 'nonempty_isolated')


if __name__ == '__main__':
    unittest.main()
