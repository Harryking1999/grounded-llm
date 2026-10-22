"""Regressions for weighted actions, irreversible dead ends and rational arithmetic."""
from fractions import Fraction
import unittest
import numpy as np
from grounded_llm.tasks.bridge.environment import Bridge
from grounded_llm.tasks.sokoban.environment import Sokoban
from grounded_llm.tasks.game24.environment import Game24


class RulesTest(unittest.TestCase):
    def test_bridge_optimum_and_judge(self):
        env = Bridge([1, 2, 5, 10])
        self.assertEqual(env.graph()['distances'][0], 17)
        result = env.judge([(0, 1), (0,), (2, 3), (1,), (0, 1)])
        self.assertTrue(result['optimal'])
        self.assertFalse(env.judge([(0, 1), (2, 3)])['success'])

    def test_bridge_weighted_bellman_and_reverse(self):
        graph = Bridge([1, 2, 5, 10]).graph()
        for s, row in enumerate(graph['children']):
            for a in np.flatnonzero(row >= 0):
                t = row[a]
                self.assertEqual(graph['children'][t, a], s)
            if np.isfinite(graph['distances'][s]) and not graph['terminal'][s]:
                valid = row >= 0
                self.assertEqual(np.min(graph['costs'][s, valid] + graph['distances'][row[valid]]), graph['distances'][s])

    def test_sokoban_push_and_no_pull(self):
        env = Sokoban(['#####', '#@$.#', '#####'])
        self.assertTrue(env.judge(['R'])['success'])
        child = next(env.transitions(env.start))[1]
        self.assertEqual(list(env.transitions(child))[0][1][1], child[1])
        self.assertFalse(env.judge(['U'])['success'])

    def test_sokoban_corner_is_legal_dead_end(self):
        env = Sokoban(['#####', '#   #', '# $@#', '# . #', '#####'])
        # Left then up from underneath can strand a box in a non-goal corner.
        graph = env.graph()
        self.assertGreater(np.isinf(graph['distances']).sum(), 0)
        finite = np.isfinite(graph['distances'])
        row_ids, actions = np.where(graph['children'] >= 0)
        targets = graph['children'][row_ids, actions]
        self.assertTrue(np.any(finite[row_ids] & ~finite[targets]))

    def test_game24_fractional_solution(self):
        env = Game24()
        graph = env.graph([3, 3, 8, 8])
        self.assertEqual(graph['distances'][0], 3)
        # 8 / (3 - 8 / 3), re-sort after each operation.
        self.assertTrue(env.judge([3, 3, 8, 8], [(0, 'r/', 2), (0, 'r-', 1), (0, 'r/', 1)])['success'])
        self.assertFalse(env.judge([3, 3, 8, 8], [])['success'])

    def test_game24_zero_division_and_duplicate_slots(self):
        env = Game24()
        state = env.canonical([0, 1, 1])
        actions = [env.actions[a] for a, _, _ in env.transitions(state)]
        self.assertNotIn((0, 'r/', 1), actions)
        self.assertIn((1, '-', 2), actions)
        self.assertTrue(all(len(t) == 2 for _, t, _ in env.transitions(state)))

    def test_game24_unsolvable_and_depth_degeneracy(self):
        env = Game24()
        self.assertTrue(np.isinf(env.graph([1, 1, 1, 1])['distances'][0]))
        graph = env.graph([1, 3, 4, 6])
        for values, d in zip(graph['metadata']['exact_rational_states'], graph['distances']):
            if np.isfinite(d): self.assertEqual(d, len(values) - 1)

    def test_game24_exact_fraction(self):
        env = Game24()
        self.assertEqual(env.canonical([Fraction(1, 3), Fraction(2, 6)]), (Fraction(1, 3),) * 2)


if __name__ == '__main__': unittest.main()
