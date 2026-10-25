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

    def test_bridge_all_goals_against_floyd_and_intermediate_stop(self):
        env=Bridge([1,2,5,10]);graph=env.graph();n=len(graph['children'])
        exact=np.full((n,n),np.inf);np.fill_diagonal(exact,0)
        for s in range(n):
            for _,t,c in env.transitions(s):exact[s,t]=min(exact[s,t],c)
        for k in range(n):exact=np.minimum(exact,exact[:,k,None]+exact[None,k,:])
        for goal in range(n):np.testing.assert_array_equal(env.graph(goal)['distances'],exact[:,goal])
        target,cost=env.apply(0,[0,1])
        self.assertTrue(env.judge([[0,1]],goal=target)['optimal'])
        self.assertTrue(env.judge([],start=target,goal=target)['optimal'])

    def test_game24_intermediate_targets_and_variable_capacity(self):
        env=Game24();start=env.canonical([1,2,3,4]);action,goal,_=next(env.transitions(start))
        self.assertTrue(env.judge(start,[env.actions[action]],goal=goal)['success'])
        self.assertTrue(env.judge(goal,[],goal=goal)['success'])
        self.assertFalse(env.judge(start,[],goal=goal)['success'])
        five=Game24(5);state=five.canonical([1,2,3,4,5])
        self.assertEqual(five.features([state]).shape,(1,20))
        self.assertTrue(any(five.actions[a][2]==4 for a,_,_ in five.transitions(state)))

    def test_geometry_encoding_preserves_component_identity_under_renaming(self):
        from grounded_llm.tasks.rush_hour.environment import RushHour
        from grounded_llm.core.qmap.physical_data import enumerate_component
        board='..B...'+ '..B.C.'+'AAB.C.'+'....C.'+'......'+'......'
        renamed=board.translate(str.maketrans({'B':'Z','C':'D'}))
        original=RushHour(board,'geometry_order');other=RushHour(renamed,'geometry_order')
        states,_,_=enumerate_component(original,original.start,10000)
        inverse=str.maketrans({'Z':'B','D':'C'})
        mapped=[tuple(s[original.labels[label.translate(inverse)]]
                      for label,_,_,_ in other.pieces) for s in states]
        np.testing.assert_array_equal(original.features(states),other.features(mapped))
        self.assertTrue(all(original.geometry_channels[0]==0 for _ in states))


if __name__ == '__main__': unittest.main()
