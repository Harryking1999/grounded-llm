import unittest
import numpy as np
from experiments.cube2_qmap.src.coverage_data import coverage,CoverageGuard
from experiments.cube2_qmap.src.full_state_joint import task_pairs
from experiments.cube2_qmap.tests.test_full_state_joint import Ring
from experiments.cube2_qmap.src.coverage_train import take_cycle,Pool,GoalPool
from experiments.cube2_qmap.src.coverage_evaluate import paired_change
from experiments.cube2_qmap.src.goal_layers import reverse_layers,real_path_edges


class CoverageTests(unittest.TestCase):
    def test_path_metrics_exclude_terminal_and_padding_transitions(self):
        mask=real_path_edges(np.array([1,3]),5)
        self.assertTrue(np.array_equal(mask,np.array([[True,False,False,False],[True,True,True,False]])))
        self.assertEqual(int(mask.sum()),4)

    def test_layer_sampler_balances_goals_and_layers_and_counts_repeats(self):
        env=Ring();rows=np.array([[3,1],[4,1],[20,1],[3,2],[15,2],[16,2]])
        pool=GoalPool(rows,env,5,True);drawn=pool.draw(30)
        self.assertEqual(int((drawn[:,1]==1).sum()),15)
        self.assertTrue((pool.seen==5).all())

    def test_reverse_bfs_matches_graph_distances_and_every_optimal_descent(self):
        env=Ring();goal=10;distance=reverse_layers(env,goal)
        self.assertTrue(np.array_equal(distance,env.distance(np.arange(env.size),goal)))
        sources=np.flatnonzero(distance>0)
        self.assertTrue((distance[env.successors(sources)]==distance[sources,None]-1).any(-1).all())

    def test_grouped_cross_batches_keep_all_targets_and_visit_every_group(self):
        rows=np.column_stack((np.repeat(np.arange(10),8),np.tile(np.arange(8),10)))
        pool=Pool(rows,42,True);drawn=np.r_[pool.draw(32),pool.draw(32),pool.draw(16)]
        self.assertEqual(len(np.unique(drawn,axis=0)),80)
        self.assertTrue((pool.seen==1).all())
        for group in drawn.reshape(-1,8,2):
            self.assertEqual(len(np.unique(group[:,0])),1)
            self.assertEqual(len(np.unique(group[:,1])),8)

    def test_paired_counts_preserve_improvements_and_regressions(self):
        r=paired_change([True,False,False,True],[True,True,False,False])
        self.assertEqual((r['gained'],r['lost'],r['net']),(1,1,0))
        self.assertEqual(r['gained_indices'],[1])
        self.assertEqual(r['lost_indices'],[3])
        self.assertEqual(r['exact_mcnemar_two_sided_p'],1.)

    def test_cycle_boundary_never_drops_unseen_parents(self):
        rng=np.random.default_rng(5);order=rng.permutation(10)
        first,order,cursor,passed=take_cycle(order,0,6,rng)
        second,new_order,cursor,passed=take_cycle(order,cursor,6,rng)
        self.assertEqual(len(np.unique(np.r_[first,second[:4]])),10)
        self.assertTrue(np.array_equal(second[4:],new_order[:2]))
        self.assertEqual((cursor,passed),(2,1))

    def test_same_start_incompatible_goals_and_repeats_are_separate_counts(self):
        env=Ring();env.depth=env.distance(np.arange(env.size),0)
        rows=np.array([[10,1],[10,20],[10,1],[11,1]])
        r=coverage(env,rows)
        self.assertEqual(r['exposures'],4)
        self.assertEqual(r['unique_complete_decisions'],3)
        self.assertEqual(r['unique_parents'],2)
        self.assertEqual(r['parents_requiring_goal_dependent_action'],1)
        self.assertEqual(r['unique_candidate_labels'],6)

    def test_withheld_goal_never_enters_candidate_or_distance_labels(self):
        env=Ring();guard=CoverageGuard(env,task_pairs(env,np.array([[3,10]])),
                                      np.zeros(env.size,dtype=bool),[5])
        self.assertTrue(guard.task_blocked(env,np.array([[20,5]]))[0])
        self.assertTrue(guard.task_blocked(env,np.array([[4,20]]))[0])
        self.assertFalse(guard.contains(np.array([[4,20]]))[0])
        self.assertTrue(guard.task_blocked(env,np.array([[10,2]]))[0])
        self.assertFalse(guard.task_blocked(env,np.array([[20,8]]))[0])


if __name__=='__main__':unittest.main()
