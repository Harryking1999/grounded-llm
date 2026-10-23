import unittest
import numpy as np
from experiments.cube2_qmap.src.coverage_data import coverage,CoverageGuard
from experiments.cube2_qmap.src.full_state_joint import task_pairs
from experiments.cube2_qmap.tests.test_full_state_joint import Ring
from experiments.cube2_qmap.src.coverage_train import take_cycle
from experiments.cube2_qmap.src.coverage_evaluate import paired_change


class CoverageTests(unittest.TestCase):
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
        self.assertTrue(guard.task_blocked(env,np.array([[10,2]]))[0])
        self.assertFalse(guard.task_blocked(env,np.array([[20,8]]))[0])


if __name__=='__main__':unittest.main()
