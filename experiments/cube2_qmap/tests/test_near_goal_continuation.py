import unittest

import numpy as np

from experiments.cube2_qmap.src.near_goal_continuation import ProtectedRelations, sample_tasks, training_tasks
from experiments.cube2_qmap.src.full_state_joint import fixed_batch, task_pairs
from experiments.cube2_qmap.tests.test_full_state_joint import Ring
from experiments.cube2_qmap.src.landmarks import near_route_bank,near_route_tasks,near_segment_bank


class NearGoalTests(unittest.TestCase):
    def test_candidate_closure_and_state_mask_are_excluded_from_both_losses(self):
        env=Ring();mask=np.zeros(env.size,dtype=bool);mask[18]=True
        excluded=task_pairs(env,np.array([[6,2],[20,1]]))
        guard=ProtectedRelations(env,excluded,mask)
        self.assertTrue(guard.contains(np.array([[2,7]]))[0])
        self.assertTrue(guard.task_blocked(env,np.array([[17,3]]))[0])
        pools={1:np.array([1,40]),2:np.array([2,39]),3:np.array([3,38])}
        rng=np.random.default_rng(3)
        for _ in range(3):
            tasks=training_tasks(env,pools,30,rng,guard,0)
            batch=fixed_batch(env,tasks)
            physical=batch['states'][batch['pairs']]
            self.assertFalse(guard.contains(physical).any())
            np.testing.assert_array_equal(batch['labels'],env.distance(*physical.T))
            self.assertFalse(guard.task_blocked(env,tasks).any())
            depths=env.distance(tasks[:,1],0)
            self.assertEqual(list(np.bincount(depths)[1:]),[10,10,10])

    def test_fresh_monitor_stars_do_not_overlap_the_previous_cohort(self):
        env=Ring();mask=np.zeros(env.size,dtype=bool)
        guard=ProtectedRelations(env,task_pairs(env,np.array([[20,2]])),mask)
        tasks=sample_tasks(env,{1:np.array([1,40]),2:np.array([2,39])},12,
                           np.random.default_rng(4),guard)
        self.assertFalse(guard.task_blocked(env,tasks).any())
        self.assertTrue(np.all(tasks[:,0]!=tasks[:,1]))

    def test_local_sampling_covers_exact_entry_distances_without_oracle_actions(self):
        env=Ring();mask=np.zeros(env.size,dtype=bool)
        guard=ProtectedRelations(env,np.empty((0,2),dtype=int),mask)
        relatives={d:np.array([d,env.size-d]) for d in range(1,7)}
        tasks=training_tasks(env,{1:np.array([1,40]),2:np.array([2,39])},120,
                             np.random.default_rng(10),guard,0,relatives,1.)
        distances=env.distance(tasks[:,0],tasks[:,1])
        self.assertEqual(set(distances),set(range(1,7)))
        self.assertTrue(np.all((distances>=1)&(distances<=6)))

    def test_route_bank_keeps_complete_exact_paths_and_every_competitor_exclusion(self):
        env=Ring();env.depth=env.distance(np.arange(env.size),0)
        guard=ProtectedRelations(env,task_pairs(env,np.array([[20,2]])),np.zeros(env.size,dtype=bool))
        bank=near_route_bank(env,{1:np.array([1,40]),2:np.array([2,39])},guard,
            dict(pool_size=12,candidate_batch_size=32,maximum_candidate_batches=20),np.random.default_rng(3))
        tasks=near_route_tasks(bank,4,np.random.default_rng(4))
        self.assertFalse(guard.task_blocked(env,tasks).any())
        for path,length,goal in zip(bank['paths'],bank['lengths'],bank['goals']):
            self.assertEqual(path[length],goal)
            self.assertEqual(length,env.distance(path[0],goal))
            for index in range(length):
                self.assertIn(path[index+1],env.successors(path[index]))
                self.assertEqual(env.distance(path[index+1],goal),length-index-1)

    def test_segment_bank_keeps_contiguous_exact_stars_without_claiming_target_arrival(self):
        env=Ring();env.depth=env.distance(np.arange(env.size),0)
        guard=ProtectedRelations(env,task_pairs(env,np.array([[20,2]])),np.zeros(env.size,dtype=bool))
        bank=near_segment_bank(env,{1:np.array([1,40]),2:np.array([2,39])},guard,
            dict(pool_size=12,candidate_batch_size=32,maximum_candidate_batches=20,
                 minimum_segment_parents=3,maximum_segment_parents=5),np.random.default_rng(5))
        self.assertTrue(np.all((bank['lengths']>=3)&(bank['lengths']<=5)))
        self.assertFalse(guard.task_blocked(env,near_route_tasks(bank,4,np.random.default_rng(4))).any())
        for path,length,goal in zip(bank['paths'],bank['lengths'],bank['goals']):
            for index in range(length):
                self.assertIn(path[index+1],env.successors(path[index]))
                self.assertEqual(env.distance(path[index+1],goal),env.distance(path[index],goal)-1)


if __name__=='__main__':unittest.main()
