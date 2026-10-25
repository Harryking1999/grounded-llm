import unittest
from types import SimpleNamespace
import numpy as np
from grounded_llm.core.qmap.physical_data import PhysicalCorpus
from grounded_llm.core.qmap.sampling import CoverageSampler
from experiments.new_tasks_qmap.src.relation_data import BoardRelationSampler,closure


class ReversibleSamplingTests(unittest.TestCase):
    def test_all_decisions_can_have_reachable_and_unreachable_children(self):
        # An operation DAG can have an impossible alternative at every finite
        # decision. A non-boundary quota would require inventing examples.
        children=np.array([[1,3],[2,3],[-1,-1],[-1,-1]])
        distances=np.array([[2,1,0,np.inf]],dtype=np.float32)
        scene=SimpleNamespace(states=np.arange(4),children=children,goals=np.array([2]),distances=distances,
            metadata=dict(split='train',training_goals=1),features=lambda ids:np.eye(4,dtype=np.float32)[ids],
            distance=lambda source,goal:distances[0,source])
        corpus=PhysicalCorpus([scene]);reserved=closure(corpus,np.array([[2,2]]))
        data=dict(pairs_train=np.array([[0,2],[1,2]]),tasks_train=np.array([[0,2],[1,2]]),reserved_relations=reserved)
        options=dict(finite_point_fraction=.5,boundary_rank_fraction=1.,distance_bin_upper_bounds=[1,2,3])
        sampler=BoardRelationSampler(corpus,data,options);rng=np.random.default_rng(61)
        sources=CoverageSampler(np.ones(2),rng,2,1.)
        _,_,pairs,tasks=sampler.draw(sources,32,16,rng)
        self.assertEqual(len(tasks),16);self.assertEqual(len(pairs),32)
        self.assertEqual(sampler.summary['boundary_ranking_bank'],sampler.summary['ranking_bank'])
        self.assertTrue((tasks==[0,2]).all())
        np.testing.assert_array_equal(corpus.successors(tasks[:,0]),np.tile([1,3],(16,1)))
        self.assertFalse(np.intersect1d(closure(corpus,tasks),reserved).size)
        with self.assertRaises(ValueError):BoardRelationSampler(corpus,data,dict(options,boundary_rank_fraction=.5))

    def test_no_dead_boundary_quota_keeps_held_relations_and_complete_candidates(self):
        children=np.array([[1,-1],[0,2],[1,3],[2,4],[3,-1]])
        distances=np.array([[0,1,2,3,4],[4,3,2,1,0]],dtype=np.float32)
        columns=np.array([0,-1,-1,-1,1])
        scene=SimpleNamespace(states=np.arange(5),children=children,goals=np.array([0,4]),distances=distances,
            metadata=dict(split='train',training_goals=2),features=lambda ids:np.eye(5,dtype=np.float32)[ids],
            distance=lambda source,goal:distances[columns[goal],source])
        corpus=PhysicalCorpus([scene]);reserved=closure(corpus,np.array([[2,0]]))
        data=dict(pairs_train=np.array([[0,4],[1,4],[2,4],[3,4],[4,0]]),tasks_train=np.array([[0,4],[1,4],[2,4],[3,4]]),reserved_relations=reserved)
        options=dict(finite_point_fraction=.25,boundary_rank_fraction=0.,distance_bin_upper_bounds=[1,2,4],balance_layouts=True,layouts_per_batch=1)
        sampler=BoardRelationSampler(corpus,data,options);rng=np.random.default_rng(61)
        sources=CoverageSampler(np.ones(5),rng,5,1.)
        _,_,pairs,tasks=sampler.draw(sources,32,16,rng)
        self.assertEqual(sampler.summary['boundary_ranking_bank'],0)
        self.assertEqual(len(pairs),32);self.assertEqual(len(tasks),16)
        self.assertFalse(np.intersect1d(closure(corpus,tasks),reserved).size)
        self.assertTrue(np.isfinite(corpus.distance(pairs[:,0],pairs[:,1])).all())
        np.testing.assert_array_equal(corpus.successors(tasks[:,0]),children[tasks[:,0]])
        with self.assertRaises(ValueError):BoardRelationSampler(corpus,data,dict(options,boundary_rank_fraction=.5))

    def test_parallel_scene_preparation_preserves_banks_and_sampling_rng(self):
        children=np.array([[1,-1],[0,2],[1,3],[2,4],[3,-1]])
        distances=np.array([[0,1,2,3,4],[4,3,2,1,0]],dtype=np.float32);columns=np.array([0,-1,-1,-1,1])
        scene=SimpleNamespace(states=np.arange(5),children=children,goals=np.array([0,4]),distances=distances,
            metadata=dict(split='train',training_goals=2),features=lambda ids:np.eye(5,dtype=np.float32)[ids],
            distance=lambda source,goal:distances[columns[goal],source])
        corpus=PhysicalCorpus([scene,scene]);reserved=closure(corpus,np.array([[2,0],[7,5]]))
        pairs=np.array([[0,4],[1,4],[2,4],[3,4],[4,0],[5,9],[6,9],[7,9],[8,9],[9,5]])
        tasks=pairs[~np.isin(pairs[:,0],[4,9])]
        data=dict(pairs_train=pairs,tasks_train=tasks,reserved_relations=reserved)
        options=dict(finite_point_fraction=.25,boundary_rank_fraction=0.,distance_bin_upper_bounds=[1,2,4],balance_layouts=True,layouts_per_batch=2)
        a=BoardRelationSampler(corpus,data,options);b=BoardRelationSampler(corpus,data,dict(options,preparation_workers=2))
        np.testing.assert_array_equal(a.points,b.points);np.testing.assert_array_equal(a.ranks,b.ranks)
        self.assertEqual(set(map(tuple,a.points)),set(map(tuple,pairs)))
        self.assertEqual(set(map(tuple,a.ranks)),set(map(tuple,tasks)))
        outputs=[]
        for sampler in (a,b):
            rng=np.random.default_rng(63);source=CoverageSampler(np.ones(10),rng,10,1.)
            outputs.append(sampler.draw(source,32,16,rng))
        for x,y in zip(*outputs):np.testing.assert_array_equal(x,y)


if __name__=='__main__':unittest.main()
