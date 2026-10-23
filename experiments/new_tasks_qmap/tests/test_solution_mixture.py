"""Check joint split boundaries, identity remapping and matched mixture budgets."""
from copy import deepcopy
import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from experiments.new_tasks_qmap.tests.test_solution_structure import fixture
from experiments.new_tasks_qmap.src.relation_data import materialize,keys
from experiments.new_tasks_qmap.src.solution_structure import context_points
from experiments.new_tasks_qmap.src.solution_mixture import combine_banks,remap_ids,MixtureSampler
from experiments.new_tasks_qmap.src.solution_migration import decision_outcomes,paired_counts


def bank():
    corpus=fixture();tasks=np.array([[0,3],[1,3],[2,3]])
    exact=materialize(corpus,tasks);pairs=np.unique(context_points(tasks,exact['children']),axis=0)
    return corpus,dict(**{k+'_train':v for k,v in exact.items()},pairs_train=pairs,
                       pair_distances_train=corpus.distance(pairs[:,0],pairs[:,1]),reserved_relations=np.array([],np.int64))


class MixtureTests(unittest.TestCase):
    def test_prepared_feature_cache_preserves_training_updates(self):
        from experiments.new_tasks_qmap.src.relations import train
        corpus,original=bank();data={k:np.concatenate([v,v]) for k,v in original.items() if k.endswith('_train')}
        data.update(broad_point_count=len(original['pairs_train']),broad_rank_count=len(original['tasks_train']))
        data.update({k+'_validation':v for k,v in materialize(corpus,np.array([[0,3]])).items()})
        config=dict(task='sokoban',seed=19,model=dict(dimension=128,hidden=16,cap=256,metric='directed_max'),
            training=dict(steps=2,batch_size=64,decision_batch_size=8,learning_rate=.0001,rank_weight=1.,
                rank_temperature=1.,eval_every=2,cpu_threads=1,unreachable_target=240.,evaluation_checkpoint='last.pt',
                solution_structure=dict(mode='broad_graph_mixture',graph_fraction=.5,graphs_per_batch=2,
                                        neighborhood_size=3,root_layer_sampling='uniform')),
            evaluation=dict(decision_count=8,rollout_count=1,horizon=32))
        saved=[]
        def prepare(cfg,path):
            path.mkdir()
            np.save(path/'feature_ids.npy',np.arange(corpus.size))
            np.save(path/'features_packed.npy',np.packbits(corpus.features(np.arange(corpus.size)).astype(np.uint8),axis=1,bitorder='little'))
            return corpus,data,dict(finite_max=2)
        with tempfile.TemporaryDirectory() as tmp,patch('experiments.new_tasks_qmap.src.solution_structure.prepare',side_effect=prepare):
            for mode in ['packed_bits','prepared_packed_bits']:
                out=Path(tmp)/mode
                cfg=deepcopy(config);cfg['training']['features_on_device']=mode
                train(cfg,out,'mixture-fixture','cpu');saved.append(torch.load(out/'last.pt',weights_only=False))
        for key,value in saved[0]['model'].items():torch.testing.assert_close(value,saved[1]['model'][key],rtol=0,atol=0)
        np.testing.assert_array_equal(saved[0]['rank_draws'],saved[1]['rank_draws'])

    def test_physical_identity_remap_preserves_padding_and_rejects_changed_scene(self):
        small=[dict(path='a',states=5,initial_board=['a'])]
        full=[dict(path='b',states=2,initial_board=['b']),small[0]]
        np.testing.assert_array_equal(remap_ids(np.array([[0,4,-1]]),small,full),[[2,6,-1]])
        altered=deepcopy(full);altered[1]['states']=6
        with self.assertRaisesRegex(ValueError,'identity'):remap_ids(np.array([0]),small,altered)

    def test_protected_child_rejects_whole_context_in_both_pools(self):
        corpus,graph=bank();records=[dict(path='a',states=5,initial_board=['a'],split='train')]
        reserved=keys(np.array([4]),np.array([3]),5)
        combined,audit=combine_banks(graph,graph,records,records,reserved)
        self.assertEqual(audit['broad']['decisions_removed'],1)
        self.assertEqual(audit['graph']['decisions_removed'],1)
        for row,children in zip(combined['tasks_train'],combined['children_train']):
            np.testing.assert_array_equal(children,corpus.successors(row[:1])[0])
        self.assertFalse((combined['tasks_train'][:,0]==1).any())
        self.assertFalse(((combined['pairs_train']==[4,3]).all(axis=1)).any())

    def test_all_mixture_arms_match_budgets_and_graph_candidate_points(self):
        corpus,original=bank();data={k:np.concatenate([v,v]) for k,v in original.items() if k.endswith('_train')}
        data.update(broad_point_count=len(original['pairs_train']),broad_rank_count=len(original['tasks_train']))
        for fraction in [0,.25,.5,1]:
            rng=np.random.default_rng(31);options=dict(graph_fraction=fraction,graphs_per_batch=2,neighborhood_size=3,root_layer_sampling='uniform')
            sampler=MixtureSampler(corpus,data,options,rng);state=deepcopy(rng.bit_generator.state)
            p,r=sampler.draw(64,8);self.assertEqual(len(p),64);self.assertEqual(len(r),8)
            self.assertEqual(int((p>=data['broad_point_count']).sum()),round(64*fraction))
            self.assertEqual(int((r>=data['broad_rank_count']).sum()),round(8*fraction))
            graph=r[r>=data['broad_rank_count']]
            for row in context_points(data['tasks_train'][graph],data['children_train'][graph]):
                self.assertIn(tuple(row),set(map(tuple,data['pairs_train'][p[p>=data['broad_point_count']]])))
            rng.bit_generator.state=state;a,b=sampler.draw(64,8)
            np.testing.assert_array_equal(p,a);np.testing.assert_array_equal(r,b)
            if fraction:self.assertEqual(len(sampler.graph.strata[3]),2)

    def test_complete_order_is_distinct_from_best_action(self):
        data=dict(children_train=np.array([[1,2,3]]),distances_train=np.array([2]),child_distances_train=np.array([[1,2,np.inf]]))
        outcomes=decision_outcomes(data,np.array([[1,3,2]]))
        self.assertTrue(outcomes['shortest_action'][0]);self.assertFalse(outcomes['complete_candidate_order'][0])
        self.assertEqual(paired_counts(np.array([True,False]),np.array([False,True]))['introduced'],1)


if __name__=='__main__':unittest.main()
