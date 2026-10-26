"""Numerical deduplication must preserve complete arithmetic continuations."""
from contextlib import ExitStack,redirect_stdout
from fractions import Fraction
from pathlib import Path
import tempfile
import unittest
from itertools import combinations
from unittest.mock import patch
from io import StringIO
import json
import numpy as np
import torch
from grounded_llm.core.files import atomic_json
from grounded_llm.core.qmap.physical_data import save_scene
from grounded_llm.tasks.game24.environment import Game24
from grounded_llm.tasks.game24.relation_data import GameScene,GameCorpus,build
from grounded_llm.core.qmap.model import StateEncoder
from experiments.new_tasks_qmap.src.game_finite_evaluate import numerical_queries,main,target_switch_metrics


class FiniteGameTests(unittest.TestCase):
    def test_cached_subdomain_reuses_physics_and_both_held_operation_layers(self):
        with tempfile.TemporaryDirectory() as directory,ExitStack() as stack,redirect_stdout(StringIO()):
            root=Path(directory)
            def close(corpus):
                for scene in corpus.scenes:
                    for key in ['states','children','distances','reverse_distances']:getattr(scene,key)._mmap.close()
            spec=dict(max_numbers=4,initial_numbers=4,initial_sampling='exhaustive_distinct',source_values=[1,2,3,4,5],
                root_targets=[24,21],scenes=dict(train=5,validation=0,test=0),goals_per_scene=16,heldout_goals=4,minimum_reachable_per_goal=2)
            cfg=dict(seed=71,data=spec)
            source=build(cfg,root/'source');stack.callback(close,source)
            small=json.loads(json.dumps(cfg));small['data'].update(source_values=[1,2,3,4],scenes=dict(train=1,validation=0,test=0),
                goals_per_scene=20,heldout_goals=8,heldout_goal_arities=[2,3,2,3,2,3,2,3],graphs_from=str(root/'source'))
            with patch.object(Game24,'graph',side_effect=RuntimeError('Cached physical DAG should be reused')):
                cached=build(small,root/'cached');stack.callback(close,cached)
            fresh=json.loads(json.dumps(small));fresh['data'].pop('graphs_from')
            reference=build(fresh,root/'fresh');stack.callback(close,reference)
            a,b=cached.scenes[0],reference.scenes[0]
            self.assertEqual(a.values,b.values)
            for key in ['states','children','goals','distances','reverse_distances']:np.testing.assert_array_equal(getattr(a,key),getattr(b,key))
            self.assertEqual(a.metadata['training_goals'],12)
            self.assertEqual(a.metadata['goal_layers'][12:16],[2,3,2,3]);self.assertEqual(a.metadata['goal_layers'][16:],[2,3,2,3])
            from experiments.new_tasks_qmap.src.relation_data import board_cohorts,closure
            small.update(task='game24',relations=dict(holdout_goal_endpoints=False,preserve_goal_execution=['24','21'],
                holdout_per_goal=1,evaluation_per_goal=4,unreachable_eval_per_goal=2,complete_training_sources=True,
                evaluation_training_goals=2,exhaustive_training_goals=True))
            groups,_,decisions,reserved,_=board_cohorts(cached,small,np.random.default_rng(8))
            for split in ['validation','test']:
                rows=groups['unseen_goal_'+split]
                self.assertEqual({len(a.values[g]) for g in rows[:,1]},{2,3})
            self.assertFalse(np.intersect1d(closure(cached,decisions),reserved).size)

    def test_deduplication_keeps_exact_reachability_and_both_targets(self):
        with tempfile.TemporaryDirectory() as directory,ExitStack() as stack:
            env=Game24(4);scenes=[];records=[];alphabet=[1,2,3,4,6,8]
            root=Path(directory);graph_root=root/'run/data/graphs';graph_root.mkdir(parents=True)
            for i,numbers in enumerate(combinations(alphabet,4)):
                graph=env.graph(numbers)
                values=[tuple(Fraction(v) for v in row) for row in graph['metadata']['exact_rational_states']]
                children=graph['children']
                for target in (21,24):
                    if (target,) not in values:
                        values.append((Fraction(target),));children=np.vstack([children,np.full((1,len(env.actions)),-1,dtype=np.int32)])
                goals=[values.index((t,)) for t in (21,24)]+[next(j for j,v in enumerate(values) if len(v)==n) for n in (2,3)]
                path=graph_root/str(i)
                record=save_scene(path,np.arange(len(values))[:,None],children,goals,
                    dict(task='game24',split='train',max_numbers=4,training_goals=2,original_state=0,source=dict(numbers=numbers)),state_dtype=np.int32)
                atomic_json(path/'values.json',[[str(v) for v in row] for row in values])
                records.append(dict(path=str(i),**record))
                scene=GameScene(path,record);scenes.append(scene)
                for name in ('states','children','distances','reverse_distances'):stack.callback(getattr(scene,name)._mmap.close)
            corpus=GameCorpus(scenes);exposure=np.ones(corpus.size,dtype=np.int64)
            groups,coverage=numerical_queries(corpus,[21,24],exposure)
            class OracleCache:
                def __init__(self,*args):pass
                def scores(self,s,g):return corpus.distance(s,g)
            class IgnoreGoalCache(OracleCache):
                def scores(self,s,g):
                    s,g=np.broadcast_arrays(s,g);sid=corpus.locate(s)
                    target=np.asarray([int(corpus.offsets[i])+int(scenes[i].goals[1]) for i in sid.ravel()]).reshape(s.shape)
                    return corpus.distance(s,target)
            model=StateEncoder(input_dim=16,dimension=4,hidden=8,cap=16,metric='directed_max')
            with patch('experiments.new_tasks_qmap.src.game_finite_evaluate.QCache',OracleCache):
                switched=target_switch_metrics(model,corpus,groups,4)
            self.assertGreater(switched['count'],0);self.assertEqual(switched['both_complete'],switched['count'])
            self.assertEqual(switched['both_first_actions_optimal'],switched['count'])
            with patch('experiments.new_tasks_qmap.src.game_finite_evaluate.QCache',IgnoreGoalCache):
                ignored=target_switch_metrics(model,corpus,groups,4)
            self.assertEqual(ignored['different_first_actions'],0);self.assertEqual(ignored['both_complete'],0)
            self.assertEqual(coverage['numerical_input_coverage'],1)
            self.assertLess(coverage['exact_numerical_states'],corpus.size)
            for target in (21,24):
                self.assertEqual(len(groups[f'root4_to_{target}']),len(scenes))
                for arity in (2,3):
                    rows=groups[f'suffix{arity}_to_{target}'];seen=[]
                    for source,goal in rows:
                        sid=int(corpus.locate(source));scene=scenes[sid];offset=corpus.offsets[sid]
                        value=scene.values[source-offset];seen.append(value)
                        self.assertEqual(scene.values[goal-offset],(target,))
                    expected={v for scene in scenes for v in scene.values if len(v)==arity}
                    self.assertEqual(len(seen),len(set(seen)));self.assertEqual(set(seen),expected)
                    labels=corpus.distance(*rows.T);finite=np.isfinite(labels)
                    self.assertTrue(finite.any());self.assertTrue((~finite).any())
                    self.assertTrue((labels[finite]==arity-1).all())
            from experiments.new_tasks_qmap.src.relation_data import BoardRelationSampler
            from grounded_llm.core.qmap.sampling import CoverageSampler
            sources=[];queries=[]
            for sid,scene in enumerate(scenes):
                offset=corpus.offsets[sid];local=np.arange(len(scene.states))+offset
                sources.append(np.column_stack([local,np.full(len(local),int(scene.goals[0])+offset)]))
                for j,g in enumerate(scene.goals[:2]):
                    keep=np.flatnonzero(np.isfinite(scene.distances[j])&(scene.distances[j]>0))+offset
                    queries.append(np.column_stack([keep,np.full(len(keep),int(g)+offset)]))
            sampler_data=dict(pairs_train=np.vstack(sources),tasks_train=np.vstack(queries),reserved_relations=np.empty(0,dtype=np.int64))
            sampler=BoardRelationSampler(corpus,sampler_data,dict(finite_point_fraction=.5,unreachable_point_fraction=.25,
                boundary_rank_fraction=1.,distance_bin_upper_bounds=[1,2,3],game_layer_balance=True,balance_layouts=True))
            rng=np.random.default_rng(13);source_sampler=CoverageSampler(np.zeros(corpus.size),rng,corpus.size,1.)
            _,_,point,ranking=sampler.draw(source_sampler,8192,2048,rng)
            point_truth=corpus.distance(*point.T)
            self.assertTrue(np.isinf(point_truth[2048:4096]).all())
            self.assertTrue(np.isfinite(point_truth[4096:]).all());self.assertTrue((point_truth[4096:]>0).all())
            negative_arities=[]
            for source in point[2048:4096,0]:
                sid=int(corpus.locate(source));negative_arities.append(len(scenes[sid].values[source-corpus.offsets[sid]]))
            self.assertGreater(negative_arities.count(4),0)
            self.assertGreater(negative_arities.count(3),200)
            self.assertTrue((corpus.distance(*ranking.T)>0).all())
            # A complete training step must leave broad/negative quotas intact
            # when direct physical edges replace only the finite supplement.
            from experiments.new_tasks_qmap.src.relations import train
            from experiments.new_tasks_qmap.src.relation_data import materialize
            training_data=dict(sampler_data,pair_distances_train=corpus.distance(*sampler_data['pairs_train'].T))
            for name,rows in [('train',sampler_data['tasks_train']),('validation',groups['root4_to_24'])]:
                training_data.update({k+'_'+name:v for k,v in materialize(corpus,rows).items()})
            tiny=dict(task='game24',seed=13,data=dict(max_numbers=4),
                model=dict(dimension=4,hidden=8,cap=16,metric='directed_max'),
                training=dict(steps=1,batch_size=64,decision_batch_size=16,learning_rate=.001,rank_weight=1.,
                    rank_temperature=1.,eval_every=1,cpu_threads=1,unreachable_target=12.,features_on_device='float32',
                    record_training_samples=True,action_edge_fraction=.25,action_edges_replace_finite_only=True,
                    relation_sampling=dict(sampler.options)),
                evaluation=dict(horizon=4,decision_count=16,rollout_count=8))
            with patch('experiments.new_tasks_qmap.src.relations.prepare',return_value=(corpus,training_data,dict(finite_max=3))),redirect_stdout(StringIO()):
                train(tiny,root/'training','fixture','cpu')
            with np.load(next((root/'training/samples/attempt_000').glob('*.npz'))) as trace:
                np.testing.assert_array_equal(trace['point_origin'][:32],np.zeros(32,dtype=np.uint8))
                self.assertTrue(np.isinf(trace['point_distance'][16:32]).all())
                self.assertEqual(int((trace['point_origin']==4).sum()),16)
                self.assertTrue((trace['point_distance'][trace['point_origin']==4]==1).all())
            exposure[:]=0
            _,coverage=numerical_queries(corpus,[21,24],exposure)
            self.assertEqual(coverage['numerical_input_coverage'],0)
            # Exercise the real CLI lifecycle as well as query construction:
            # all cohorts, exact oracle, writing a negative result and cleanup.
            atomic_json(graph_root/'corpus.json',dict(scenes=records))
            held=np.array([[int(corpus.offsets[sid]),int(corpus.offsets[sid])+int(scene.goals[j])]
                for sid,scene in enumerate(scenes) for j in (2,3)],dtype=np.int64)
            np.savez(root/'run/data/relations.npz',tasks_unseen_goal_test=held)
            config=dict(task='game24',seed=1,data=dict(initial_sampling='exhaustive_distinct',initial_numbers=4,source_values=alphabet),
                model=dict(dimension=4,hidden=8,cap=16,metric='directed_max'),
                training=dict(cpu_threads=1,rank_temperature=1,unreachable_target=12),evaluation={})
            model=StateEncoder(input_dim=16,**config['model'])
            torch.save(dict(config=config,input_dim=16,model=model.state_dict(),step=1,source_commit='fixture',exposure=exposure),root/'run/best.pt')
            atomic_json(root/'run/summary.json',dict(evaluated_step=1,metrics=dict(reachability_threshold=6)))
            atomic_json(root/'protocol.json',dict(source_values=alphabet,targets=[21,24],expected_queries={k:len(v) for k,v in groups.items()},
                minimum_completion_rate=.8,target_completion_rate=.9,minimum_unreachable_recall=.8,maximum_finite_false_positive=.2))
            with patch('sys.argv',['evaluate','--protocol',str(root/'protocol.json'),'--run',str(root/'run'),'--out',str(root/'result.json')]),redirect_stdout(StringIO()):main()
            result=json.loads((root/'result.json').read_text())
            self.assertTrue(result['oracle_ok']);self.assertFalse(result['passed'])
            self.assertEqual(result['coverage']['numerical_input_coverage'],0)
            for cohort in result['metrics']['cohorts'].values():
                self.assertEqual(cohort['planning']['oracle']['count'],cohort['finite_count'])


if __name__=='__main__':unittest.main()
