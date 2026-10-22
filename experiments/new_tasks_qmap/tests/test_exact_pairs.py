"""Independent graph checks, leakage boundaries and oracle-free reader loading."""
from collections import deque
from contextlib import ExitStack
from itertools import combinations,permutations
import json
from pathlib import Path
import tempfile
import shutil
import unittest
from unittest.mock import patch
import numpy as np
import torch
from grounded_llm.core.files import atomic_json
from grounded_llm.core.qmap.exact_graph import UnitGraphOracle
from grounded_llm.core.qmap.model import StateEncoder
from grounded_llm.core.qmap.physical_data import enumerate_component,save_scene,PhysicalScene,PhysicalCorpus
from grounded_llm.core.reader.memory import CURRENT,GOAL,SUCCESSOR,AddressedMapMemoryEncoder
from grounded_llm.tasks.sokoban.environment import Sokoban
from grounded_llm.tasks.rush_hour.environment import RushHour
from grounded_llm.tasks.token_swapping.environment import TokenSwapping
from grounded_llm.tasks.token_swapping.relation_data import TokenCorpus,rank
from experiments.new_tasks_qmap.src.relation_data import token_cohorts,closure,keys,materialize,sample_goals
from experiments.new_tasks_qmap.src.relation_evaluate import rollout,QCache
from experiments.new_tasks_qmap.src.relations import train


def small_oracle(path,n,edges):
    states=list(permutations(range(n))); index={s:i for i,s in enumerate(states)}
    children=np.asarray([[index[tuple(s[:a]+(s[b],)+s[a+1:b]+(s[a],)+s[b+1:])] for a,b in edges] for s in states],dtype=np.int32)
    distances={tuple(range(n)):0}; queue=deque(distances)
    while queue:
        state=queue.popleft()
        for a,b in edges:
            child=list(state); child[a],child[b]=child[b],child[a]; child=tuple(child)
            if child not in distances:
                distances[child]=distances[state]+1; queue.append(child)
    np.asarray(states,dtype=np.uint8).tofile(path/'states.bin')
    children.astype('<i4').tofile(path/'children.bin')
    np.asarray([distances[s] for s in states],dtype=np.uint8).tofile(path/'distances.bin')
    atomic_json(path/'oracle.json',dict(scope=dict(vertex_count=n,edges=edges),statistics=dict(diameter=max(distances.values()))))
    return states,children


class ExactPairTests(unittest.TestCase):
    def test_broad_sources_fresh_goals_and_sampler_resume(self):
        from grounded_llm.core.qmap.sampling import CoverageSampler
        with tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
            path=Path(tmp); small_oracle(path,5,[[0,1],[1,2],[1,3],[3,4]])
            corpus=TokenCorpus(path); stack.callback(corpus.close)
            config=dict(relations=dict(endpoint_holdout_fraction=.05,evaluation_count=4,complete_training_sources=True))
            groups,pairs,tasks,reserved,partition=token_cohorts(corpus,config,np.random.default_rng(7))
            allowed=np.flatnonzero(partition==0)
            np.testing.assert_array_equal(np.sort(pairs[:,0]),allowed)
            fresh=sample_goals(corpus,tasks[:,0],np.random.default_rng(8),reserved,ranking=True,allowed=allowed)
            self.assertFalse(np.intersect1d(closure(corpus,fresh),reserved).size)
            self.assertTrue((partition[fresh]==0).all())
            self.assertFalse(np.array_equal(fresh[:,1],tasks[:,1]))
            local=sample_goals(corpus,tasks[:,0],np.random.default_rng(8),reserved,ranking=True,allowed=allowed,
                               local_fraction=.5,walk_max=4)
            self.assertFalse(np.intersect1d(closure(corpus,local),reserved).size)
            self.assertTrue((partition[local]==0).all())
            near=sample_goals(corpus,np.tile(np.arange(corpus.size),10),np.random.default_rng(4),np.array([],dtype=np.int64),
                              allowed=np.arange(corpus.size),local_fraction=1.,walk_max=2)
            self.assertTrue((corpus.distance(near[:,0],near[:,1])<=2).all())
            rng=np.random.default_rng(9); sampler=CoverageSampler(corpus.distance(pairs[:,0],pairs[:,1]),rng,len(pairs),.5)
            selected=np.concatenate([sampler.draw(20) for _ in range(12)])
            self.assertEqual(len(np.unique(selected)),len(pairs))
            state=sampler.state_dict(); rng_state=rng.bit_generator.state
            expected=sampler.draw(20)
            other_rng=np.random.default_rng(0)
            other=CoverageSampler(corpus.distance(pairs[:,0],pairs[:,1]),other_rng,len(pairs),.5)
            other.load_state_dict(state); other_rng.bit_generator.state=rng_state
            np.testing.assert_array_equal(other.draw(20),expected)
    @unittest.skipUnless(shutil.which('g++'),'C++ full-oracle generator is checked on the Linux runtime')
    def test_cpp_oracle_matches_independent_small_graph(self):
        from grounded_llm.tasks.token_swapping.relation_data import build
        with tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
            root=Path(tmp); expected=root/'expected'; expected.mkdir()
            edges=[[0,1],[1,2],[1,3],[3,4]]
            states,children=small_oracle(expected,5,edges)
            tree=root/'tree.json'; atomic_json(tree,dict(vertex_count=5,edges=edges))
            corpus=build(dict(seed=0,data=dict(tree=str(tree))),root/'actual'); stack.callback(corpus.close)
            np.testing.assert_array_equal(corpus.states,np.asarray(states))
            np.testing.assert_array_equal(corpus.children,children)
            np.testing.assert_array_equal(corpus.distances,np.fromfile(expected/'distances.bin',dtype=np.uint8))
            self.assertEqual(corpus.metadata['verification']['bellman_states'],120)

    def test_directed_complete_sokoban_graph_and_dead_end(self):
        env=Sokoban(['#####','#@$.#','#   #','#####'])
        states=[(p,b) for b in combinations(sorted(env.floor),1) for p in sorted(env.floor) if p not in b]
        index={s:i for i,s in enumerate(states)}
        children=np.full((len(states),4),-1,dtype=np.int32)
        for i,s in enumerate(states):
            env.validate(s)
            for a,t,cost in env.transitions(s): children[i,a]=index[t]; self.assertEqual(cost,1)
        goal=(7,(8,)); oracle=UnitGraphOracle(children); label=oracle.to_goal(index[goal])
        for start in states:
            distance={start:0}; queue=deque([start])
            while queue:
                state=queue.popleft()
                for _,child,_ in env.transitions(state):
                    if child not in distance: distance[child]=distance[state]+1; queue.append(child)
            self.assertEqual(label[index[start]],distance.get(goal,np.inf))
        self.assertEqual(label[index[goal]],0)
        self.assertTrue(np.isinf(label).any())
        start=index[env.start]
        self.assertTrue(np.isfinite(label[start]))
        self.assertTrue(np.isinf(oracle.from_goal(index[goal])[start]))
        self.assertTrue(any(np.isfinite(label[i]) and any(np.isinf(label[child]) for child in row if child>=0)
                            for i,row in enumerate(children)))

    def test_rush_complete_component_and_identity_features(self):
        env=RushHour(''.join(['......','BB....','AA....']+['......']*3))
        states,index,children=enumerate_component(env,env.start,1000)
        target=RushHour(''.join(['......','.BB...','....AA']+['......']*3)).start
        oracle=UnitGraphOracle(children)
        labels=oracle.to_goal(index[target]); reverse=oracle.from_goal(index[target])
        self.assertEqual(labels[index[target]],0); self.assertEqual(labels[0],2)
        np.testing.assert_array_equal(labels,reverse)
        features=env.features([env.start,target])
        self.assertEqual(features.shape,(2,936)); self.assertFalse(np.array_equal(*features))
        for state in states:
            env.validate(state)
            for action,child,cost in env.transitions(state):
                self.assertEqual(env.apply(child,dict(action,displacement=-action['displacement'])),(state,cost))

    def test_relative_oracle_all_pairs_and_labelled_features(self):
        with tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
            path=Path(tmp); states,_=small_oracle(path,4,[[0,1],[1,2],[1,3]])
            corpus=TokenCorpus(path)
            stack.callback(corpus.close)
            np.testing.assert_array_equal(rank(np.asarray(states)),np.arange(24))
            for goal in states:
                direct={goal:0}; queue=deque([goal])
                while queue:
                    s=queue.popleft()
                    for _,child,_ in corpus.env.transitions(s):
                        if child not in direct: direct[child]=direct[s]+1; queue.append(child)
                for i,source in enumerate(states):
                    self.assertEqual(corpus.distance(i,int(rank(np.asarray(goal)))),direct[source])
            self.assertEqual(corpus.features(np.array([0,1])).shape,(2,16))
            np.testing.assert_array_equal(corpus.distance(np.arange(24),np.arange(24)),np.zeros(24))

    def test_teacher_landmark_probe_is_exact_with_complete_bank(self):
        from experiments.new_tasks_qmap.src.relation_diagnose import geometry,encoder_axes
        with tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
            path=Path(tmp); small_oracle(path,4,[[0,1],[1,2],[1,3]])
            corpus=TokenCorpus(path); stack.callback(corpus.close)
            rows=np.array([[a,b] for a in range(24) for b in range(24)])
            data={k+'_validation':v for k,v in materialize(corpus,rows).items()}
            config=dict(seed=0,evaluation=dict(landmarks=128,decision_count=1000),
                        model=dict(dimension=128,metric='linf'),training=dict(unreachable_target=30))
            result=geometry(corpus,data,config,['validation'])['validation']
            self.assertEqual(result['finite_mae'],0)
            self.assertEqual(result['shortest_action_rate'],1)
            model=StateEncoder(128,16,64,'linf',input_dim=16)
            axes=encoder_axes(model,corpus,data,config,['validation'])['validation']
            self.assertEqual(sum(axes['winning_dimension_counts']),24*23)
            self.assertTrue(1<=axes['effective_winning_dimensions']<=128)
            config['evaluation']['landmarks']=2
            self.assertLess(geometry(corpus,data,config,['validation'])['validation']['query_goal_in_landmarks'],len(rows))

    def test_multilayout_corpus_padding_and_undefined_cross_scene_distance(self):
        from grounded_llm.tasks.rush_hour.relation_data import decode,baseline
        with tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
            scenes=[]; goals=[]
            boards=[''.join(['......','BB....','AA....']+['......']*3),
                    ''.join(['CC....','BB....','AA....']+['......']*3)]
            for i,board in enumerate(boards):
                env=RushHour(board); states,index,children=enumerate_component(env,env.start,1000)
                g=next(j for j,s in enumerate(states) if env.terminal(s))
                path=Path(tmp)/str(i)
                save_scene(path,states,children,[g],dict(source=dict(row=i),task='rush_hour'),symmetric=True)
                scene=PhysicalScene(path,env,decode,baseline); scenes.append(scene); goals.append(g)
                for name in ('states','children','distances','reverse_distances'):
                    stack.callback(getattr(scene,name)._mmap.close)
            corpus=PhysicalCorpus(scenes); offset=int(corpus.offsets[1])
            ids=np.array([[0,offset],[1,offset+1]])
            self.assertEqual(corpus.features(ids).shape,(2,2,936))
            self.assertEqual(corpus.successors(ids).shape,(2,2,corpus.max_candidates))
            with self.assertRaises(ValueError): corpus.distance(np.array([0]),np.array([offset]))
            rows=np.array([[0,goals[0]],[offset,offset+goals[1]]])
            batch=materialize(corpus,rows)
            self.assertTrue(batch['optimal'].any(axis=1).all())
            model=StateEncoder(128,16,64,'linf',input_dim=936)
            result=rollout(corpus,QCache(model,corpus,'cpu'),rows,32,'oracle')
            self.assertEqual(result['optimal_rate'],1)

    def test_pair_and_endpoint_holdout_excludes_both_losses(self):
        with tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
            path=Path(tmp); small_oracle(path,5,[[0,1],[1,2],[1,3],[3,4]])
            corpus=TokenCorpus(path)
            stack.callback(corpus.close)
            config=dict(relations=dict(endpoint_holdout_fraction=.05,evaluation_count=4,point_count=40,decision_count=30))
            groups,pairs,tasks,reserved,partition=token_cohorts(corpus,config,np.random.default_rng(7))
            self.assertFalse(np.isin(keys(pairs[:,0],pairs[:,1],corpus.size),reserved).any())
            self.assertFalse(np.intersect1d(closure(corpus,tasks),reserved).size)
            self.assertFalse(np.intersect1d(closure(corpus,groups['validation']),closure(corpus,groups['test'])).size)
            exposed=np.concatenate([pairs.ravel(),corpus.successors(tasks[:,0]).ravel(),tasks[:,1]])
            self.assertTrue((partition[exposed]==0).all())
            self.assertTrue((partition[groups['unseen_both_test']]==2).all())
            batch=materialize(corpus,tasks)
            self.assertTrue(batch['optimal'].any(axis=1).all())
            model=StateEncoder(128,16,32,'linf',input_dim=25)
            cache=QCache(model,corpus,'cpu')
            result=rollout(corpus,cache,tasks,32,'oracle')
            self.assertEqual(result['optimal_rate'],1)

    def test_reader_is_frozen_single_goal_and_cross_attention_ready(self):
        from grounded_llm.tasks.sokoban.qmap import MapReader as SokobanReader
        from grounded_llm.tasks.rush_hour.qmap import MapReader as RushReader
        from grounded_llm.tasks.token_swapping.qmap import MapReader as TokenReader
        board=['##########','#@ $ .   #','# $ .    #','# $ .    #','# $ .    #']+['#        #']*4+['##########']
        envs=[('sokoban',Sokoban(board),SokobanReader,400,{}),
              ('rush_hour',RushHour('............AA......................'),RushReader,936,{}),
              ('token_swapping',TokenSwapping(3,[[0,1],[1,2]]),TokenReader,9,dict(vertex_count=3,edges=[[0,1],[1,2]]))]
        with tempfile.TemporaryDirectory() as tmp:
            for task,env,reader_type,dimension,scope in envs:
                start=env.goal if task=='token_swapping' else env.start
                env.validate(start)
                model=StateEncoder(128,16,32,'linf',input_dim=dimension)
                path=Path(tmp)/(task+'.pt')
                torch.save(dict(model_type='exact_state_pairs',task=task,input_dim=dimension,
                    config=dict(model=dict(dimension=128,hidden=16,cap=32,metric='linf')),
                    model=model.state_dict(),scope=scope,step=0),path)
                reader=reader_type(env,path)
                batch,actions=reader.map_batch(start,start)
                batch.validate(max(1,len(actions)))
                self.assertEqual(int((batch.roles==CURRENT).sum()),1)
                self.assertEqual(int((batch.roles==GOAL).sum()),1)
                self.assertEqual(int((batch.roles==SUCCESSOR).sum()),len(list(env.transitions(start))))
                self.assertTrue(all(not p.requires_grad for p in reader.encoder.parameters()))
                before=reader.encode([start]).clone()
                children=list(env.transitions(start))
                reader.map_batch(start,children[0][1])
                torch.testing.assert_close(before,reader.encode([start]))
                memory=AddressedMapMemoryEncoder(128,32,max(1,len(actions)),16,4)
                addressed=memory(batch)
                self.assertEqual(addressed.values.shape[:2],batch.vectors.shape[:2])

    def test_training_checkpoint_and_completed_resume_smoke(self):
        with tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
            root=Path(tmp); graphs=root/'graphs'; graphs.mkdir()
            small_oracle(graphs,5,[[0,1],[1,2],[1,3],[3,4]])
            corpus=TokenCorpus(graphs); stack.callback(corpus.close)
            cohort_config=dict(relations=dict(endpoint_holdout_fraction=.05,evaluation_count=4,point_count=40,decision_count=30))
            groups,pairs,tasks,reserved,partition=token_cohorts(corpus,cohort_config,np.random.default_rng(7))
            data=dict(pairs_train=pairs,pair_distances_train=corpus.distance(pairs[:,0],pairs[:,1]),reserved_relations=reserved)
            for name,rows in [('train',tasks),*groups.items()]:
                for key,value in materialize(corpus,rows).items(): data[key+'_'+name]=value
            config=dict(task='token_swapping',seed=7,model=dict(dimension=128,hidden=16,cap=32,metric='linf'),
                        training=dict(steps=2,batch_size=8,decision_batch_size=4,learning_rate=.001,
                        rank_weight=1.,rank_temperature=1.,eval_every=1,cpu_threads=1,unreachable_target=30.),
                        evaluation=dict(horizon=32,decision_count=16,rollout_count=4))
            prepared=(corpus,data,dict(states=corpus.size))
            out=root/'run'
            with patch('experiments.new_tasks_qmap.src.relations.prepare',return_value=prepared):
                train(config,out,'small-fixture','cpu')
                before=torch.load(out/'last.pt',weights_only=False)
                train(config,out,'small-fixture','cpu',resume=True)
                after=torch.load(out/'last.pt',weights_only=False)
            self.assertEqual(before['step'],2)
            for key,value in before['model'].items(): torch.testing.assert_close(value,after['model'][key])
            self.assertTrue((after['exposure']>0).any())
            self.assertIn('optimizer',after); self.assertIn('pair_sampler',after)
            summary=json.loads((out/'summary.json').read_text())
            for cohort in summary['metrics']['cohorts'].values():
                self.assertEqual(cohort['planning']['oracle']['optimal_rate'],1)
            warm_config=json.loads(json.dumps(config))
            warm_config['training'].update(initialize_from='../run/last.pt',features_on_device=True,
                record_training_samples=True,online_goals=True,local_goal_fraction=.5,goal_walk_max=4)
            phase=root/'phase2'
            def prepare_fixture(config,path):
                path.mkdir(); np.save(path/'endpoint_partition.npy',partition)
                return prepared
            with patch('experiments.new_tasks_qmap.src.relations.prepare',side_effect=prepare_fixture):
                train(warm_config,phase,'warm-fixture','cpu')
            initialization=json.loads((phase/'initialization.json').read_text())
            self.assertFalse(initialization['optimizer_restored'])
            self.assertFalse(initialization['sampler_restored'])
            initial=json.loads((phase/'initial.json').read_text())
            self.assertEqual(initial['cohorts']['validation']['finite_mae'],summary['metrics']['cohorts']['validation']['finite_mae'])
            for shard in (phase/'samples/attempt_000').glob('*.npz'):
                with np.load(shard) as trace:
                    self.assertTrue((partition[trace['pairs']]==0).all())
                    self.assertFalse(np.intersect1d(closure(corpus,trace['tasks']),reserved).size)
                    np.testing.assert_array_equal(trace['point_distance'],corpus.distance(trace['pairs'][:,0],trace['pairs'][:,1]))


if __name__=='__main__': unittest.main()
