"""Independent graph checks, leakage boundaries and oracle-free reader loading."""
from collections import deque
from contextlib import ExitStack
from itertools import combinations,permutations,product
import json
from pathlib import Path
import tempfile
import shutil
import unittest
from types import SimpleNamespace
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
    def test_vectorized_sokoban_graph_matches_every_physical_transition(self):
        from grounded_llm.tasks.sokoban.relation_data import physical_graph,decode
        env=Sokoban(['######','#@$$ #','# $$ #','#....#','######'])
        packed,children,state_id=physical_graph(env)
        states=[(p,b) for b in combinations(sorted(env.floor),4) for p in sorted(env.floor) if p not in b]
        self.assertEqual(decode(packed),states)
        for i,state in enumerate(states):
            self.assertEqual(state_id(state),i)
            expected={a:state_id(child) for a,child,cost in env.transitions(state) if cost==1}
            self.assertEqual({a:int(t) for a,t in enumerate(children[i]) if t>=0},expected)

    def test_group_sampling_balances_unequal_layout_sizes_and_restores(self):
        from copy import deepcopy
        from grounded_llm.core.qmap.sampling import GroupCoverageSampler
        groups=np.array([0]+[1]*9+[2]*31)
        rng=np.random.default_rng(9); sampler=GroupCoverageSampler(groups,rng)
        selected=np.concatenate([sampler.draw(7) for _ in range(30)])
        np.testing.assert_array_equal(np.bincount(groups[selected]),[70,70,70])
        self.assertEqual(len(np.unique(selected)),len(groups))
        saved=deepcopy(sampler.state_dict()); state=deepcopy(rng.bit_generator.state)
        expected=sampler.draw(47)
        sampler.load_state_dict(saved); rng.bit_generator.state=state
        np.testing.assert_array_equal(sampler.draw(47),expected)

    def test_relation_sampling_equal_layout_then_distance_stratum(self):
        from experiments.new_tasks_qmap.src.relation_data import BoardRelationSampler
        groups=[[np.arange(1),np.arange(1,10)],
                [np.arange(10,1010),np.arange(1010,5010)]]
        rows=BoardRelationSampler._draw(groups,40000,np.random.default_rng(18))
        fractions=np.array([(rows<1).mean(),((rows>=1)&(rows<10)).mean(),
                            ((rows>=10)&(rows<1010)).mean(),(rows>=1010).mean()])
        np.testing.assert_allclose(fractions,.25,atol=.01)
        limited=BoardRelationSampler._draw(groups,40000,np.random.default_rng(18),max_groups=1)
        self.assertTrue((limited<10).all() or (limited>=10).all())

    def test_decision_neighborhood_uses_independent_rng_and_all_legal_children(self):
        from copy import deepcopy
        from experiments.new_tasks_qmap.src.relation_data import decision_neighborhood_points
        point=np.array([[0,9],[1,9],[2,9],[3,9],[4,9],[5,9],[6,9],[7,9]])
        tasks=np.array([[10,9],[11,9],[12,9]])
        children=np.array([[20,-1,21],[22,23,24],[25,26,-1]])
        lookup=np.full(27,-1);lookup[:8]=np.arange(8)
        rng=np.random.default_rng(4);saved=deepcopy(rng.bit_generator.state)
        same,index,role=decision_neighborhood_points(point,np.arange(8),tasks,children,None,0,rng)
        np.testing.assert_array_equal(same,point);self.assertEqual(rng.bit_generator.state,saved)
        coupled,index,role=decision_neighborhood_points(point,np.arange(8),tasks,children,lookup,.5,rng)
        np.testing.assert_array_equal(coupled[:4],point[:4]);self.assertEqual((index==-1).sum(),4)
        self.assertEqual(role.tolist(),[0,0,0,0,1,1,2,2])
        for parent,child in zip(coupled[4:6],coupled[6:]):
            i=int(np.flatnonzero(tasks[:,0]==parent[0])[0]);self.assertIn(child[0],children[i])
            self.assertEqual(parent[1],child[1]);self.assertEqual(parent[1],9)

    def test_sampled_board_sources_keep_caps_goals_and_holdout_closure(self):
        from experiments.new_tasks_qmap.src.relation_data import board_cohorts
        env=Sokoban(['######','#@$. #','#    #','######'])
        states=[(p,b) for b in combinations(sorted(env.floor),1) for p in sorted(env.floor) if p not in b]
        index={s:i for i,s in enumerate(states)}; children=np.full((len(states),4),-1,dtype=np.int32)
        for i,state in enumerate(states):
            for a,child,_ in env.transitions(state): children[i,a]=index[child]
        goals=[index[(p,tuple(env.goals))] for p in sorted(env.floor-env.goals)]
        with tempfile.TemporaryDirectory() as tmp,ExitStack() as stack:
            path=Path(tmp)/'scene'
            save_scene(path,[[p,*b] for p,b in states],children,goals,
                       dict(split='train',training_goals=len(goals)-2,source=dict(puzzle_id='fixture')))
            scene=PhysicalScene(path,env,lambda rows:[(int(row[0]),tuple(map(int,row[1:]))) for row in rows],lambda s,g:np.zeros(len(s)))
            for name in ['states','children','distances','reverse_distances']: stack.callback(getattr(scene,name)._mmap.close)
            corpus=PhysicalCorpus([scene])
            config=dict(seed=17,relations=dict(training_sources_per_scene=12,holdout_per_goal=1,
                        evaluation_per_goal=4,unreachable_eval_per_goal=2,evaluation_training_goals=1))
            groups,pairs,tasks,reserved,_=board_cohorts(corpus,config,np.random.default_rng(18))
            self.assertLessEqual(len(pairs),12); self.assertLessEqual(len(tasks),12)
            self.assertGreater(len(pairs),0); self.assertGreater(len(tasks),0)
            self.assertFalse(np.isin(pairs[:,0],scene.goals[-2:]).any())
            self.assertFalse(np.isin(pairs[:,1],scene.goals[-2:]).any())
            self.assertFalse(np.isin(keys(pairs[:,0],pairs[:,1],corpus.size),reserved).any())
            self.assertEqual(len(np.intersect1d(closure(corpus,tasks),reserved)),0)

    def test_lane_order_encoding_keeps_each_vehicle_and_reader_goal_independence(self):
        from grounded_llm.tasks.rush_hour.qmap import MapReader
        from experiments.new_tasks_qmap.src.relation_diagnose import vehicle_encoding_check
        boards=[''.join(['BBCC..','......','AA....','......','......','......']),
                ''.join(['..B...','..B...','AAB.C.','....C.','......','......'])]
        for board in boards:
            env=RushHour(board); states,_,_=enumerate_component(env,env.start,1000)
            proof=vehicle_encoding_check(SimpleNamespace(scenes=[SimpleNamespace(env=env,states=np.array(states,dtype=np.uint8),goals=np.array([0]))]),31)
            self.assertEqual(proof['physical_states'],len(states))
            self.assertTrue(proof['all_component_vehicle_orders_constant'])
            canonical=env.features(states,encoding='lane_order')
            self.assertEqual(len(np.unique(canonical,axis=0)),len(states))
            self.assertTrue((canonical.reshape(len(states),26,36).sum(axis=2)>0).sum(axis=1).tolist()==[len(env.pieces)]*len(states))
            fixed_channels=[]
            for i,(_,horizontal,_,lane) in enumerate(env.pieces):
                cells=np.array([lane*6+s[i] if horizontal else s[i]*6+lane for s in states])
                channel=canonical.reshape(len(states),26,36)[np.arange(len(states)),:,cells].argmax(axis=1)
                self.assertTrue((channel==channel[0]).all())
                fixed_channels.append(int(channel[0]))
            self.assertEqual(fixed_channels[0],0); self.assertEqual(len(set(fixed_channels)),len(env.pieces))
            rename=str.maketrans({'B':'C','C':'B'})
            other=RushHour(board.translate(rename),feature_encoding='lane_order')
            mapped=[RushHour(''.join(env.render(s)).translate(rename)).start for s in states]
            np.testing.assert_array_equal(canonical,other.features(mapped))
            with tempfile.TemporaryDirectory() as tmp:
                config=dict(model=dict(dimension=128,hidden=16,cap=256,metric='linf'),input_encoding='lane_order')
                model=StateEncoder(input_dim=936,**config['model'])
                path=Path(tmp)/'map.pt'
                torch.save(dict(model_type='exact_state_pairs',task='rush_hour',scope={},input_dim=936,
                    config=config,model=model.state_dict(),step=0),path)
                reader=MapReader(env,path)
                expected=model.encode(torch.as_tensor(canonical[:2]))
                torch.testing.assert_close(reader.encode(states[:2]),expected)
                before=reader.encode([states[0]])
                reader.map_batch(states[0],states[-1])
                torch.testing.assert_close(before,reader.encode([states[0]]))
                self.assertTrue(all(not parameter.requires_grad for parameter in reader.encoder.parameters()))

    def test_input_isometries_preserve_physics_and_complete_state_identity(self):
        from experiments.new_tasks_qmap.src.relation_diagnose import isometry_feature_view
        board=np.full((10,10),'#')
        board[1:4,1:4]=' '; board[1,3]='.'; board[2,2]='$'; board[2,1]='@'
        env=Sokoban([''.join(row) for row in board])
        states,_,_=enumerate_component(env,env.start,1000)
        corpus=SimpleNamespace(features=lambda ids:env.features([states[i] for i in ids]))
        for variant in range(8):
            grid=np.rot90(np.arange(100).reshape(10,10),variant%4)
            transformed=np.rot90(board,variant%4)
            if variant//4: grid=np.flip(grid,-1); transformed=np.flip(transformed,-1)
            cell=np.empty(100,dtype=np.int64); cell[grid.ravel()]=np.arange(100)
            def transform(state): return int(cell[state[0]]),tuple(sorted(map(int,cell[list(state[1])])) )
            target=Sokoban([''.join(row) for row in transformed])
            mapped=[transform(state) for state in states]
            view,_=isometry_feature_view(corpus,'sokoban',variant,31)
            np.testing.assert_array_equal(view.features(np.arange(len(states))),target.features(mapped))
            self.assertEqual(len(set(mapped)),len(states))
            for source,dest in zip(states,mapped):
                self.assertEqual({transform(child) for _,child,cost in env.transitions(source) if cost==1},
                                 {child for _,child,cost in target.transitions(dest) if cost==1})
        env=RushHour(''.join(['..B...','..B...','AAB.C.','....C.','......','......']))
        states,_,_=enumerate_component(env,env.start,1000)
        corpus=SimpleNamespace(features=lambda ids:env.features([states[i] for i in ids]),
            scenes=[SimpleNamespace(env=env)],locate=lambda ids:np.zeros(len(ids),dtype=int))
        for variant in range(8):
            view,mapping=isometry_feature_view(corpus,'rush_hour',variant,31)
            names=str.maketrans({chr(65+i):chr(65+j) for i,j in enumerate(mapping['0'])})
            def transform(state): return RushHour(''.join(env.render(state)).translate(names)).start
            target=RushHour(''.join(env.render(env.start)).translate(names)); mapped=[transform(state) for state in states]
            np.testing.assert_array_equal(view.features(np.arange(len(states))),target.features(mapped))
            self.assertEqual(len(set(mapped)),len(states))
            for source,dest in zip(states,mapped):
                self.assertEqual({transform(child) for _,child,_ in env.transitions(source)},
                                 {child for _,child,_ in target.transitions(dest)})
        with tempfile.TemporaryDirectory() as tmp,ExitStack() as stack:
            path=Path(tmp); small_oracle(path,3,[[0,1],[1,2]])
            corpus=TokenCorpus(path); stack.callback(corpus.close); ids=np.arange(corpus.size)
            for variant in range(8):
                view,mapping=isometry_feature_view(corpus,'token_swapping',variant,31)
                mapped=rank(np.array(mapping['0'])[corpus.states])
                np.testing.assert_array_equal(view.features(ids),corpus.features(mapped))
                np.testing.assert_array_equal(corpus.distance(ids[:,None],ids[None,:]),corpus.distance(mapped[:,None],mapped[None,:]))

    def test_fixed_near_validation_pairs_have_exact_distances_and_held_endpoints(self):
        from experiments.new_tasks_qmap.src.relation_diagnose import near_endpoint_cases
        with tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
            path=Path(tmp); small_oracle(path,5,[[0,1],[1,2],[1,3],[3,4]])
            corpus=TokenCorpus(path); stack.callback(corpus.close)
            partition=np.zeros(corpus.size,dtype=np.int8); order=np.random.default_rng(7).permutation(corpus.size)
            partition[order[:20]]=1; partition[order[20:40]]=2
            cases=near_endpoint_cases(corpus,partition,11,20)
            repeated=near_endpoint_cases(corpus,partition,11,20)
            for name,rows in cases.items():
                np.testing.assert_array_equal(rows,repeated[name]); self.assertEqual(len(np.unique(rows,axis=0)),20)
                self.assertFalse((partition[rows]==2).any()); self.assertTrue((partition[rows]==1).any(axis=1).all())
                distances=corpus.distance(rows[:,0],rows[:,1])
                self.assertEqual({d:int((distances==d).sum()) for d in range(1,6)},{d:4 for d in range(1,6)})

    def test_planning_queries_deduplicate_pairs_before_distance_allocation(self):
        from experiments.new_tasks_qmap.src.relation_evaluate import planning_ids
        with tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
            path=Path(tmp); small_oracle(path,4,[[0,1],[1,2],[1,3]])
            corpus=TokenCorpus(path); stack.callback(corpus.close)
            unique=np.array([[2,0],[2,1],[3,0],[4,0]])
            repeated=np.concatenate([np.repeat(unique[:1],50,axis=0),np.repeat(unique[1:2],20,axis=0),unique[2:]])
            truth=corpus.distance(repeated[:,0],repeated[:,1])
            all_ids=planning_ids(corpus,repeated,truth,100,np.random.default_rng(4))
            np.testing.assert_array_equal(repeated[all_ids],unique)
            for count in (1,3):
                ids=planning_ids(corpus,repeated,truth,count,np.random.default_rng(4))
                expected=planning_ids(corpus,unique,corpus.distance(unique[:,0],unique[:,1]),count,np.random.default_rng(4))
                self.assertEqual(len(np.unique(repeated[ids],axis=0)),count)
                np.testing.assert_array_equal(repeated[ids],unique[expected])

    def test_many_layouts_small_quota_still_tests_long_distances(self):
        from experiments.new_tasks_qmap.src.relation_evaluate import planning_ids
        scene=np.repeat(np.arange(100),149); truth=np.tile(np.arange(1,150),100)
        rows=np.column_stack([scene*150+truth,scene*150])
        corpus=SimpleNamespace(scenes=[None]*100,locate=lambda ids:ids//150)
        old=planning_ids(corpus,rows,truth,1024,np.random.default_rng(23))
        self.assertLessEqual(truth[old].max(),11)
        new=planning_ids(corpus,rows,truth,1024,np.random.default_rng(23),[1,2,5,10,20,40,80,256])
        self.assertEqual(len(np.unique(rows[new],axis=0)),1024)
        self.assertGreater((truth[new]>80).sum(),100)
        self.assertGreater((truth[new]>20).sum(),300)
        self.assertEqual(len(np.unique(scene[new])),100)
        repeat=planning_ids(corpus,rows,truth,1024,np.random.default_rng(23),[1,2,5,10,20,40,80,256])
        np.testing.assert_array_equal(new,repeat)

    def test_correct_candidate_order_does_not_force_unit_distance_descent(self):
        from experiments.new_tasks_qmap.src.relation_evaluate import unit_cost_consistency
        from grounded_llm.core.qmap.model import soft_rank_loss
        parent=np.array([2.,2.]); children=np.array([[1.,3.,np.inf],[1.,1.,3.]])
        valid=np.isfinite(children); optimal=valid & (children==1)
        exact=unit_cost_consistency(parent,children,optimal,valid)
        self.assertEqual(exact['mae'],0); self.assertEqual(exact['teacher_optimal_non_descent_rate'],0)
        broken=unit_cost_consistency(np.array([.2,2.]),children,optimal,valid)
        self.assertEqual(broken['chosen_true_optimal_count'],2)
        self.assertEqual(broken['chosen_true_optimal_but_non_descent_count'],1)
        self.assertEqual(broken['chosen_non_descent_rate'],.5)
        self.assertAlmostEqual(broken['teacher_optimal_drop_mae_to_one'],.6)
        mask=torch.as_tensor(valid); scores=torch.as_tensor(np.where(valid,children,0))
        truth=torch.as_tensor(np.where(valid,children,30))
        torch.testing.assert_close(soft_rank_loss(scores,truth,mask,1.),soft_rank_loss(scores+17,truth,mask,1.))

    def test_spatial_encoder_preserves_raw_board_and_checkpoint_interface(self):
        options=dict(dimension=128,hidden=256,cap=256,metric='directed_max',input_dim=400)
        mlp=StateEncoder(**options)
        spatial=StateEncoder(**options,architecture='spatial_cnn',spatial_shape=[4,10,10],conv_channels=16,spatial_hidden=112)
        self.assertLess(abs(sum(p.numel() for p in mlp.parameters())-sum(p.numel() for p in spatial.parameters()))/sum(p.numel() for p in mlp.parameters()),.02)
        bits=torch.randint(0,2,(3,400)).float(); q=spatial.encode(bits)
        self.assertEqual(tuple(q.shape),(3,128)); self.assertTrue(((q>0)&(q<256)).all())
        torch.testing.assert_close(q[0],spatial.encode(bits[:1])[0])
        torch.testing.assert_close(spatial.distance(q,q),torch.zeros(3))
        with self.assertRaises(ValueError): StateEncoder(**options,architecture='spatial_cnn',spatial_shape=[4,9,10])
        env=Sokoban(['##########','#@ $ .   #','# $ .    #','# $ .    #','# $ .    #']+['#        #']*4+['##########'])
        from grounded_llm.tasks.sokoban.qmap import MapReader
        config=dict(model={k:v for k,v in options.items() if k!='input_dim'})
        config['model'].update(architecture='spatial_cnn',spatial_shape=[4,10,10],conv_channels=16,spatial_hidden=112)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'map.pt'
            torch.save(dict(model_type='exact_state_pairs',task='sokoban',input_dim=400,config=config,
                model=spatial.state_dict(),scope={},step=0),path)
            reader=MapReader(env,path); expected=spatial.encode(torch.as_tensor(env.features([env.start])))
            torch.testing.assert_close(reader.encode([env.start]),expected)
            batch,actions=reader.map_batch(env.start,env.start)
            self.assertEqual(batch.vectors.shape[-1],128); self.assertGreater(len(actions),0)
            self.assertFalse(any(p.requires_grad for p in reader.encoder.parameters()))

    def test_linf_incompatibility_witness_and_axis_orientation(self):
        from experiments.new_tasks_qmap.src.relation_diagnose import pair_incompatibility_gaps
        cube=np.asarray(list(product((0,1),repeat=3)))
        hamming=SimpleNamespace(distance=lambda a,b:np.abs(cube[a]-cube[b]).sum(axis=-1))
        gaps=pair_incompatibility_gaps(hamming,np.array([[0,7],[1,6],[2,5],[3,4]]))
        np.testing.assert_array_equal(gaps[~np.eye(4,dtype=bool)],np.full(12,2))
        # An exact existing3D Linf embedding must never require incompatible
        # tight pairs to share an axis, even when their signs differ.
        q=np.random.default_rng(7).integers(-10,11,size=(12,3))
        corpus=SimpleNamespace(distance=lambda a,b:np.abs(q[a]-q[b]).max(axis=-1))
        rows=np.asarray(list(combinations(range(12),2))); axes=np.abs(q[rows[:,0]]-q[rows[:,1]]).argmax(axis=-1)
        gaps=pair_incompatibility_gaps(corpus,rows)
        for axis in range(3): self.assertTrue((gaps[np.ix_(axes==axis,axes==axis)]<=0).all())

    def test_actual_batch_stream_counts_and_full_legal_boundary_coverage(self):
        from experiments.new_tasks_qmap.src.relation_diagnose import stream_statistics,batch_gradient_probe
        env=Sokoban(['#####','#@$.#','#   #','#####'])
        states=[(p,b) for b in combinations(sorted(env.floor),1) for p in sorted(env.floor) if p not in b]
        index={s:i for i,s in enumerate(states)}
        children=np.full((len(states),4),-1,dtype=np.int32)
        for i,s in enumerate(states):
            for a,t,_ in env.transitions(s): children[i,a]=index[t]
        goal=index[(7,(8,))]; truth=UnitGraphOracle(children).to_goal(goal)
        scene=SimpleNamespace(states=np.array([[p,*boxes] for p,boxes in states]),goals=np.array([goal]),children=children,
            distances=truth[None],metadata=dict(split='train',training_goals=1),
            features=lambda ids:np.eye(len(states),dtype=np.float32)[ids],
            distance=lambda s,g:truth[s])
        corpus=PhysicalCorpus([scene]); parents=np.flatnonzero(np.isfinite(truth)&(truth>0))
        pairs=np.column_stack([np.arange(len(states)),np.full(len(states),goal)])
        from experiments.new_tasks_qmap.src.relation_diagnose import geometry
        probe=geometry(corpus,{k+'_validation':v for k,v in materialize(corpus,pairs).items()},
            dict(task='sokoban',seed=0,model=dict(dimension=128,metric='directed_max'),
                 training=dict(unreachable_target=30),evaluation=dict(decision_count=1000)),['validation'])['validation']
        self.assertEqual(probe['finite_mae'],0); self.assertEqual(probe['shortest_action_rate'],1)
        self.assertEqual(probe['dimensions_per_scene'],{'0':100})
        self.assertIn('Common100-cell',probe['coordinate_assignment'])
        rank=materialize(corpus,np.column_stack([parents,np.full(len(parents),goal)]))
        from experiments.new_tasks_qmap.src.relation_data import BoardRelationSampler
        from grounded_llm.core.qmap.sampling import CoverageSampler
        data=dict(pairs_train=pairs,pair_distances_train=truth,tasks_train=rank['tasks'],reserved_relations=np.array([],dtype=np.int64))
        options=dict(finite_point_fraction=.75,boundary_rank_fraction=.5,distance_bin_upper_bounds=[0,1,2,5,10,20,40,80,128])
        sampler=BoardRelationSampler(corpus,data,options); rng=np.random.default_rng(7)
        self.assertTrue((corpus.distance(sampler.points[:,0],sampler.points[:,1])>0).all())
        source_sampler=CoverageSampler(truth,rng,len(pairs),1.)
        for _ in range(10):
            p,r,point,task=sampler.draw(source_sampler,64,32,rng)
            self.assertGreaterEqual(np.isfinite(corpus.distance(point[:,0],point[:,1])).mean(),.75)
            decision=materialize(corpus,task); legal=decision['children']>=0
            boundary=(legal & np.isfinite(decision['child_distances'])).any(axis=1) & (legal & ~np.isfinite(decision['child_distances'])).any(axis=1)
            self.assertEqual(boundary.sum(),16)
            np.testing.assert_array_equal(pairs[p,0],point[:,0]); np.testing.assert_array_equal(rank['tasks'][r,0],task[:,0])
        saved=source_sampler.state_dict(); rng_state=rng.bit_generator.state
        expected=sampler.draw(source_sampler,64,32,rng)
        source_sampler.load_state_dict(saved); rng.bit_generator.state=rng_state
        actual=sampler.draw(source_sampler,64,32,rng)
        for a,b in zip(expected,actual): np.testing.assert_array_equal(a,b)
        row=dict(pairs=pairs,point_distance=truth,tasks=rank['tasks'],children=rank['children'],
            child_distance=rank['child_distances'],parent_distance=rank['distances'],optimal=rank['optimal'])
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp); shard=path/'samples/attempt_000'; shard.mkdir(parents=True)
            packed={key:np.concatenate([value,value]) for key,value in row.items()}
            packed.update(point_step=np.repeat([1,2],len(pairs)),rank_step=np.repeat([1,2],len(parents)))
            np.savez(shard/'00000.npz',**packed)
            result=stream_statistics(corpus,data,path,path/'audit.json')
            self.assertEqual(result['actual_batches'],2)
            self.assertEqual(result['point_finite_fraction'],np.isfinite(truth).mean())
            self.assertEqual(result['ranking_relation_coverage'],1)
            self.assertGreater(result['totals']['boundary_decisions'],0)
            self.assertEqual(result['totals']['rows_without_optimal'],0)
            self.assertEqual(result['totals']['initial_finite_to_actual_unreachable'],0)
            self.assertEqual(result['totals']['initial_near_to_actual_far'],0)
            self.assertEqual(result['totals']['point_initial_row_finite'],result['totals']['point_finite'])
            packed['children']=packed['children'].copy(); packed['children'][0,:]=-1
            np.savez(shard/'00000.npz',**packed)
            with self.assertRaisesRegex(ValueError,'Missing legal'): stream_statistics(corpus,data,path,path/'bad.json')
        from experiments.new_tasks_qmap.src.relation_data import decision_neighborhood_points
        ranking=materialize(corpus,rank['tasks'][rank['tasks'][:,0]>=6])
        self.assertGreater(len(ranking['tasks']),0)
        base=np.tile(pairs[:6],(2,1))[:8]
        lookup=np.full(corpus.size,-1,dtype=np.int64);lookup[:6]=np.arange(6)
        point,indices,origin=decision_neighborhood_points(base,lookup[base[:,0]],ranking['tasks'],ranking['children'],lookup,.5,np.random.default_rng(11))
        self.assertTrue((indices[origin==1]==-1).all())
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp);shard=path/'samples/attempt_000';shard.mkdir(parents=True)
            packed=dict(pairs=point,point_distance=corpus.distance(point[:,0],point[:,1]),point_origin=origin,
                tasks=ranking['tasks'],children=ranking['children'],child_distance=ranking['child_distances'],
                parent_distance=ranking['distances'],optimal=ranking['optimal'],
                point_step=np.ones(len(point),dtype=np.int32),rank_step=np.ones(len(ranking['tasks']),dtype=np.int32))
            np.savez(shard/'00000.npz',**packed)
            sampled=dict(pairs_train=pairs[:6],pair_distances_train=truth[:6],tasks_train=ranking['tasks'],reserved_relations=np.array([],dtype=np.int64))
            result=stream_statistics(corpus,sampled,path,path/'coupled.json')
            self.assertEqual(result['totals']['point_neighborhood_parent'],2)
            self.assertEqual(result['totals']['point_neighborhood_child'],2)
            self.assertGreaterEqual(result['totals']['point_initial_unavailable'],2)
            self.assertTrue(result['all_actual_point_and_child_labels_equal_physical_oracle'])
            packed['pairs'][np.flatnonzero(origin==1)[0],0]=0
            packed['point_distance']=corpus.distance(packed['pairs'][:,0],packed['pairs'][:,1])
            np.savez(shard/'00000.npz',**packed)
            with self.assertRaisesRegex(ValueError,'same-batch same-goal'):
                stream_statistics(corpus,sampled,path,path/'bad_origin.json')
        training_data=dict(pairs_train=pairs[::2],pair_distances_train=truth[::2],
                           reserved_relations=np.array([],dtype=np.int64))
        for name in ['train','validation']:
            for key,value in rank.items(): training_data[key+'_'+name]=value
        training_config=dict(task='sokoban',seed=19,model=dict(dimension=128,hidden=16,cap=32,metric='directed_max'),
            training=dict(steps=20,batch_size=8,decision_batch_size=4,learning_rate=.001,rank_weight=1.,
                rank_temperature=1.,eval_every=20,cpu_threads=1,unreachable_target=30.,online_goals=True,
                relation_sampling=options,decision_neighborhood_point_fraction=.5,record_training_samples=True),
            evaluation=dict(horizon=32,decision_count=16,rollout_count=4))
        reverse=UnitGraphOracle(children).from_goal(goal)
        scene.reverse=lambda s,g:reverse[s]
        scene.heuristic=lambda s,g:np.zeros(len(s))
        scene.decode=lambda rows:[(int(row[0]),tuple(map(int,row[1:]))) for row in rows]
        scene.env=env;scene.metadata['source']=dict(puzzle_id='small-fixture')
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'run'
            with patch('experiments.new_tasks_qmap.src.relations.prepare',return_value=(corpus,training_data,dict(states=corpus.size))):
                train(training_config,path,'neighborhood-fixture','cpu')
                saved=torch.load(path/'last.pt',weights_only=False)
                counted=total=0
                for file in path.glob('samples/attempt_*/*.npz'):
                    with np.load(file) as archive:
                        sources=archive['pairs'][:,0];total+=len(sources)
                        counted+=int(np.isin(sources,training_data['pairs_train'][:,0]).sum())
                self.assertEqual(int(saved['pair_draws'].sum()),counted)
                self.assertLess(counted,total);self.assertEqual(total,160)
                train(training_config,path,'neighborhood-fixture','cpu',resume=True)
                self.assertEqual(torch.load(path/'last.pt',weights_only=False)['neighborhood_rng'],saved['neighborhood_rng'])
        # Replay includes decisions whose sources are absent from both original
        # banks. Their counters must remain separate, and every legal branch
        # must appear in point supervision at that same batch goal.
        from copy import deepcopy
        replay_data=deepcopy(training_data)
        legal=rank['children']>=0
        boundary=(legal&np.isfinite(rank['child_distances'])).any(axis=1)&(legal&~np.isfinite(rank['child_distances'])).any(axis=1)
        keep=np.array([np.flatnonzero(boundary)[0],np.flatnonzero(~boundary)[0]])
        for key,value in rank.items(): replay_data[key+'_train']=value[keep]
        replay_bank=dict(**rank,reason=np.full(len(rank['tasks']),17,dtype=np.uint8))
        replay_config=deepcopy(training_config)
        replay_config['training'].update(batch_size=16,decision_neighborhood_point_fraction=0.,
            failure_replay=dict(bank='../replay.npz',ranking_fraction=.5,critical_fraction=.5,
                distance_bin_upper_bounds=options['distance_bin_upper_bounds'],layouts_per_batch=1))
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'run';np.savez(Path(tmp)/'replay.npz',**replay_bank)
            with patch('experiments.new_tasks_qmap.src.relations.prepare',return_value=(corpus,replay_data,dict(states=corpus.size))):
                train(replay_config,path,'replay-fixture','cpu')
                saved=torch.load(path/'last.pt',weights_only=False)
                self.assertEqual(int(saved['replay_draws'].sum()),40)
                counted=0
                for file in path.glob('samples/attempt_*/*.npz'):
                    with np.load(file) as archive:
                        counted+=int(np.isin(archive['tasks'][:,0],replay_data['tasks_train'][:,0]).sum())
                        for step in np.unique(archive['rank_step']):
                            rm=(archive['rank_step']==step)&(archive['rank_origin']>0)
                            contexts=archive['tasks'][rm];kids=archive['children'][rm];valid=kids>=0
                            parent_keys=contexts[:,0]*corpus.size+contexts[:,1]
                            child_keys=kids[valid]*corpus.size+np.broadcast_to(contexts[:,1,None],kids.shape)[valid]
                            pm=archive['point_step']==step;points=archive['pairs'][pm];origin=archive['point_origin'][pm]
                            keys_=points[:,0]*corpus.size+points[:,1]
                            self.assertTrue(np.isin(parent_keys,keys_[origin==1]).all())
                            self.assertTrue(np.isin(child_keys,keys_[origin==2]).all())
                self.assertEqual(int(saved['rank_draws'].sum()),counted);self.assertLess(counted,80)
                audit=stream_statistics(corpus,replay_data,path,path/'audit.json')
                self.assertEqual(audit['totals']['rank_replay_critical'],20)
                self.assertEqual(audit['totals']['rank_replay_continuation'],20)
                self.assertTrue(audit['all_actual_point_and_child_labels_equal_physical_oracle'])
                train(replay_config,path,'replay-fixture','cpu',resume=True)
                restored=torch.load(path/'last.pt',weights_only=False)
                self.assertEqual(restored['replay_rng'],saved['replay_rng'])
                np.testing.assert_array_equal(restored['replay_draws'],saved['replay_draws'])
                bad=dict(replay_bank);bad['children']=bad['children'].copy();bad['children'][0,:]=-1
                np.savez(path/'failure_replay_bank.npz',**bad)
                with self.assertRaisesRegex(ValueError,'Replay labels or legal candidates changed'):
                    train(replay_config,path,'replay-fixture','cpu',resume=True)
        model=StateEncoder(128,16,32,'directed_max',input_dim=len(states))
        config=dict(training=dict(unreachable_target=30.,rank_temperature=1.,rank_weight=1.))
        result=batch_gradient_probe(model,corpus,row,config)
        self.assertGreater(result['components']['finite_point']['gradient_l2'],0)
        self.assertGreater(result['components']['unreachable_point']['gradient_l2'],0)
        self.assertAlmostEqual(sum(v['projection_on_total'] for v in result['components'].values()),1,places=5)
        self.assertLessEqual(result['point_huber_abs_derivative_mean']['unreachable'],1)
        qa=model.encode(torch.as_tensor(corpus.features(pairs[:,0])))
        qg=model.encode(torch.as_tensor(corpus.features(pairs[:,1])))
        loss=torch.nn.functional.smooth_l1_loss(model.distance(qa,qg),torch.as_tensor(np.where(np.isfinite(truth),truth,30.)))
        self.assertAlmostEqual(float(loss.detach()),sum(result['components'][k]['weighted_loss'] for k in ('finite_point','unreachable_point')),places=5)

    def test_failure_mining_keeps_cycle_members_branches_and_training_boundaries(self):
        from experiments.new_tasks_qmap.src.relation_data import training_decision_mask
        from experiments.new_tasks_qmap.src.relation_diagnose import training_failure_bank
        children=np.array([[1,2,-1],[0,2,-1],[3,-1,-1],[-1,-1,-1]])
        truth=UnitGraphOracle(children).to_goal(3)
        scene=SimpleNamespace(states=np.arange(4)[:,None],children=children,goals=np.array([3]),
            metadata=dict(split='train',training_goals=1),features=lambda ids:np.eye(4)[ids],distance=lambda s,g:truth[s])
        corpus=PhysicalCorpus([scene]); rows=np.array([[0,3],[1,3]])
        packed=materialize(corpus,rows)
        edges=dict(source=rows[:,0],goal=rows[:,1],children=packed['children'],choice=np.array([0,0]),
            parent_distance=packed['distances'],child_distances=packed['child_distances'],
            parent_prediction=np.ones(2),child_predictions=np.ones((2,3)),
            query_index=np.zeros(2,dtype=int),step=np.array([0,1]),next_repeated=np.array([False,True]),next_is_goal=np.zeros(2,dtype=bool))
        data=dict(reserved_relations=np.empty(0,dtype=np.int64))
        result,bank=training_failure_bank(corpus,data,edges,continuation_seeds_per_layout=1)
        self.assertEqual(result['cycle_member_decisions'],2)
        self.assertEqual(result['nonoptimal_decisions'],2)
        self.assertEqual(result['failed_queries'],1)
        np.testing.assert_array_equal(bank['tasks'],[[0,3],[1,3],[2,3]])
        np.testing.assert_array_equal(bank['children'],children[:3])
        self.assertTrue((bank['reason'][:2]&8).all())
        self.assertTrue(bank['reason'][2]&16)
        np.testing.assert_array_equal(bank['optimal'],[[False,True,False],[False,True,False],[True,False,False]])
        # A nonoptimal step in a trajectory that eventually succeeds is not
        # automatically a failure seed. Its detour must not be mislabeled.
        successful_rows=np.array([[0,3],[1,3],[2,3]])
        successful_truth=materialize(corpus,successful_rows)
        success=dict(source=successful_rows[:,0],goal=successful_rows[:,1],children=successful_truth['children'],
            choice=np.array([0,1,0]),parent_distance=successful_truth['distances'],child_distances=successful_truth['child_distances'],
            parent_prediction=np.ones(3),child_predictions=np.ones((3,3)),query_index=np.zeros(3,dtype=int),
            step=np.arange(3),next_repeated=np.zeros(3,dtype=bool),next_is_goal=np.array([False,False,True]))
        result,empty=training_failure_bank(corpus,data,success)
        self.assertEqual(result['failed_queries'],0)
        self.assertEqual(result['unique_bank_decisions'],0)
        protected=dict(reserved_relations=keys(np.array([2]),np.array([3]),corpus.size))
        self.assertFalse(training_decision_mask(corpus,protected,bank['tasks']).any())
        result,empty=training_failure_bank(corpus,protected,edges)
        self.assertEqual(result['unique_bank_decisions'],0)
        self.assertEqual(empty['tasks'].shape,(0,2))
        scene.metadata['split']='test'
        self.assertFalse(training_decision_mask(corpus,data,rows).any())
        scene.metadata['split']='train';scene.goals=np.array([3,2])
        self.assertFalse(training_decision_mask(corpus,data,rows).any())
        self.assertFalse(training_decision_mask(corpus,data,np.array([[0,2]])).any())
        scene.goals=np.array([3])
        bad={**edges,'children':edges['children'].copy()};bad['children'][0,1]=-1
        with self.assertRaisesRegex(ValueError,'legal candidates'): training_failure_bank(corpus,data,bad)
        bad={**edges,'parent_distance':np.array([99,99])}
        with self.assertRaisesRegex(ValueError,'labels differ'): training_failure_bank(corpus,data,bad)

    def test_trajectory_progress_checks_do_not_change_greedy_selection(self):
        with tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
            path=Path(tmp); small_oracle(path,3,[[0,1],[1,2]])
            corpus=TokenCorpus(path); stack.callback(corpus.close)
            rows=np.array([[i,0] for i in range(1,corpus.size)])
            teacher=SimpleNamespace(scores=lambda s,g:corpus.distance(s,g))
            result=rollout(corpus,teacher,rows,32,'q')
            records=[]
            recorded=rollout(corpus,teacher,rows,32,'q',edge_records=records)
            self.assertEqual(result,recorded)
            from experiments.new_tasks_qmap.src.relation_diagnose import policy_boundaries
            boundaries,edges=policy_boundaries(records)
            self.assertEqual(boundaries['first_non_descent']['executed_edges'],0)
            self.assertEqual(boundaries['cycle_closing']['executed_edges'],0)
            self.assertEqual(boundaries['all_executed']['true_optimal_action_rate'],1)
            self.assertEqual(len(edges['source']),result['predicted_progress']['counts']['executed_edges'])
            self.assertTrue(np.array_equal(corpus.successors(edges['source']),edges['children']))
            self.assertEqual(result['optimal_rate'],1)
            self.assertEqual(result['predicted_progress']['predicted_strict_descent_edge_rate'],1)
            self.assertEqual(result['predicted_progress']['strict_descent_trajectory_count'],len(rows))
            layers=result['predicted_progress']['executed_edges_by_true_distance']
            self.assertEqual(sum(v['executed_edges'] for v in layers.values()),result['predicted_progress']['counts']['executed_edges'])
            self.assertTrue(all(v['true_optimal_edges']==v['executed_edges'] for v in layers.values()))
            flat=SimpleNamespace(scores=lambda s,g:np.zeros(np.broadcast_arrays(s,g)[0].shape,dtype=np.float32))
            records=[]; result=rollout(corpus,flat,rows,32,'q',edge_records=records)
            boundaries,edges=policy_boundaries(records)
            self.assertEqual(boundaries['first_non_descent']['executed_edges'],len(rows))
            self.assertEqual(boundaries['cycle_closing']['executed_edges'],result['cycles'])
            self.assertEqual(boundaries['all_non_descent']['executed_edges'],len(edges['source']))
            self.assertGreater(result['cycles'],0)
            self.assertEqual(result['predicted_progress']['predicted_strict_descent_edge_rate'],0)
            self.assertEqual(result['predicted_progress']['counts']['predicted_plateau'],result['predicted_progress']['counts']['executed_edges'])

    def test_output_scale_and_interval_diagnostics_keep_physical_units(self):
        from experiments.new_tasks_qmap.src.relation_evaluate import coordinate_statistics
        outputs=[]; slopes=[]
        for activation in ('sigmoid','softsign','linear'):
            model=StateEncoder(1,2,256,'directed_max',input_dim=1,output_activation=activation)
            model.network=torch.nn.Identity()
            raw=torch.tensor([[0.]],requires_grad=True); q=model.encode(raw)
            outputs.append(float(q.detach())); slopes.append(float(torch.autograd.grad(q,raw)[0]))
            high=model.encode(torch.tensor([[1.]])); low=model.encode(torch.tensor([[0.]]))
            self.assertEqual(float(model.distance(high,low)),float((high-low).clamp_min(0)))
        self.assertEqual(outputs,[128.,128.,0.]); self.assertEqual(slopes,[64.,64.,256.])
        source=np.array([[10.,0.],[0.,0.]],dtype=np.float32); goal=np.zeros_like(source)
        values=coordinate_statistics(source,goal,np.array([2.,np.inf]),np.array([3.,8.]),10,'linear')
        self.assertEqual(values['all']['finite_mae'],1)
        self.assertIsNone(values['all']['boundary_fraction'])
        self.assertEqual(values['all']['low_output_derivative_fraction'],0)
        self.assertEqual(values['by_true_distance']['proven_unreachable']['count'],1)
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
                options=dict(dimension=128,hidden=16,cap=32,metric='linf')
                if task=='token_swapping': options['output_activation']='linear'
                if task=='token_swapping': options.update(metric='max_l1',group_size=8)
                if task=='rush_hour': options.update(hidden=256,architecture='spatial_cnn',spatial_shape=[26,6,6],conv_channels=32,spatial_hidden=244)
                model=StateEncoder(input_dim=dimension,**options)
                path=Path(tmp)/(task+'.pt')
                torch.save(dict(model_type='exact_state_pairs',task=task,input_dim=dimension,
                    config=dict(model=options),
                    model=model.state_dict(),scope=scope,step=0),path)
                reader=reader_type(env,path)
                torch.testing.assert_close(reader.encode([start]),model.encode(torch.as_tensor(env.features([start]))))
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
                        rank_weight=1.,rank_temperature=1.,eval_every=1,cpu_threads=1,unreachable_target=30.,features_on_device='packed_bits'),
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
