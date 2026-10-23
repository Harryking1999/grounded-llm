"""Verify exact intermediate labels, protected branches and fixed pilot budgets."""
from copy import deepcopy
from types import SimpleNamespace
import unittest
import json
import tempfile
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch

from experiments.new_tasks_qmap.src.relation_data import keys, materialize
from experiments.new_tasks_qmap.src.solution_structure import (
    IntermediateGoals, SolutionSampler, context_points, exact_path, permitted, path_metrics)
from grounded_llm.core.qmap.exact_graph import UnitGraphOracle
from grounded_llm.core.qmap.physical_data import PhysicalCorpus


def fixture():
    # 0 -> 1 -> 2 -> 3; 1 also has a shortcut to 3 and an irreversible trap.
    children = np.array([[1, -1, -1], [2, 3, 4], [3, -1, -1], [-1, -1, -1], [4, -1, -1]])
    oracle = UnitGraphOracle(children); truth = oracle.to_goal(3)
    scene = SimpleNamespace(states=np.arange(5)[:, None], children=children, goals=np.array([3]),
        goal_columns={3: 0}, metadata=dict(split='train', training_goals=1,source=dict(puzzle_id='fixture')),
        features=lambda ids:np.eye(5, dtype=np.float32)[ids],
        heuristic=lambda s,g:np.zeros(len(s)),decode=lambda rows:rows.tolist(),env=SimpleNamespace(actions=['a','b','c']),
        distance=lambda s,g:truth[s], reverse=lambda s,g:oracle.from_goal(3)[s])
    return IntermediateGoals(PhysicalCorpus([scene]))


class SolutionStructureTests(unittest.TestCase):
    def test_feedback_is_graded_per_training_target_with_background_mass(self):
        from experiments.new_tasks_qmap.src.solution_graph_sampling import training_graph_feedback
        from grounded_llm.core.qmap.model import StateEncoder
        corpus=fixture();tasks=np.array([[0,2],[0,3],[1,2],[1,3],[2,3]])
        exact=materialize(corpus,tasks);data={k+'_train':v for k,v in exact.items()}
        torch.manual_seed(4);model=StateEncoder(input_dim=5,dimension=128,hidden=8,cap=256,metric='directed_max');model.eval()
        weights,description,edges=training_graph_feedback(corpus,data,model,np.random.default_rng(5),1)
        self.assertTrue(((weights>=1)&(weights<=4)).all())
        for g in [2,3]:self.assertEqual(len(np.unique(weights[tasks[:,1]==g])),1)
        self.assertTrue(edges)
        for row in description['queries']:self.assertIn(tuple(row),set(map(tuple,tasks)))

    def test_connected_graph_sampler_keeps_all_candidates_and_restores_rng(self):
        from experiments.new_tasks_qmap.src.solution_graph_sampling import ConnectedGraphSampler
        corpus=fixture();tasks=np.array([[0,2],[0,3],[1,2],[1,3],[2,3]])
        exact=materialize(corpus,tasks);pairs=np.unique(context_points(tasks,exact['children']),axis=0)
        data=dict(pairs_train=pairs,**{k+'_train':v for k,v in exact.items()})
        rng=np.random.default_rng(22);sampler=ConnectedGraphSampler(corpus,data,dict(graphs_per_batch=2,neighborhood_size=4),rng)
        state=deepcopy(rng.bit_generator.state);p,r=sampler.draw(64,8)
        self.assertEqual(len(p),64);self.assertEqual(len(r),8)
        for row in context_points(tasks[r],exact['children'][r]):self.assertIn(tuple(row),set(map(tuple,pairs[p])))
        rng.bit_generator.state=state;a,b=sampler.draw(64,8)
        np.testing.assert_array_equal(a,p);np.testing.assert_array_equal(b,r)

    def test_global_order_diagnostic_distinguishes_near_and_far_errors(self):
        from experiments.new_tasks_qmap.src.solution_global_order import global_order_metrics,prepare_order_bank
        corpus=fixture();rows=np.array([[0,3],[1,3],[2,3]])
        truth=np.array([2,1,1]);pred=np.array([.5,1,0])
        result=global_order_metrics(corpus,rows,truth,pred)['aggregate']
        self.assertEqual(result['all']['comparable'],2)
        self.assertEqual(result['all']['correct'],1)
        self.assertEqual(result['gap_1']['accuracy'],.5)
        self.assertIsNone(result['gap_2']['accuracy'])
        data=dict(tasks_train=rows,tasks_new_goal_validation=rows,tasks_new_layout_goal_validation=rows)
        with self.assertRaisesRegex(ValueError,'target is trained'):prepare_order_bank(corpus,data,32,1)

    def test_full_basin_adds_recoverable_continuations_and_keeps_traps(self):
        from experiments.new_tasks_qmap.src.solution_basins import prepare_basins
        corpus=fixture();tasks=np.array([[0,3],[1,3]])
        exact=materialize(corpus,tasks)
        with tempfile.TemporaryDirectory() as tmp:
            source=Path(tmp)/'source';source.mkdir()
            (source/'scenes.json').write_text('{}')
            np.savez(source/'relations.npz',reserved_relations=np.array([],dtype=np.int64),
                **{k+'_train':v for k,v in exact.items()})
            cfg=dict(model=dict(dimension=128,hidden=16,cap=256,metric='directed_max'),
                training=dict(unreachable_target=240),data=dict(basin_source=str(source),condition='B',structure_seed=9))
            out=Path(tmp)/'out';out.mkdir()
            with patch('experiments.new_tasks_qmap.src.solution_basins.load_corpus',return_value=corpus):
                _,data,summary=prepare_basins(cfg,out)
            self.assertEqual(set(data['tasks_train'][:,0]),{0,1,2})
            ids=np.flatnonzero((data['pairs_train']==[4,3]).all(axis=1))
            self.assertEqual(len(ids),1);self.assertTrue(np.isinf(data['pair_distances_train'][ids[0]]))
            self.assertEqual(summary['coverage'][0]['finite_edges'],summary['coverage'][0]['finite_edges_with_next_context'])

    def test_branch_closure_covers_alternatives_and_rejects_truncation(self):
        from experiments.new_tasks_qmap.src.solution_branches import continuation_closure
        corpus=fixture();scene=corpus.scenes[0]
        scene.children=np.array([[1,2,-1],[3,-1,-1],[3,-1,-1],[-1,-1,-1],[4,-1,-1]])
        truth=UnitGraphOracle(scene.children).to_goal(3)
        scene.distance=lambda s,g:truth[s]
        rows=np.array([[0,3]]);empty=np.array([],dtype=np.int64)
        closed,reason=continuation_closure(corpus,empty,rows,10)
        self.assertIsNone(reason);self.assertEqual(set(closed[:,0]),{0,1,2})
        self.assertEqual(continuation_closure(corpus,empty,rows,2)[1],'size')
        protected=keys(np.array([2]),np.array([3]),corpus.size)
        self.assertEqual(continuation_closure(corpus,protected,rows,10)[1],'protected')

    def test_intermediate_labels_use_graph_shortcuts_and_direction(self):
        corpus = fixture()
        np.testing.assert_array_equal(exact_path(corpus, 0, 3, 32), [0, 1, 3])
        self.assertEqual(corpus.distance(np.array([0]), np.array([2]))[0], 2)
        self.assertTrue(np.isinf(corpus.distance(np.array([2]), np.array([0]))[0]))
        self.assertEqual(corpus.distance(np.array([1]), np.array([3]))[0], 1)
        self.assertTrue(np.isinf(corpus.distance(np.array([4]), np.array([2]))[0]))
        with self.assertRaisesRegex(ValueError, 'horizon'): exact_path(corpus, 0, 3, 1)

    def test_candidate_holdout_rejects_whole_context_without_pruning(self):
        corpus = fixture(); rows = np.array([[1, 3]])
        reserved = keys(np.array([2]), np.array([3]), corpus.size)
        self.assertFalse(permitted(corpus, reserved, rows)[0])
        np.testing.assert_array_equal(corpus.successors(np.array([1]))[0], [2, 3, 4])
        self.assertTrue(permitted(corpus, np.array([], dtype=np.int64), rows)[0])
        corpus.scenes[0].goals = np.array([3, 2])
        self.assertFalse(permitted(corpus, np.array([], dtype=np.int64), rows)[0])

    def test_fixed_budgets_all_children_and_rng_restore(self):
        corpus = fixture(); tasks = np.array([[0, 2], [0, 3], [1, 2], [1, 3]])
        exact = materialize(corpus, tasks)
        pairs = np.unique(context_points(tasks, exact['children']), axis=0)
        data = dict(pairs_train=pairs, terminal_ids=np.array([1, 3]), auxiliary_ids=np.array([0, 2]),
            terminal_groups=np.array([[1, 3]]), auxiliary_groups=np.array([[0, 2]]))
        data.update({k+'_train':v for k,v in exact.items()})
        for mode in ['flat_terminal', 'path_terminal', 'path_multiscale', 'balanced_branches']:
            rng = np.random.default_rng(4); sampler = SolutionSampler(corpus, data, dict(mode=mode), rng)
            state = deepcopy(rng.bit_generator.state); p,r = sampler.draw(64, 8)
            self.assertEqual(len(p), 64); self.assertEqual(len(r), 8)
            if mode != 'flat_terminal':
                required = context_points(tasks[r], exact['children'][r])
                for row in required: self.assertIn(tuple(row), set(map(tuple, pairs[p])))
            if mode == 'path_multiscale': self.assertEqual(np.isin(r, [0,2]).sum(), 4)
            rng.bit_generator.state = state
            a,b = sampler.draw(64,8);np.testing.assert_array_equal(a,p);np.testing.assert_array_equal(b,r)
        data['hard_context_ids']=np.array([2])
        sampler=SolutionSampler(corpus,data,dict(mode='balanced_branches',hard_fraction=.5),np.random.default_rng(8))
        p,r=sampler.draw(64,8)
        np.testing.assert_array_equal(r[:4],np.full(4,2))
        for row in context_points(tasks[r],exact['children'][r]):
            self.assertIn(tuple(row),set(map(tuple,pairs[p])))

    def test_structure_metric_detects_local_and_long_span_failure(self):
        corpus = fixture(); path = np.array([[0, 1, 3, -1]])
        class Cache:
            def __init__(self, values): self.values=np.array(values)
            def scores(self, source, goal): return self.values[np.asarray(source)]
        exact = path_metrics(Cache([2,1,1,0,240]), corpus, path)['all']
        self.assertEqual(exact['strict_descent'], 1)
        self.assertEqual(exact['unit_drop_mae'], 0)
        bad = path_metrics(Cache([.5,1,1,0,240]), corpus, path)['all']
        self.assertEqual(bad['strict_descent'], 0)
        self.assertLess(bad['all_span_order_rate'], 1)

    def test_existing_trainer_uses_structure_sampler_and_last_checkpoint(self):
        from experiments.new_tasks_qmap.src.relations import train
        corpus=fixture();tasks=np.array([[0,2],[0,3],[1,2],[1,3]])
        exact=materialize(corpus,tasks);pairs=np.unique(context_points(tasks,exact['children']),axis=0)
        data=dict(pairs_train=pairs,pair_distances_train=corpus.distance(pairs[:,0],pairs[:,1]),
            terminal_ids=np.array([1,3]),auxiliary_ids=np.array([0,2]),reserved_relations=np.array([],dtype=np.int64),
            terminal_groups=np.array([[1,3]]),auxiliary_groups=np.array([[0,2]]),
            solution_paths_validation=np.array([[0,1,3,-1]]))
        data.update({k+'_train':v for k,v in exact.items()})
        data.update({k+'_validation':v for k,v in materialize(corpus,np.array([[0,3]])).items()})
        config=dict(task='sokoban',seed=5,model=dict(dimension=128,hidden=16,cap=256,metric='directed_max'),
            training=dict(steps=3,batch_size=64,decision_batch_size=8,learning_rate=.0001,rank_weight=1.,
                rank_temperature=1.,eval_every=1,cpu_threads=1,unreachable_target=240.,training_probe_batches=2,
                solution_structure=dict(mode='path_multiscale'),evaluation_checkpoint='last.pt'),
            evaluation=dict(decision_count=8,rollout_count=1,horizon=32))
        with tempfile.TemporaryDirectory() as tmp,patch('experiments.new_tasks_qmap.src.solution_structure.prepare',return_value=(corpus,data,dict(finite_max=2))):
            out=Path(tmp)/'run';train(config,out,'structure-fixture','cpu')
            summary=json.loads((out/'summary.json').read_text())
            self.assertEqual(summary['evaluated_step'],3);self.assertEqual(summary['evaluated_checkpoint'],'last.pt')
            self.assertIn('solution_paths',summary['metrics']['cohorts']['validation'])
            saved=torch.load(out/'last.pt',weights_only=False)
            self.assertEqual(saved['rank_draws'].sum(),24);self.assertEqual(saved['pair_draws'].sum(),192)
            train(config,out,'structure-fixture','cpu',resume=True)
            after=torch.load(out/'last.pt',weights_only=False)
            for key,value in saved['model'].items():torch.testing.assert_close(value,after['model'][key],rtol=0,atol=0)


if __name__ == '__main__': unittest.main()
