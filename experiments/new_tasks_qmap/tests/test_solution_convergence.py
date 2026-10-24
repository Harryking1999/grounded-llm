"""A budget extension must preserve optimizer/RNG trajectory and all conditions."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import json
import numpy as np
import torch
from experiments.new_tasks_qmap.tests.test_solution_mixture import bank
from experiments.new_tasks_qmap.src.relation_data import materialize
from experiments.new_tasks_qmap.src.relations import train,validate_continuation,retain_checkpoint
from experiments.new_tasks_qmap.src.solution_convergence import ConvergenceProbe,exposure_counts,whole_graph_structure


class ConvergenceTests(unittest.TestCase):
    def test_only_budget_and_diagnostic_metadata_can_change(self):
        old=dict(task='sokoban',training=dict(steps=6,learning_rate=.0001,solution_structure=dict(graph_fraction=.2)),
                 data=dict(source='fixed'),evaluation=dict(horizon=32))
        new=deepcopy(old);new['training']['steps']=12;new['convergence']=dict(checkpoints=[12])
        validate_continuation(old,new,6)
        for change in [('training','learning_rate',.001),('data','source','other'),('evaluation','horizon',64)]:
            bad=deepcopy(new);bad[change[0]][change[1]]=change[2]
            with self.assertRaises(ValueError):validate_continuation(old,bad,6)
        bad=deepcopy(new);bad['training']['solution_structure']['graph_fraction']=.5
        with self.assertRaises(ValueError):validate_continuation(old,bad,6)

    def test_full_continuation_and_observation_preserve_update_trajectory(self):
        corpus,original=bank();data={k:np.concatenate([v,v]) for k,v in original.items() if k.endswith('_train')}
        data.update(broad_point_count=len(original['pairs_train']),broad_rank_count=len(original['tasks_train']))
        data.update({k+'_validation':v for k,v in materialize(corpus,np.array([[0,3]])).items()})
        cfg=dict(task='sokoban',seed=19,model=dict(dimension=128,hidden=16,cap=256,metric='directed_max'),
            training=dict(steps=4,batch_size=64,decision_batch_size=8,learning_rate=.0001,rank_weight=1.,
                rank_temperature=1.,eval_every=2,cpu_threads=1,unreachable_target=240.,evaluation_checkpoint='last.pt',
                training_probe_batches=2,solution_structure=dict(mode='broad_graph_mixture',graph_fraction=.5,
                graphs_per_batch=2,neighborhood_size=3,root_layer_sampling='uniform')),
            evaluation=dict(decision_count=8,rollout_count=1,horizon=32))
        diagnostics=dict(seed=32,checkpoints=[4],path_distance_bins=[5,10,20,32],queries_per_goal_layer=1,
                         order_distance_bins=[1,2,5,10,20,32,64,float('inf')],order_per_layer=2)
        def prepared(config,path):
            path.mkdir(exist_ok=True);return corpus,data,dict(finite_max=2)
        with tempfile.TemporaryDirectory() as tmp,patch('experiments.new_tasks_qmap.src.solution_structure.prepare',side_effect=prepared):
            root=Path(tmp);train(cfg,root/'continuous','fixture','cpu')
            short=deepcopy(cfg);short['training']['steps']=2;train(short,root/'short','fixture','cpu')
            extended=deepcopy(cfg);extended['continuation']=dict(checkpoint=str(root/'short/last.pt'));extended['convergence']=diagnostics
            train(extended,root/'extended','fixture','cpu')
            a=torch.load(root/'continuous/last.pt',weights_only=False);b=torch.load(root/'extended/last.pt',weights_only=False)
            for name,value in a['model'].items():torch.testing.assert_close(value,b['model'][name],rtol=0,atol=0)
            for key in ('pair_draws','rank_draws','exposure'):np.testing.assert_array_equal(a[key],b[key])
            self.assertEqual(a['rng'],b['rng']);torch.testing.assert_close(a['torch_rng'],b['torch_rng'],rtol=0,atol=0)
            for sid,row in a['optimizer']['state'].items():
                for key,value in row.items():torch.testing.assert_close(value,b['optimizer']['state'][sid][key],rtol=0,atol=0)
            self.assertTrue((root/'extended/checkpoint-4.pt').exists())
            report=json.loads((root/'extended/convergence/4.json').read_text());self.assertEqual(report['total_decision_draws'],32)
            self.assertTrue(json.loads((root/'extended/continuation.json').read_text())['optimizer_restored'])
            prior=torch.load(root/'extended/checkpoint-4.pt',weights_only=False)
            train(extended,root/'extended','fixture','cpu',resume=True)
            later=torch.load(root/'extended/last.pt',weights_only=False)
            for name,value in prior['model'].items():torch.testing.assert_close(value,later['model'][name],rtol=0,atol=0)

    def test_retained_checkpoint_survives_atomic_source_replacement(self):
        from grounded_llm.core.files import atomic_save
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);atomic_save(dict(step=1),root/'last.pt');retain_checkpoint(root/'last.pt',root/'snapshot.pt')
            atomic_save(dict(step=2),root/'last.pt')
            self.assertEqual(torch.load(root/'snapshot.pt',weights_only=False)['step'],1)

    def test_exposure_mean_does_not_claim_complete_epochs(self):
        row=exposure_counts(np.array([0,0,4,4]))
        self.assertEqual(row['mean_draws'],2);self.assertEqual(row['minimum'],0);self.assertEqual(row['unique_fraction'],.5)

    def test_all_start_execution_separates_goals_cycles_dead_and_protected_boundaries(self):
        # One fixed chosen successor per context; no search or teacher action.
        data=dict(tasks_train=np.array([[s,14] for s in [10,11,12,13,15,17]]),
            children_train=np.array([[s] for s in [11,14,13,12,16,18]]),
            distances_train=np.array([2,1,2,3,5,4]),
            child_distances_train=np.array([[v] for v in [1,0,3,2,np.inf,3]]),
            pairs_train=np.array([[s,14] for s in range(10,19)]))
        row=whole_graph_structure(data,np.arange(9,dtype=np.float32))['all_start_execution']['true_distance_at_most32']
        self.assertEqual([row[k] for k in ['starts','solved_within32','finite_cycle_detectable_within32',
            'proven_unreachable_entered_within32','protected_boundary_entered_within32','still_executing_at32']],
            [6,2,2,1,1,0])

    def test_all_start_execution_obeys_original32_step_limit(self):
        data=dict(tasks_train=np.array([[s,36] for s in range(1,36)]),children_train=np.arange(2,37)[:,None],
            distances_train=np.arange(35,0,-1),child_distances_train=np.arange(34,-1,-1)[:,None],
            pairs_train=np.array([[s,36] for s in range(1,37)]))
        row=whole_graph_structure(data,np.arange(35,-1,-1,dtype=np.float32))['all_start_execution']
        self.assertEqual(row['all_accepted']['solved_within32'],32)
        self.assertEqual(row['all_accepted']['still_executing_at32'],3)
        self.assertEqual(row['true_distance_at_most32']['starts'],32)


if __name__=='__main__':unittest.main()
