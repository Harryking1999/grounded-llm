"""Efficiency selection must use validation trajectories and preserve completion."""
from contextlib import ExitStack
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
from grounded_llm.core.files import atomic_json
from grounded_llm.core.qmap.model import StateEncoder
from grounded_llm.tasks.token_swapping.relation_data import TokenCorpus
from experiments.new_tasks_qmap.tests.test_exact_pairs import small_oracle
from experiments.new_tasks_qmap.src.token_acceptance import validation_efficiency_candidates
from experiments.new_tasks_qmap.src.relations import completion_criteria


class EfficiencySelectionTests(unittest.TestCase):
    def test_cost_protocol_explicitly_selects_training_summary_completion_profile(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);base=root/'base.json';cost=root/'cost.json'
            expected=dict(required_test_cohorts={'token_swapping':['test']},minimum_completion_rate=.8,target_completion_rate=.9)
            atomic_json(base,expected)
            atomic_json(cost,dict(minimum_completion_rate=.95,target_completion_rate=.98,candidates=['candidate']))
            with self.assertRaisesRegex(ValueError,'completion criteria'):completion_criteria(cost,'token_swapping')
            atomic_json(cost,dict(training_summary_criteria=str(base),minimum_completion_rate=.95,target_completion_rate=.98))
            self.assertEqual(completion_criteria(cost,'token_swapping'),expected)

    def test_validation_only_cost_selection_rejects_completion_regression(self):
        with tempfile.TemporaryDirectory() as directory,ExitStack() as stack:
            root=Path(directory);graphs=root/'graphs';graphs.mkdir()
            small_oracle(graphs,5,[(0,1),(1,2),(2,3),(3,4)])
            corpus=TokenCorpus(graphs);stack.callback(corpus.close)
            data={f'tasks_{name}':np.array([[0,1]]) for name in ['validation','unseen_endpoint_validation','unseen_both_validation','test']}
            config=dict(model=dict(dimension=4,hidden=8,cap=32,metric='l1'),training=dict(cpu_threads=1),evaluation={})
            names=['baseline','better_cost','bad_completion'];values=[(100,70),(99,85),(90,99)]
            for index,name in enumerate(names):
                p=root/name;(p/'data').mkdir(parents=True);atomic_json(p/'summary.json',{})
                np.save(p/'data/endpoint_partition.npy',np.zeros(corpus.size,dtype=np.int8))
                model=StateEncoder(input_dim=25,**config['model'])
                with torch.no_grad():model.network[-1].bias.fill_(index)
                torch.save(dict(model=model.state_dict(),config=config,input_dim=25,step=1,exposure=np.ones(corpus.size),pair_draws=np.ones(5)),p/'best.pt')
            def fake_evaluate(model,corpus,data,config,cohorts,exposure):
                self.assertEqual(set(cohorts),{'validation','unseen_endpoint_validation','unseen_both_validation'})
                tag=int(model.network[-1].bias[0]);solved,optimal=values[tag]
                p=dict(count=100,success_rate=solved/100,optimal_rate=optimal/100,mean_cost_ratio_on_success=1.03,
                    by_distance={'21':dict(count=100,solved=solved,optimal=optimal)})
                return dict(cohorts={name:dict(finite_huber=.1,candidate_soft_rank_loss=1.,planning=dict(q=p)) for name in cohorts})
            protocol=dict(candidates=names,selection_queries=128,horizon=128,long_distance_above=20,minimum_completion_rate=.95)
            with patch('experiments.new_tasks_qmap.src.token_acceptance.load',return_value=(corpus,data,{})),\
                 patch('experiments.new_tasks_qmap.src.token_acceptance.evaluate',side_effect=fake_evaluate),\
                 patch.object(corpus,'close'):
                candidates,chosen=validation_efficiency_candidates(protocol,root,'cpu')
            self.assertEqual(chosen['run'],'better_cost');self.assertEqual(len(candidates),3)
            self.assertEqual(chosen['checkpoint'],'best.pt')
            self.assertEqual(chosen['minimum_validation_optimal_including_long'],.85)


if __name__=='__main__':unittest.main()
