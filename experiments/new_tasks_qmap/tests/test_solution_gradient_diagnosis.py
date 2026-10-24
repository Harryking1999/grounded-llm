"""Check frozen diagnostics against existing loss evaluation and real Adam."""
from copy import deepcopy
import unittest
import numpy as np
import torch
from grounded_llm.core.qmap.model import StateEncoder
from experiments.new_tasks_qmap.tests.test_solution_mixture import bank
from experiments.new_tasks_qmap.src.relations import training_probe
from experiments.new_tasks_qmap.src.solution_gradient_diagnosis import component_gradients,frozen_adam_direction


class GradientDiagnosisTests(unittest.TestCase):
    def test_decomposition_matches_existing_objective_without_mutating_model(self):
        torch.manual_seed(41);torch.set_num_threads(1);corpus,data=bank()
        model=StateEncoder(input_dim=corpus.input_dim,dimension=8,hidden=16,cap=16,metric='directed_max')
        batch=dict(pairs=data['pairs_train'],point_truth=data['pair_distances_train'],tasks=data['tasks_train'],
            children=data['children_train'],child_truth=data['child_distances_train'],
            point_is_graph=np.arange(len(data['pairs_train']))%2==0,
            decision_is_graph=np.arange(len(data['tasks_train']))%2==0)
        options=dict(unreachable_target=15.,rank_weight=1.,rank_temperature=1.)
        before=deepcopy(model.state_dict())
        for parameter in model.parameters():parameter.grad=torch.ones_like(parameter)
        report=component_gradients(model,batch,corpus.features,options)
        original=training_probe(model,corpus,[batch],options)
        self.assertAlmostEqual(report['total'],original['distance_huber']+original['candidate_soft_rank_loss'],places=5)
        for name,value in model.state_dict().items():torch.testing.assert_close(value,before[name],rtol=0,atol=0)
        for parameter in model.parameters():torch.testing.assert_close(parameter.grad,torch.ones_like(parameter),rtol=0,atol=0)
        # Independent central difference through the existing read-only evaluator.
        direction=report['total_gradient']/report['total_gradient'].norm();epsilon=.001;values=[]
        for sign in [-1,1]:
            offset=0
            with torch.no_grad():
                for name,parameter in model.named_parameters():
                    size=parameter.numel();parameter.copy_(before[name]+sign*epsilon*direction[offset:offset+size].reshape(parameter.shape));offset+=size
            row=training_probe(model,corpus,[batch],options);values.append(row['distance_huber']+row['candidate_soft_rank_loss'])
        self.assertAlmostEqual((values[1]-values[0])/(2*epsilon),float(report['total_gradient'].norm()),delta=.005)

    def test_frozen_direction_matches_real_adam_and_preserves_saved_moments(self):
        torch.manual_seed(42);parameter=torch.nn.Parameter(torch.randn(4));optimizer=torch.optim.Adam([parameter],lr=.02)
        for _ in range(3):parameter.grad=torch.randn_like(parameter);optimizer.step()
        gradient=torch.randn_like(parameter);state=deepcopy(optimizer.state_dict());saved=deepcopy(state);before=parameter.detach().clone()
        predicted=frozen_adam_direction([parameter],gradient,state)
        torch.testing.assert_close(parameter,before,rtol=0,atol=0)
        for name,value in state['state'][0].items():torch.testing.assert_close(value,saved['state'][0][name],rtol=0,atol=0)
        parameter.grad=gradient;optimizer.step()
        torch.testing.assert_close(parameter.detach()-before,predicted,rtol=1e-4,atol=1e-7)


if __name__=='__main__':unittest.main()
