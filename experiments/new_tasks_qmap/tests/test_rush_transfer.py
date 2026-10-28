import unittest
from copy import deepcopy
import numpy as np

from grounded_llm.tasks.rush_hour.environment import RushHour
from experiments.new_tasks_qmap.src.rush_transfer import action_context,canonical_pair,add_observation,summarize,training_groups


class RushTransferTests(unittest.TestCase):
    def test_local_context_ignores_ordinary_names_but_transports_action_roles(self):
        cells=list('o'*36)
        for letter,indices in [('A',[12,13]),('B',[9,15]),('C',[25,26])]:
            for index in indices:cells[index]=letter
        board=''.join(cells);translation=str.maketrans({'B':'Y','C':'X'})
        contexts=[]
        for env,labels in [(RushHour(board),('B','C')),(RushHour(board.translate(translation)),('Y','X'))]:
            source=env.start;goal=list(source);goal[env.labels[labels[0]]]=3;goal[0]=3;env.validate(tuple(goal))
            actions=list(env.transitions(source));physical=[child for _,child,_ in actions]
            a=next(i for i,(action,_,_) in enumerate(actions) if action==dict(vehicle=labels[0],displacement=-1))
            b=next(i for i,(action,_,_) in enumerate(actions) if action==dict(vehicle=labels[1],displacement=1))
            context=action_context(env,source,goal,physical)
            contexts.append((env,context,a,b))
        for level in ('direct','neighborhood'):
            values=[canonical_pair(env,context,a,b,level) for env,context,a,b in contexts]
            self.assertEqual(values[0],values[1])
            env,context,a,b=contexts[0];key,sign=canonical_pair(env,context,b,a,level)
            self.assertEqual(key,values[0][0]);self.assertEqual(sign,-values[0][1])

    def test_teacher_disagreement_is_separated_from_model_transfer_failure(self):
        bank={};key=('same physical local context',)
        add_observation(bank,key,-1,-1,0,{'source':0})
        add_observation(bank,key,-1,-1,1,{'source':1})
        held=[(key,-1,1,0,{'source':2})]
        result=summarize(bank,held)
        self.assertEqual(result['train_fitted_pairs'],1)
        self.assertEqual(result['train_fitted_held_correct'],0)
        result=summarize(bank,[(key,1,1,0,{'source':2})])
        self.assertEqual(result['held_label_differs_from_consistent_training'],1)
        self.assertEqual(result['train_fitted_pairs'],0)
        add_observation(bank,key,1,1,2,{'source':3})
        result=summarize(bank,held)
        self.assertEqual(result['training_label_varies'],1)
        self.assertEqual(result['train_fitted_pairs'],0)

    def test_fit_gate_applies_to_each_layout_instead_of_a_dominant_layout(self):
        bank={};key=('context',)
        for _ in range(100):add_observation(bank,key,-1,-1,'large',{})
        add_observation(bank,key,-1,-1,'small',{})
        add_observation(bank,key,-1,1,'small',{})
        result=summarize(bank,[(key,-1,1,0,{})])
        self.assertEqual(result['train_fitted_pairs'],0)
        self.assertEqual(result['train_underfit_pairs'],1)

    def test_prepared_groups_use_training_cases_and_preserve_opposite_teacher_orders(self):
        bank={}
        add_observation(bank,('contrast',),-1,-1,'left',{'source':1,'goal':9})
        add_observation(bank,('contrast',),1,1,'right',{'source':2,'goal':8})
        add_observation(bank,('retain',),-1,-1,'left',{'source':3,'goal':7})
        add_observation(bank,('retain',),-1,-1,'right',{'source':4,'goal':6})
        groups=training_groups({'direct':bank})
        self.assertEqual([g[0] for g in groups],['global_context_contrast','retain_consistent_relation'])
        self.assertEqual({case['source'] for _,_,a,b in groups for case in (a,b)},{1,2,3,4})

    def test_structural_continuation_preserves_other_training_conditions(self):
        from experiments.new_tasks_qmap.src.relations import validate_continuation
        previous=dict(task='rush_hour',training=dict(steps=60000,batch_size=65536,
            current_failure_replay=dict(contexts_per_batch=256,queries='all')))
        proposed=deepcopy(previous);proposed['training']['steps']=80000
        proposed['continuation']=dict(mode='structural_replay')
        proposed['training']['current_failure_replay']['structural_groups']=dict(bank='groups.npz',groups_per_batch=64)
        validate_continuation(previous,proposed,60000)
        changed=deepcopy(proposed);changed['training']['batch_size']=32768
        with self.assertRaises(ValueError):validate_continuation(previous,changed,60000)
        changed=deepcopy(proposed);changed['training']['current_failure_replay']['structural_groups']['groups_per_batch']=128
        with self.assertRaises(ValueError):validate_continuation(previous,changed,60000)


if __name__=='__main__':unittest.main()
