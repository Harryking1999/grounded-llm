"""Risks specific to complete-goal FLAN SFT adapters."""
import json
from pathlib import Path
import numpy as np
import torch
from grounded_llm.tasks.blocks import rules
from grounded_llm.tasks.state_graph_sft import StateTask, demonstration, prompt, score
from grounded_llm.core.reader.numbering import numbered_step

ROOT=Path(__file__).resolve().parents[1]


def block_row(operation,start,goal):
    return dict(task='blocks',operation=operation,start=str(start),goal=str(goal),
        map_checkpoint=str(ROOT/'tasks/blocks/Qmap/checkpoints/decision_clear_300_4f413b5.pt'),
        context={},initial_text='A complete board target.',reference_actions=[])


def test_complete_clearing_candidates_and_distance():
    tile=rules.PLACEMENTS[0][0]
    row=block_row('remove',tile,0);env=StateTask(row,ROOT)
    options=list(env.transitions(tile))
    assert len(options)==len(rules.legal_indices(tile))
    assert env.distance(tile,0)==1
    action,child,_=next(x for x in options if x[1]==0)
    step=env.make_step(tile,0)
    assert torch.allclose(torch.tensor(step.candidate_map_distances),env.reader.distance(step.map_batch.vectors[0,2:],step.map_batch.vectors[0,1]))
    perm=tuple(reversed(range(len(options))));changed=numbered_step(step,perm)
    chosen=changed.candidate_actions.index(action)+1
    assert changed.execute(chosen)[1]==0
    assert score(f'Reason: places the tile.\n<action>{chosen}</action>',changed,env)['optimal']
    assert np.isinf(env.distance(0,tile))


def test_nonempty8_goal_and_placement_demonstration():
    tile0=rules.PLACEMENTS[0][0]
    tile1=next(tile for tile,_ in rules.PLACEMENTS if not(tile&tile0))
    env=StateTask(block_row('remove',tile0|tile1,tile1),ROOT)
    assert env.distance(tile0|tile1,tile1)==1
    step=env.make_step(tile0|tile1,tile1)
    action=next(a for a,c,_ in env.transitions(tile0|tile1) if c==tile1)
    row=dict(env.record,reference_actions=[action])
    done=env.make_step(tile1,tile1)
    plain=demonstration(row,[step,done],with_map=False)
    mapped=demonstration(row,[step,done],np.random.default_rng(1),with_map=True)
    assert plain.executed_path==mapped.executed_path
    assert 'Map-distance ranking' not in plain.turns[0].answer_text
    assert mapped.turns[-1].answer_text.endswith('<done/>')


def test_complete_bridge_goal_weighted_action():
    row=dict(task='bridge',start=0,goal=3,context={'times':[1,2,5,10,15]},
        map_checkpoint=str(ROOT/'tasks/bridge/Qmap/checkpoints/bridge5_multigoal.pt'),
        initial_text='Match the full people and torch state.',reference_actions=[])
    env=StateTask(row,ROOT);step=env.make_step(0,3)
    assert env.distance(0,3)==1
    chosen=step.candidate_actions.index('[0]')+1
    assert score(f'Reason: cross.\n<action>{chosen}</action>',step,env)['optimal']
    assert env.env.times==(1,2,5,10,15)


def test_study_budgets_and_fixed_big_ood():
    root=ROOT/'experiments/map_sft_baselines/configs'
    study=json.loads((root/'flan_generalization_50k.json').read_text())
    assert sum(map(len,study['subclasses'].values()))>=16
    for cohort in study['cohorts']:
        assert sum(cohort['counts'].values())==50000
        assert max(cohort['counts'].values())-min(cohort['counts'].values())<=1
        assert {'rush_hour','game24'}<=set(cohort['big_ood_tasks'])
        for group in ['map']:
            cfg=json.loads((root/f"flan_generalization_{len(cohort['tasks'])}_50k_{group}.json").read_text())
            t=cfg['training']
            assert t['world_size']==2 and t['micro_batch']*t['world_size']*t['gradient_accumulation_steps']==t['global_batch']==4
            assert cfg['execution']['evaluation_gpus']==[6]
            assert cfg['execution']['gpu_pairs']==[[2*(len(cohort['tasks'])-4),2*(len(cohort['tasks'])-4)+1]]
            assert cfg['baselines']['B2_flan_four']['renumber_each_epoch']
            assert cfg['map']['feature_ffn_hidden_dim']==512
            assert cfg['evaluation']['tasks']==len(cohort['tasks'])*study['evaluation_per_group']
            assert t['save_steps']==cfg['data']['samples_per_epoch']//t['global_batch']
            assert cfg['evaluation']['priority_step_interval']==50000
            from experiments.map_sft_baselines.src.contract import checkpoint_priority
            steps=[12500,25000,37500,50000]
            assert sorted(steps,key=lambda s:checkpoint_priority(s,cfg))==[50000,37500,25000,12500]


def test_six_person_ood_pool_excludes_stranded_torch():
    from experiments.map_sft_baselines.src.generalization_evaluation import bridge_goal_pool
    from grounded_llm.tasks.bridge.environment import Bridge
    pool=bridge_goal_pool(6)
    assert set(pool)==set(range(1,128))-{1,126}
    costs=Bridge([2,3,5,7,11,17]).graph(0)['distances']
    assert np.isfinite(costs[pool]).all()
    assert 127 in pool and not np.isfinite(costs[[1,126]]).any()


def test_evaluation_half_preserves_strata_and_ignores_model_outcomes():
    from collections import Counter
    from experiments.map_sft_baselines.src.generalization_prepare import evaluation_subset
    rows=[]
    for variant,size in [('a',200),('b',100),('c',100),('d',50),('e',50)]:
        for i in range(size):
            length=1 if i<size//2 else 3
            rows.append(dict(task='game24',variant=variant,trajectory_id=f'{variant}_{i:03d}',
                shortest_moves=length,reference_actions=['move']*length,cohort='id',map_success=i%2))
    chosen=evaluation_subset(rows,250,2026102102)
    assert len(chosen)==len({r['trajectory_id'] for r in chosen})==250
    assert Counter(r['variant'] for r in chosen)==dict(a=100,b=50,c=50,d=25,e=25)
    assert Counter(r['shortest_moves'] for r in chosen if r['variant']=='a')=={1:50,3:50}
    altered=[dict(r,cohort='big_ood',map_success=1-r['map_success']) for r in rows]
    ids=lambda records:[r['trajectory_id'] for r in records]
    assert ids(chosen)==ids(evaluation_subset(rows,250,2026102102))
    assert ids(chosen)==ids(evaluation_subset(altered,250,2026102102))
