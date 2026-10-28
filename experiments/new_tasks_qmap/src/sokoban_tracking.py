"""Paired fixed-query completion, cycle migration and retention diagnostics."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from grounded_llm.core.files import atomic_json
from grounded_llm.core.qmap.model import StateEncoder
from .relation_data import load
from .relation_evaluate import QCache,planning_ids,rollout
from .sokoban_failures import mine_fixed_wall_failures


def cycle_identities(policy):
    take=policy['cycle_member'];source=policy['tasks'][take,0];following=policy['successor'][take]
    return {tuple(row) for row in np.column_stack([policy['tasks'][take,1],np.minimum(source,following),np.maximum(source,following)]).tolist()}


def paired_summary(before,after):
    if not np.array_equal(before['tasks'],after['tasks']) or not np.array_equal(before['distance'],after['distance']):
        raise ValueError('Paired tracking requires identical ordered queries and exact distances')
    old=before['solved'];new=after['solved'];lost=old&~new;gained=~old&new
    old_cycle=before['cycle_member'];new_cycle=after['cycle_member'];same=before['successor']==after['successor']
    left=cycle_identities(before);right=cycle_identities(after)
    return dict(queries=len(old),before_solved=int(old.sum()),after_solved=int(new.sum()),
        newly_solved=int(gained.sum()),previously_solved_lost=int(lost.sum()),previously_solved_retained=int((old&new).sum()),
        retention_rate=float((old&new).sum()/old.sum()) if old.any() else None,
        lost_to_dead_basin=int((lost&after['dead_basin']).sum()),
        lost_to_budget=int((lost&after['budget_exhausted']).sum()),
        lost_to_finite_cycle=int((lost&~after['dead_basin']&~after['budget_exhausted']).sum()),
        cycle_identities=dict(before=len(left),after=len(right),persistent=len(left&right),disappeared=len(left-right),new=len(right-left)),
        original_cycle_contexts=dict(count=int(old_cycle.sum()),now_solved=int((old_cycle&new).sum()),
            same_cycle=int((old_cycle&new_cycle&same).sum()),different_two_cycle=int((old_cycle&new_cycle&~same).sum()),
            left_two_cycle_but_still_failed=int((old_cycle&~new_cycle&~new).sum())),
        previously_correct_first_actions_lost=int((before['optimal_action']&~after['optimal_action']).sum()))


def evaluation_policies(model,corpus,data,config):
    cache=QCache(model,corpus,next(model.parameters()).device);result={};summary={}
    names=('validation','test','unseen_goal_validation','unseen_goal_test','original_training_goals','original_unseen_goals')
    for name in names:
        if 'tasks_'+name not in data:continue
        pool=data['tasks_'+name];truth=data['distances_'+name]
        selected=pool[planning_ids(corpus,pool,truth,config['evaluation']['rollout_count'],
            np.random.default_rng(config['seed']+len(name)),config['evaluation'].get('planning_distance_bins'))]
        records=[];metrics=rollout(corpus,cache,selected,config['evaluation']['horizon'],'q',edge_records=records)
        solved=np.zeros(len(selected),dtype=bool);dead=solved.copy();cycles=solved.copy()
        for row in records:
            np.logical_or.at(solved,row['query_index'],row['next_is_goal'])
            np.logical_or.at(cycles,row['query_index'],row['next_repeated']&~row['next_is_goal'])
            chosen=row['child_distances'][np.arange(len(row['choice'])),row['choice']]
            np.logical_or.at(dead,row['query_index'],~np.isfinite(chosen))
        if int(solved.sum())!=metrics['solved']:raise ValueError('Evaluation query tracking differs from native rollout')
        result[name]=dict(tasks=selected,solved=solved,dead=dead,cycles=cycles)
        summary[name]={key:metrics[key] for key in ('count','solved','optimal','cycles','entered_unreachable')}
    return result,summary


def evaluation_retention(before,after):
    if set(before)!=set(after):raise ValueError('Evaluation cohorts differ')
    result={}
    for name in before:
        left,right=before[name],after[name]
        if not np.array_equal(left['tasks'],right['tasks']):raise ValueError('Evaluation queries differ')
        old,new=left['solved'],right['solved'];lost=old&~new
        result[name]=dict(queries=len(old),before_solved=int(old.sum()),after_solved=int(new.sum()),
            newly_solved=int((~old&new).sum()),previously_solved_lost=int(lost.sum()),
            previously_solved_retained=int((old&new).sum()),lost_entered_unreachable=int((lost&right['dead']).sum()),
            lost_entered_cycle=int((lost&right['cycles']).sum()))
    return result


def collect(checkpoint,corpus,data,device,horizon):
    saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
    if saved['task']!='sokoban' or saved['input_dim']!=corpus.input_dim:
        raise ValueError('Tracking checkpoint differs from the fixed Sokoban input scope')
    model=StateEncoder(input_dim=saved['input_dim'],**saved['config']['model']).to(device)
    model.load_state_dict(saved['model']);model.eval().requires_grad_(False)
    records=[]
    report,_=mine_fixed_wall_failures(model,corpus,data,device,maximum_steps=horizon,policy_records=records)
    policy={key:np.concatenate([row[key] for row in records]) for key in records[0]}
    order=np.lexsort((policy['tasks'][:,1],policy['tasks'][:,0]));policy={key:value[order] for key,value in policy.items()}
    if len(np.unique(policy['tasks'],axis=0))!=len(policy['tasks']):raise ValueError('Duplicate tracking query')
    if int(policy['solved'].sum())!=report['solved']:raise ValueError('Tracking differs from native complete policy result')
    metadata=dict(checkpoint=str(checkpoint),step=saved['step'],source_commit=saved['source_commit'],
        queries=report['queries'],solved=report['solved'],success_rate=report['solved']/report['queries'],
        two_cycle_contexts=int(policy['cycle_member'].sum()),two_cycle_identities=len(cycle_identities(policy)),
        unreachable_basins=int(policy['dead_basin'].sum()),budget_exhausted=int(policy['budget_exhausted'].sum()))
    evaluation,metadata['evaluation_cohorts']=evaluation_policies(model,corpus,data,saved['config'])
    return metadata,policy,evaluation


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--checkpoints',type=Path,nargs='+',required=True)
    parser.add_argument('--data',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--device',default='cpu');parser.add_argument('--horizon',type=int,default=256)
    args=parser.parse_args();args.out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2);start=time.time()
    cfg=torch.load(args.checkpoints[0],map_location='cpu',weights_only=False)['config'];corpus,data,_=load(cfg,args.data)
    metadata=[];policies=[];evaluations=[]
    for i,path in enumerate(args.checkpoints):
        entry,policy,evaluation=collect(path,corpus,data,args.device,args.horizon);metadata.append(entry);policies.append(policy);evaluations.append(evaluation)
        extra={f'evaluation_{name}_{field}':value for name,rows in evaluation.items() for field,value in rows.items()}
        np.savez_compressed(args.out/f"policy_{i}_{entry['step']}.npz",**policy,**extra)
        print(json.dumps(entry),flush=True)
    result=dict(snapshots=metadata,comparisons=[dict(before=i,after=j,**paired_summary(policies[i],policies[j]),
        evaluation_retention=evaluation_retention(evaluations[i],evaluations[j]))
        for i in range(len(policies)) for j in range(i+1,len(policies))],
        seconds=time.time()-start,horizon=args.horizon,optimizer_updates=0,
        condition='Same permitted training queries, exact complete goals and native greedy policy; frozen snapshots. Cycle identities include the goal and both physical states. Completion retention is measured per query.')
    atomic_json(args.out/'summary.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':main()
