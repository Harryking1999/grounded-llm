"""Close admissible teacher routes without relaxing any protected relation."""
import argparse
import json
from pathlib import Path

import numpy as np

from grounded_llm.core.files import atomic_json
from grounded_llm.tasks.cube2.environment import Cube2
from .coverage_data import (CoverageGuard,keys_to_pairs,closure_keys,sample_cohort,
    load_coverage_config,hist,unique_tasks)


def prepare(cfg,args):
    opt=cfg['data'];env=Cube2(args.oracle,metric='QTM_URF_6');old=np.load(args.source/'cohorts.npz')
    meta=json.loads((args.source/'ready.json').read_text());rng=np.random.default_rng(opt['final_query_seed'])
    keys=old['excluded_relation_keys'];cohorts={k:old[k] for k in old.files if k not in ('state_mask','heldout_goals','excluded_relation_keys')}
    # New final queries are fixed before inspecting any candidate route or model.
    evaluation_guard=CoverageGuard(env,keys_to_pairs(keys,env.size),old['state_mask'],np.empty(0,dtype=np.int64))
    for name,group in (('relation_test','train'),('goal_test','test')):
        rows=sample_cohort(env,np.asarray(meta['goal_ids'][group]),opt['evaluation_per_group'],rng,evaluation_guard,evaluation_guard)
        cohorts['previous_'+name]=cohorts[name];cohorts[name]=rows
        keys=np.unique(np.r_[keys,closure_keys(env,rows)])
        evaluation_guard=CoverageGuard(env,keys_to_pairs(keys,env.size),old['state_mask'],np.empty(0,dtype=np.int64))
    guard=CoverageGuard(env,keys_to_pairs(keys,env.size),old['state_mask'],old['heldout_goals'])
    candidates=np.load(args.source/'Noncanonical.npy');rng=np.random.default_rng(opt['route_seed'])
    quota=opt['complete_routes']//len(opt['route_distance_bins']);counts=np.zeros(len(opt['route_distance_bins']),int)
    accepted=[];lengths=[];attempts=0;seen=set()
    for index in rng.permutation(len(candidates))[:opt['maximum_candidate_routes']]:
        source,goal=candidates[index];depth=int(env.distance(source,goal))
        bucket=next(i for i,(low,high) in enumerate(opt['route_distance_bins']) if low<=depth<=high)
        if counts[bucket]>=quota:continue
        attempts+=1;state=source;route=[]
        while state!=goal:
            route.append((int(state),int(goal)));children=env.successors(int(state))
            state=children[env.distance(children,goal).argmin()]
        rows=np.asarray(route,dtype=np.int64)
        if guard.task_blocked(env,rows).any() or (int(source),int(goal)) in seen:continue
        accepted.append(rows);lengths.append(depth);counts[bucket]+=1;seen.add((int(source),int(goal)))
        if (counts==quota).all():break
    if not (counts==quota).all():
        args.out.mkdir(parents=True,exist_ok=False)
        atomic_json(args.out/'preparation.json',dict(status='insufficient_complete_routes',source_commit=args.source_commit,
            requested_per_bin=quota,accepted_per_bin=counts.tolist(),attempted_routes=attempts,
            candidate_limit=opt['maximum_candidate_routes'],optimizer_updates=0,protection_relaxed=False))
        raise ValueError(f'Insufficient complete routes without relaxing protection: {counts.tolist()}')
    rows=unique_tasks(np.concatenate(accepted),env.size);restore=np.load(args.source/'Restoration.npy')
    restore=restore[~guard.task_blocked(env,restore)]
    if guard.task_blocked(env,rows).any():raise ValueError('Protected route label')
    args.out.mkdir(parents=True,exist_ok=False)
    np.save(args.out/'Noncanonical.npy',rows);np.save(args.out/'Restoration.npy',restore)
    np.savez_compressed(args.out/'cohorts.npz',**cohorts,excluded_relation_keys=keys,state_mask=old['state_mask'],heldout_goals=old['heldout_goals'])
    np.savez_compressed(args.out/'routes.npz',rows=np.concatenate(accepted),lengths=np.asarray(lengths))
    goals=np.asarray(meta['goal_ids']['train']);stars=np.column_stack((env.successors(goals).ravel(),np.repeat(goals,6)))
    allowed=~guard.task_blocked(env,stars)
    atomic_json(args.out/'ready.json',dict(source_commit=args.source_commit,train_decisions=len(rows),restore_decisions=len(restore),
        accepted_routes=len(accepted),attempted_routes=attempts,route_distance_counts=counts.tolist(),initial_distances=hist(lengths),
        accepted_goals=len(np.unique(rows[:,1])),original_training_goals=len(goals),goals_without_permitted_terminal_star=int((~allowed.reshape(-1,6).any(1)).sum()),
        route_decisions_before_deduplication=sum(lengths),protected_route_decisions=0,
        goal_ids=meta['goal_ids'],new_final_queries=True,previous_final_queries_regression_only=True,
        scope='Complete admissible routes only; unsupported goals remain disclosed, physical candidates never removed'))
    print(json.dumps(dict(accepted_routes=len(accepted),unique_decisions=len(rows),attempted_routes=attempts)),flush=True)


def main():
    p=argparse.ArgumentParser(__doc__)
    for name in ('config','source','oracle','out'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--source-commit',required=True);a=p.parse_args();prepare(load_coverage_config(a.config),a)


if __name__=='__main__':main()
