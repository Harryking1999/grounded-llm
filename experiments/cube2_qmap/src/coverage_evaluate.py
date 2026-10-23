"""Select by development cohorts, then open sealed tests once at study end."""
import argparse
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

from grounded_llm.core.files import atomic_json
from grounded_llm.tasks.cube2.environment import Cube2
from grounded_llm.tasks.cube2.evaluate import rollout
from .coverage_data import model_from
from .near_goal_continuation import inspect
from .fixed_goal import exhaustive_policy


def paired_change(before,after):
    a=np.asarray(before,dtype=bool);b=np.asarray(after,dtype=bool)
    gained=int((~a&b).sum());lost=int((a&~b).sum());discordant=gained+lost
    p=min(1.,2*sum(math.comb(discordant,k) for k in range(min(gained,lost)+1))/2**discordant) if discordant else 1.
    differences=b.astype(int)-a.astype(int)
    rng=np.random.default_rng(2026102403)
    samples=np.mean(differences[rng.integers(len(a),size=(10000,len(a)))],axis=1)
    return dict(count=len(a),before=int(a.sum()),after=int(b.sum()),gained=gained,lost=lost,
        both_solved=int((a&b).sum()),both_failed=int((~a&~b).sum()),net=gained-lost,
        paired_bootstrap_95_percent_interval=np.quantile(samples,[.025,.975]).tolist(),
        exact_mcnemar_two_sided_p=p,
        gained_indices=np.flatnonzero(~a&b).tolist(),lost_indices=np.flatnonzero(a&~b).tolist())


def run(args):
    torch.set_num_threads(2);started=time.monotonic()
    args.out.mkdir(parents=True,exist_ok=False)
    candidates=[]
    for directory in args.run:
        if not (directory/'complete.json').exists():raise ValueError(f'Arm unfinished: {directory}')
        summary=json.loads((directory/'summary.json').read_text())
        if summary['best_eligible']:
            candidates.append((tuple(summary['best_validation_and_restoration']),directory,
                summary['best_update']))
    if not candidates:raise ValueError('No restoration-eligible candidate')
    candidates.sort(key=lambda row:(row[0],-row[2]),reverse=True)
    score,directory,update=candidates[0]
    # Save the decision before accessing any sealed test results.
    atomic_json(args.out/'selection.json',dict(selected_run=str(directory),selected_checkpoint=str(directory/'best.pt'),
        validation_and_restoration=score,update=update,
        candidates=[dict(run=str(d),score=s,update=u) for s,d,u in candidates],
        rule='validation first, original restoration >=385; fewer updates breaks exact ties',
        frozen_flan_map_replaced=False,source_commit=args.source_commit))
    env=Cube2(args.oracle,metric='QTM_URF_6');fresh=np.load(args.data/'cohorts.npz')
    old=np.load(args.old_cohorts)
    cohorts={k:fresh[k] for k in ('validation','test','novel_goal_test')}
    cohorts.update(original=fresh['original'],old_small_ood=old['small_ood'],old_validation=old['validation'])
    np.savez_compressed(args.out/'evaluated_tasks.npz',**cohorts)
    results={};oracle={}
    for name,path in [('baseline1000',args.checkpoint1000),('baseline2000',args.checkpoint2000),
                      ('selected',directory/'best.pt')]:
        model=model_from(path,env,args.device)
        results[name]=dict(checkpoint=str(path),scores={k:inspect(model,env,tasks,args.device,32)
            for k,tasks in cohorts.items()})
        if name=='selected':
            for cohort,tasks in cohorts.items():
                oracle[cohort]=rollout(model,env,tasks,args.device,2,horizon=32,return_tasks=True)['oracle']
        if name in ('baseline2000','selected'):
            results[name]['full_restoration']=exhaustive_policy(model,env,fresh['state_mask'],
                {'evaluation':{'horizon':32}},args.device)
        atomic_json(args.out/(name+'.json'),results[name])
        print(json.dumps(dict(stage='tested',model=name,scores={k:v['rollout']['solved']
            for k,v in results[name]['scores'].items()})),flush=True)
    paired={baseline:{name:paired_change(results[baseline]['scores'][name]['rollout']['per_task']['solved'],
        results['selected']['scores'][name]['rollout']['per_task']['solved']) for name in cohorts}
        for baseline in ('baseline1000','baseline2000')}
    atomic_json(args.out/'paired.json',paired);atomic_json(args.out/'oracle_greedy.json',oracle)
    atomic_json(args.out/'complete.json',dict(source_commit=args.source_commit,seconds=time.monotonic()-started,
        horizon=32,selected_run=str(directory),frozen_flan_map_replaced=False,
        note='Development cohorts are selection-biased; heldout depth-2/3 goals exclude continuation labels, not inherited pretraining. Old OOD is regression, never used by this phase for selection or mining.'))


def main():
    p=argparse.ArgumentParser(__doc__)
    for name in ('data','oracle','old-cohorts','checkpoint1000','checkpoint2000','out'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--run',type=Path,action='append',required=True)
    p.add_argument('--device',default='cuda');p.add_argument('--source-commit',required=True)
    run(p.parse_args())


if __name__=='__main__':main()
