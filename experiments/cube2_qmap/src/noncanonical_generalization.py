"""Train/test the actual noncanonical target family, with held goal orbits."""
import argparse
import json
from pathlib import Path
import shutil
import time

import numpy as np
import torch

from grounded_llm.core.files import atomic_json, atomic_save
from grounded_llm.tasks.cube2.environment import Cube2
from grounded_llm.tasks.cube2.model import QMap
from .coordinate_fit import initialize, subset_batch, fit_metrics
from .coverage_data import CoverageGuard, keys_to_pairs, closure_keys, sample_cohort, hist, load_coverage_config
from .coverage_train import Pool
from .full_state_joint import fixed_batch, batch_losses
from .near_goal_continuation import training_tasks, inspect


def prepare(cfg,args):
    args.out.mkdir(parents=True,exist_ok=False);opt=cfg['data'];rng=np.random.default_rng(cfg['seed'])
    env=Cube2(args.oracle,metric='QTM_URF_6');old=np.load(args.source/'cohorts.npz')
    original=CoverageGuard(env,keys_to_pairs(old['excluded_relation_keys'],env.size),old['state_mask'],old['heldout_goals'])
    goals=np.arange(opt['goal_minimum'],opt['goal_maximum']+1)
    orbit=np.stack([env.axis_symmetry(goals,k) for k in range(6)])
    permitted=~old['state_mask'][orbit].any(0)&~np.isin(orbit,old['heldout_goals']).any(0)
    goals=goals[permitted];orbit=orbit[:,permitted];canonical=orbit.min(0)
    assignment={};folds={name:[] for name in ('train','development','test')}
    for depth in np.unique(env.depth[goals]):
        groups=np.unique(canonical[env.depth[goals]==depth]);rng.shuffle(groups)
        nv=round(len(groups)*opt['goal_development_fraction']);nt=round(len(groups)*opt['goal_test_fraction'])
        for i,group in enumerate(groups):assignment[int(group)]='development' if i<nv else 'test' if i<nv+nt else 'train'
    for goal,group in zip(goals,canonical):folds[assignment[int(group)]].append(int(goal))
    folds={k:np.asarray(v,dtype=np.int64) for k,v in folds.items()}
    # First freeze evaluation contexts using only the pre-existing restrictions.
    cohorts={'restoration':old['original']};used=original
    for name,group in (('relation_validation','train'),('goal_validation','development'),('relation_test','train'),('goal_test','test')):
        rows=sample_cohort(env,folds[group],opt['evaluation_per_group'],rng,original,used)
        cohorts[name]=rows
        used=CoverageGuard(env,keys_to_pairs(np.unique(np.r_[used.keys,closure_keys(env,rows)]),env.size),old['state_mask'],old['heldout_goals'])
    withheld=np.unique(np.r_[old['heldout_goals'],*[env.axis_symmetry(np.r_[folds['development'],folds['test']],k) for k in range(6)]])
    keys=np.unique(np.r_[old['excluded_relation_keys'],*[closure_keys(env,r) for r in cohorts.values()]])
    guard=CoverageGuard(env,keys_to_pairs(keys,env.size),old['state_mask'],withheld)
    pools={int(d):folds['train'][env.depth[folds['train']]==d] for d in np.unique(env.depth[folds['train']])}
    relative={int(d):np.flatnonzero(env.depth==d) for d in opt['local_source_depths']}
    near=training_tasks(env,pools,opt['near_decisions'],rng,guard,0,relative,opt['local_source_fraction'])
    restore=training_tasks(env,{0:np.array([0])},opt['restoration_decisions'],rng,guard,0)
    if guard.task_blocked(env,near).any() or guard.task_blocked(env,restore).any():raise ValueError('Protected training labels')
    np.save(args.out/'Noncanonical.npy',near);np.save(args.out/'Restoration.npy',restore)
    np.savez_compressed(args.out/'cohorts.npz',**cohorts,excluded_relation_keys=keys,state_mask=old['state_mask'],heldout_goals=withheld)
    atomic_json(args.out/'ready.json',dict(source_commit=args.source_commit,train_decisions=len(near),restore_decisions=len(restore),
        goal_counts={k:len(v) for k,v in folds.items()},goal_depths={k:hist(env.depth[v]) for k,v in folds.items()},
        goal_ids={k:v.tolist() for k,v in folds.items()},protected_training_decisions=0,oracle_used_only_for_labels=True,
        split='New-stage target orbits and complete query stars protected; inherited earlier exposure not fully audited'))


def train(cfg,args):
    opt=cfg['training'];torch.set_num_threads(opt['cpu_threads']);torch.manual_seed(opt['seed'])
    env=Cube2(args.oracle,metric='QTM_URF_6');saved=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    model=QMap(env.feature_dim,**saved['model_config'],cap=saved['cap']).to(args.device);model.load_state_dict(saved['model'])
    d=np.load(args.data/'cohorts.npz');monitors={n:d[n] for n in ('relation_validation','goal_validation','restoration')}
    args.out.mkdir(parents=True,exist_ok=False)
    atomic_json(args.out/'config.json',dict(contract=cfg,arm=args.arm,source_commit=args.source_commit,
        initial_checkpoint=str(args.checkpoint),data=str(args.data),optimizer_reset=True))
    reference={n:inspect(model,env,r,args.device,32) for n,r in monitors.items()}
    def objective(report):return sum(report[n]['rollout']['solved'] for n in ('relation_validation','goal_validation'))
    best=objective(reference);best_step=0;floor=reference['restoration']['rollout']['solved']-cfg['selection']['restoration_tolerance']
    shutil.copy2(args.checkpoint,args.out/'best.pt')
    initialize(model,cfg['arms'][args.arm]);optimizer=torch.optim.Adam(model.parameters(),lr=opt['learning_rate'])
    guard=CoverageGuard(env,keys_to_pairs(d['excluded_relation_keys'],env.size),d['state_mask'],d['heldout_goals'])
    roles={};batches={'Noncanonical':(opt['noncanonical_batch'],opt['noncanonical_weight']),
        'Restoration':(opt['restoration_batch'],1-opt['noncanonical_weight'])}
    for index,name in enumerate(batches):
        rows=np.load(args.data/(name+'.npy'))
        if guard.task_blocked(env,rows).any():raise ValueError('Protected complete star')
        full=fixed_batch(env,rows);features=torch.as_tensor(env.features(full['states']),device=args.device)
        roles[name]=dict(rows=rows,batch=full,features=features,pool=Pool(np.arange(len(rows))[:,None],opt['seed']+index))
    records=[];started=time.monotonic()
    for step in range(1,opt['steps']+1):
        optimizer.zero_grad(set_to_none=True);losses={}
        for name,(count,weight) in batches.items():
            role=roles[name];ids=role['pool'].draw(count).ravel();states,batch=subset_batch(role['batch'],ids)
            h,r=batch_losses(model,role['features'][states],batch,cfg)
            (weight*(h+opt['rank_weight']*r)).backward();losses[name]=[float(h.detach()),float(r.detach())]
        torch.nn.utils.clip_grad_norm_(model.parameters(),opt['clip_norm']);optimizer.step()
        if step%opt['evaluate_every'] and step!=opt['steps']:continue
        report={n:inspect(model,env,r,args.device,32) for n,r in monitors.items()}
        eligible=report['restoration']['rollout']['solved']>=floor
        payload={k:saved[k] for k in ('model_type','task','actions','model_config','cap')}
        payload.update(model=model.state_dict(),optimizer=optimizer.state_dict(),step=step,source_commit=args.source_commit,
            pools={n:v['pool'].state_dict() for n,v in roles.items()},config=cfg,arm=args.arm,
            torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all() if str(args.device).startswith('cuda') else [],
            initial_checkpoint=str(args.checkpoint),data_directory=str(args.data))
        atomic_save(payload,args.out/'last.pt')
        if eligible and objective(report)>best:best=objective(report);best_step=step;atomic_save(payload,args.out/'best.pt')
        record=dict(step=step,seconds=time.monotonic()-started,metrics=report,eligible=eligible,best_step=best_step,
            training_losses=losses,training_fit=fit_metrics(model,env,roles['Noncanonical']['rows'][:512],args.device),
            exposure={n:int(v['pool'].seen.sum()) for n,v in roles.items()})
        records.append(record);atomic_json(args.out/'progress.json',record)
        print(json.dumps(dict(step=step,solved={n:r['rollout']['solved'] for n,r in report.items()},best_step=best_step,seconds=record['seconds'])),flush=True)
    atomic_json(args.out/'selection.json',dict(best_step=best_step,objective=best,restoration_floor=floor,source_commit=args.source_commit))
    def tests(path):
        p=torch.load(path,map_location=args.device,weights_only=False);model.load_state_dict(p['model'])
        return {n:inspect(model,env,d[n],args.device,32) for n in ('relation_test','goal_test')}
    final_tests=tests(args.out/'best.pt');parent_tests=tests(args.checkpoint)
    atomic_json(args.out/'summary.json',dict(source_commit=args.source_commit,arm=args.arm,reference=reference,evaluations=records,
        best_step=best_step,test=final_tests,baseline_test=parent_tests,adopted=False))


def main():
    p=argparse.ArgumentParser(__doc__)
    for n in ('config','out','oracle'):p.add_argument('--'+n,type=Path,required=True)
    for n in ('source','data','checkpoint'):p.add_argument('--'+n,type=Path)
    p.add_argument('--prepare',action='store_true');p.add_argument('--arm',choices=['inherited','reset_head'])
    p.add_argument('--source-commit',required=True);p.add_argument('--device',default='cuda:0');a=p.parse_args()
    cfg=load_coverage_config(a.config)
    if a.prepare:prepare(cfg,a)
    else:train(cfg,a)


if __name__=='__main__':main()
