"""Fixed-model, fully exposed decision-pool comparisons; tests opened separately."""
import argparse
import json
from pathlib import Path
import shutil
import time

import numpy as np
import torch
from torch.nn import functional as F

from grounded_llm.core.files import atomic_json
from grounded_llm.tasks.cube2.environment import Cube2
from grounded_llm.tasks.cube2.model import MODEL_TYPE,QMap
from .coverage_data import CoverageGuard,keys_to_pairs
from .full_state_joint import fixed_batch,batch_losses
from .near_goal_continuation import inspect


def ordering(tasks,rng,grouped):
    if grouped:
        groups=np.arange(len(tasks)).reshape(-1,8)
        return groups[rng.permutation(len(groups))].ravel()
    return rng.permutation(len(tasks))


def take_cycle(order,cursor,count,rng):
    remaining=len(order)-cursor
    if count<=remaining:return order[cursor:cursor+count],order,cursor+count,0
    tail=order[cursor:];order=rng.permutation(len(order))
    return np.r_[tail,order[:count-remaining]],order,count-remaining,1


def edge_loss(model,env,parents,allowed,device):
    children=env.successors(parents);mask=allowed[parents]
    source=np.broadcast_to(parents[:,None],children.shape)[mask];target=children[mask]
    if not len(source):return None,0
    states=np.unique(np.r_[source,target])
    q=model.encode(torch.as_tensor(env.features(states),device=device))
    predicted=model.distance(q[np.searchsorted(states,source)],q[np.searchsorted(states,target)])
    return F.smooth_l1_loss(predicted,torch.ones_like(predicted)),len(source)


def run(args):
    cfg=json.loads(args.config.read_text());options=cfg['training']
    torch.set_num_threads(options['cpu_threads']);torch.manual_seed(cfg['seed'])
    env=Cube2(args.oracle,metric='QTM_URF_6')
    saved=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    model=QMap(env.feature_dim,**saved['model_config'],cap=saved['cap']).to(args.device)
    model.load_state_dict(saved['model'])
    optimizer=torch.optim.Adam(model.parameters(),lr=options['learning_rate'])
    optimizer.load_state_dict(saved['optimizer'])
    for group in optimizer.param_groups:group['lr']=options['learning_rate']
    tasks=np.load(args.data/(args.version+'.npy'));replay=np.load(args.data/'restoration.npy')
    allowed=np.load(args.data/'edge_allowed.npy',mmap_mode='r')
    d=np.load(args.data/'cohorts.npz');guard=CoverageGuard(env,keys_to_pairs(d['excluded_relation_keys'],env.size),
        d['state_mask'],d['heldout_goals'])
    if guard.task_blocked(env,tasks).any():raise ValueError('Train decision leaks protected labels')
    out=args.out;out.mkdir(parents=True,exist_ok=False)
    atomic_json(out/'config.json',dict(study=cfg,version=args.version,initial_checkpoint=str(args.checkpoint),
        source_commit=args.source_commit,no_edges=args.no_edges))
    rng=np.random.default_rng(options['seed']);edge_rng=np.random.default_rng(options['seed']+1)
    restoration_rng=np.random.default_rng(options['seed']+2)
    edge_order=edge_rng.permutation(env.size);edge_cursor=0;edge_passes=0
    rest_order=restoration_rng.permutation(len(replay));rest_cursor=0
    seen=np.zeros(len(tasks),dtype=np.int32);rest_seen=np.zeros(len(replay),dtype=np.int32)
    edge_seen=np.zeros(env.size,dtype=bool);updates=0
    monitors={'validation':d['validation'],'restoration':d['original']}
    initial={name:inspect(model,env,rows,args.device,32) for name,rows in monitors.items()}
    result=dict(source_commit=args.source_commit,version=args.version,initial_step=saved['step'],
        initial=initial,evaluations=[],pool_rows=len(tasks),original_checkpoint=str(args.checkpoint),
        architecture=saved['model_config'],no_edges=args.no_edges)
    best=(initial['validation']['rollout']['solved'],initial['restoration']['rollout']['solved'])
    best_eligible=best[1]>=385
    shutil.copy2(args.checkpoint,out/'best.pt');best_update=0
    # Fixed training-only fitting monitors distinguish exposure from learning.
    fixed_ids=np.random.default_rng(cfg['seed']+5).choice(len(tasks),min(512,len(tasks)),replace=False)
    fixed_tasks=tasks[fixed_ids]
    atomic_json(out/'summary.json',result)
    started=time.monotonic()
    with (out/'loss.jsonl').open('w') as log:
        for epoch in range(1,options['epochs']+1):
            model.train()
            order=ordering(tasks,rng,args.version!='V2_disjoint')
            losses=[]
            for begin in range(0,len(order),options['near_batch']):
                ids=order[begin:begin+options['near_batch']];near=tasks[ids]
                count=options['restoration_batch']
                rid,rest_order,rest_cursor,_=take_cycle(rest_order,rest_cursor,count,restoration_rng)
                restore=replay[rid]
                optimizer.zero_grad(set_to_none=True);terms=[]
                for rows,weight in ((near,options['near_weight']),(restore,1-options['near_weight'])):
                    batch=fixed_batch(env,rows)
                    h,r=batch_losses(model,torch.as_tensor(env.features(batch['states']),device=args.device),batch,cfg)
                    (weight*(h+options['rank_weight']*r)).backward()
                    terms.extend([float(h.detach()),float(r.detach())])
                el=None;edge_labels=0
                if not args.no_edges:
                    count=options['edge_parent_batch']
                    parents,edge_order,edge_cursor,passed=take_cycle(edge_order,edge_cursor,count,edge_rng)
                    edge_passes+=passed
                    el,edge_labels=edge_loss(model,env,parents,allowed,args.device)
                    if el is not None:(options['edge_loss_weight']*el).backward()
                    edge_seen[parents]=True
                torch.nn.utils.clip_grad_norm_(model.parameters(),options['clip_norm']);optimizer.step()
                updates+=1;seen[ids]+=1;rest_seen[rid]+=1
                row=dict(update=updates,epoch=epoch,near_huber=terms[0],near_rank=terms[1],
                    restore_huber=terms[2],restore_rank=terms[3],edge_huber=None if el is None else float(el.detach()),
                    edge_labels=edge_labels)
                losses.append(row);log.write(json.dumps(row)+'\n')
            log.flush()
            atomic_json(out/'progress.json',dict(epoch=epoch,updates=updates,seconds=time.monotonic()-started,
                unique_decisions_exposed=int((seen>0).sum()),min_decision_exposure=int(seen.min()),
                mean_losses={key:float(np.mean([r[key] for r in losses])) for key in
                             ('near_huber','near_rank','restore_huber','restore_rank')}))
            print(json.dumps(dict(epoch=epoch,updates=updates,seconds=time.monotonic()-started)),flush=True)
            if (epoch%options['evaluate_every_epochs'] and epoch!=options['epochs']
                    and updates not in options['matched_update_checkpoints']):continue
            metrics={name:inspect(model,env,rows,args.device,32) for name,rows in monitors.items()}
            model.eval();fit=fixed_batch(env,fixed_tasks)
            with torch.no_grad():
                h,r=batch_losses(model,torch.as_tensor(env.features(fit['states']),device=args.device),fit,cfg)
            fit_report=dict(huber=float(h),rank=float(r))
            score=(metrics['validation']['rollout']['solved'],metrics['restoration']['rollout']['solved'])
            eligible=score[1]>=385
            exposed_edges=int(allowed[edge_seen].sum()) if not args.no_edges else 0
            record=dict(epoch=epoch,updates=updates,metrics=metrics,train_monitor=fit_report,
                unique_decisions_exposed=int((seen>0).sum()),min_decision_exposure=int(seen.min()),
                unique_restore_exposed=int((rest_seen>0).sum()),unique_directed_edges_exposed=exposed_edges,
                directed_edge_coverage=exposed_edges/(env.size*len(env.actions)),eligible=eligible)
            result['evaluations'].append(record)
            payload=dict(model_type=MODEL_TYPE,task=env.name,actions=list(env.actions),model_config=saved['model_config'],
                cap=saved['cap'],model=model.state_dict(),optimizer=optimizer.state_dict(),source_commit=args.source_commit,
                step=saved['step']+updates,initial_step=saved['step'],updates=updates,epoch=epoch,config=cfg,version=args.version,
                rng=rng.bit_generator.state,edge_rng=edge_rng.bit_generator.state,restoration_rng=restoration_rng.bit_generator.state,
                edge_order=edge_order,edge_cursor=edge_cursor,edge_passes=edge_passes,rest_order=rest_order,rest_cursor=rest_cursor,
                seen=seen,rest_seen=rest_seen,edge_seen=edge_seen,torch_rng=torch.get_rng_state(),
                data_directory=str(args.data),initial_checkpoint=str(args.checkpoint))
            path=out/f'epoch_{epoch:03d}.pt';torch.save(payload,path)
            if eligible and (not best_eligible or score>best):
                shutil.copy2(path,out/'best.pt');best=score;best_update=updates;best_eligible=True
            result.update(best_update=best_update,best_validation_and_restoration=best,best_eligible=best_eligible,
                          seconds=time.monotonic()-started)
            atomic_json(out/'summary.json',result)
            print(json.dumps(dict(stage='validation',epoch=epoch,solved=score,best=best)),flush=True)
    atomic_json(out/'complete.json',dict(updates=updates,epochs=epoch,best_update=best_update,source_commit=args.source_commit,
        sealed_tests_not_opened=True,maximum_decision_exposure=int(seen.max()),minimum_decision_exposure=int(seen.min())))


def main():
    p=argparse.ArgumentParser(__doc__)
    for name in ('config','data','oracle','checkpoint','out'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--version',choices=['V1','V2','V2_disjoint','V3'],required=True)
    p.add_argument('--source-commit',required=True);p.add_argument('--device',default='cuda')
    p.add_argument('--no-edges',action='store_true')
    run(p.parse_args())


if __name__=='__main__':main()
