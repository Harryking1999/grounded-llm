"""Small source-root-disjoint controls for eight/twelve-placement goal transfer."""
import argparse
from collections import Counter
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from grounded_llm.core.files import atomic_json, atomic_save
from grounded_llm.core.qmap.model import soft_rank_loss
from grounded_llm.tasks.blocks import rules
from grounded_llm.tasks.blocks.construction import construct
from grounded_llm.tasks.blocks.model import BoardEncoder, board_bits
from grounded_llm.tasks.blocks.oracle import DistanceOracle, successors
from .blocks12_eval import trace
from .tree_supervision_eval import QScorer


def prepare(cfg, out, protected_data=None):
    out.mkdir(parents=True,exist_ok=False); opt=cfg['data']
    old=json.loads(Path(opt['excluded_roots']).read_text())
    if isinstance(old,dict): old=old.get('cases',old.get('roots'))
    excluded={int(r.get('initial',r.get('source',0))) for r in old}
    protected_count=0
    if protected_data is not None:
        prior=json.loads((protected_data/'roots.json').read_text())
        protected={int(r['initial']) for rows in prior.values() for r in rows}
        excluded.update(protected);protected_count=len(protected)
    counts={role:opt[role+'_roots_per_size'] for role in ('train','development','test')}
    roots={role:[] for role in counts}; queries={role:[] for role in counts}
    oracle=DistanceOracle(cache_limit=opt['oracle_cache_limit'],seconds=opt['oracle_seconds'])
    stars={}; source_sets={role:set() for role in counts}
    for size in opt['placements']:
        candidates=construct(sum(counts.values())+len(excluded),size,cfg['seed']+size)
        candidates=[r for r in candidates if int(r['initial']) not in excluded][:sum(counts.values())]
        cursor=0
        for role,count in counts.items():
            for root in candidates[cursor:cursor+count]:
                source=int(root['initial']);source_sets[role].add(source)
                roots[role].append(dict(root,placements=size))
                path=[source]
                for action in root['construction']:path.append(rules.apply(path[-1],action))
                goals=[('clear'+str(size),0)]
                goals.extend(('partial'+str(size),path[max(1,round(size*f))]) for f in opt['goal_fractions'])
                for group,goal in goals:
                    query=dict(source=str(source),goal=str(goal),group=group,distance=oracle.distance(source,goal))
                    queries[role].append(query)
                    if role!='train':continue
                    state=source
                    while state!=goal:
                        key=(state,goal)
                        if key not in stars:
                            children=[s for _,s in successors(state)]
                            truth=[oracle.distance(s,goal) for s in children]
                            stars[key]=dict(source=str(state),goal=str(goal),children=list(map(str,children)),
                                truth=truth,parent_truth=oracle.distance(state,goal),group=group)
                        star=stars[key]; d=oracle.distance(state,goal)
                        state=int(next(s for s,t in zip(star['children'],star['truth']) if t==d-1))
            cursor+=count
        excluded.update(int(r['initial']) for r in candidates)
        print(json.dumps(dict(prepared_placements=size,training_stars=len(stars))),flush=True)
    for left in source_sets:
        for right in source_sets:
            if left!=right and source_sets[left]&source_sets[right]:raise ValueError('Root split overlap')
    rows=list(stars.values()); states=sorted({int(s) for r in rows for s in [r['source'],r['goal'],*r['children']]})
    lookup={s:i for i,s in enumerate(states)}; width=max(len(r['children']) for r in rows)
    children=np.full((len(rows),width),-1,np.int64); truth=np.full((len(rows),width),-1,np.float32)
    for i,r in enumerate(rows):
        children[i,:len(r['children'])]=[lookup[int(s)] for s in r['children']]
        truth[i,:len(r['truth'])]=r['truth']
    np.savez_compressed(out/'data.npz',states=np.asarray(list(map(str,states))),
        parents=np.asarray([lookup[int(r['source'])] for r in rows]),
        goals=np.asarray([lookup[int(r['goal'])] for r in rows]),children=children,truth=truth,
        parent_truth=np.asarray([r['parent_truth'] for r in rows],np.float32),groups=np.asarray([r['group'] for r in rows]))
    atomic_json(out/'queries.json',queries);atomic_json(out/'roots.json',roots)
    atomic_json(out/'ready.json',dict(training_stars=len(rows),states=len(states),maximum_candidates=width,
        class_stars=dict(Counter(r['group'] for r in rows)),root_counts={k:len(v) for k,v in source_sets.items()},
        full_training_candidate_labels=int((children>=0).sum()),oracle_unknowns=0,
        excluded_prior_stage_roots=protected_count,
        split='Current-stage source roots disjoint; older/descendant exposure not excluded'))


def star_losses(model, features, data, ids, options):
    parents=data['parents'][ids];goals=data['goals'][ids];children=data['children'][ids]
    valid=children>=0; safe=children.clamp_min(0)
    states,inverse=torch.unique(torch.cat((parents,goals,safe.ravel())),return_inverse=True)
    q=model.encode(features[states]); n=len(ids); width=children.shape[1]
    source=q[inverse[:n]]; target=q[inverse[n:2*n]]
    child=q[inverse[2*n:].reshape(n,width)]
    parent_distance=model.distance(source,target); scores=model.distance(child,target[:,None])
    actual=data['truth'][ids]; proxy=options['unreachable_label']
    labels=torch.where(actual<0,proxy,actual)
    point=F.smooth_l1_loss(torch.cat((parent_distance,scores[valid])),
        torch.cat((data['parent_truth'][ids],labels[valid])))
    rank=soft_rank_loss(scores,labels,valid,options['temperature'])
    return point,rank


def evaluate(path, queries, device, options):
    scorer=QScorer(path,device)
    oracle=DistanceOracle(cache_limit=options['oracle_cache_limit'],seconds=options['oracle_seconds'])
    rows=[dict(r,**trace(scorer,int(r['source']),int(r['goal']),oracle)) for r in queries]
    groups={}
    for group in sorted({r['group'] for r in rows}):
        selected=[r for r in rows if r['group']==group];outcomes=Counter(r['outcome'] for r in selected)
        groups[group]=dict(count=len(selected),solved=outcomes['reached'],outcomes=dict(outcomes),
            success_rate=outcomes['reached']/len(selected))
    return dict(groups=groups,rows=rows)


def train(cfg,args):
    options=cfg['training']; torch.set_num_threads(options['cpu_threads']);torch.manual_seed(options['seed'])
    saved=torch.load(args.checkpoint,map_location=args.device,weights_only=False);spec=saved['config']['model']
    if saved['cap']!=13 or saved['metric']!='directed_sum' or saved.get('unreachable_label')!=19:
        raise ValueError('Expected the adopted clearing map and unchanged labels')
    model=BoardEncoder(spec['state_dim'],spec['encoder_hidden_dim'],saved['cap'],saved['metric']).to(args.device)
    model.load_state_dict(saved['model']);optimizer=torch.optim.Adam(model.parameters())
    optimizer.load_state_dict(saved['optimizer'])
    for group in optimizer.param_groups:group['lr']=options['learning_rate']
    with np.load(args.data/'data.npz') as d:
        groups=d['groups'];features=board_bits(d['states']).to(args.device)
        data={k:torch.as_tensor(d[k],device=args.device) for k in ('parents','goals','children','truth','parent_truth')}
    pools={name:torch.as_tensor(np.flatnonzero(groups==name),device=args.device) for name in cfg['arms'][args.arm]}
    queries=json.loads((args.data/'queries.json').read_text());args.out.mkdir(parents=True,exist_ok=False)
    atomic_json(args.out/'config.json',dict(contract=cfg,arm=args.arm,source_commit=args.source_commit,
        initial_checkpoint=str(args.checkpoint),data=str(args.data)))
    initial=evaluate(args.checkpoint,queries['development'],args.device,cfg['data']); records=[]
    best=float(np.mean([v['success_rate'] for v in initial['groups'].values()]));best_step=0
    atomic_save(saved,args.out/'best.pt');seen=torch.zeros(len(groups),dtype=torch.int64,device=args.device)
    started=time.monotonic()
    for step in range(1,options['steps']+1):
        parts=[pool[torch.randint(len(pool),(options['batch_size']//len(pools),),device=args.device)] for pool in pools.values()]
        ids=torch.cat(parts);seen.scatter_add_(0,ids,torch.ones_like(ids));optimizer.zero_grad(set_to_none=True)
        h,r=star_losses(model,features,data,ids,options)
        (h+options['listwise_weight']*r).backward();optimizer.step()
        if step%options['evaluate_every'] and step!=options['steps']:continue
        payload={k:saved[k] for k in ('model_type','metric','cap','unreachable_label')}
        payload.update(model=model.state_dict(),optimizer=optimizer.state_dict(),step=saved['step']+step,
            continuation_step=step,source_commit=args.source_commit,config=dict(model=spec,training=options),
            torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all() if str(args.device).startswith('cuda') else [],
            seen=seen.cpu(),initial_checkpoint=str(args.checkpoint),data_directory=str(args.data))
        atomic_save(payload,args.out/'last.pt')
        report=evaluate(args.out/'last.pt',queries['development'],args.device,cfg['data'])
        tolerance=cfg['selection']['clear_loss_tolerance_per_size']
        eligible=all(report['groups'][g]['solved']>=initial['groups'][g]['solved']-tolerance for g in ('clear8','clear12'))
        score=float(np.mean([v['success_rate'] for v in report['groups'].values()]))
        if eligible and score>best:best=score;best_step=step;atomic_save(payload,args.out/'best.pt')
        record=dict(step=step,seconds=time.monotonic()-started,groups=report['groups'],eligible=eligible,
            best_step=best_step,huber=float(h.detach()),rank=float(r.detach()),unique_stars=int((seen>0).sum()))
        records.append(record);atomic_json(args.out/'progress.json',record);print(json.dumps(record),flush=True)
    atomic_json(args.out/'selection.json',dict(best_step=best_step,objective=best,source_commit=args.source_commit))
    final=evaluate(args.out/'best.pt',queries['test'],args.device,cfg['data'])
    baseline=evaluate(args.checkpoint,queries['test'],args.device,cfg['data'])
    atomic_json(args.out/'summary.json',dict(source_commit=args.source_commit,arm=args.arm,initial_development=initial,
        evaluations=records,best_step=best_step,test=final,baseline_test=baseline,adopted=False))


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--config',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--prepare',action='store_true');p.add_argument('--data',type=Path);p.add_argument('--checkpoint',type=Path)
    p.add_argument('--protected-data',type=Path)
    p.add_argument('--arm',choices=['clear_control','balanced_goals']);p.add_argument('--source-commit',required=True)
    p.add_argument('--device',default='cuda:0');a=p.parse_args();cfg=json.loads(a.config.read_text())
    if 'parent_contract' in cfg:
        parent=json.loads(Path(cfg['parent_contract']).read_text())
        cfg={**parent,**cfg,**{key:{**parent[key],**cfg.get(key,{})} for key in ('data','training')}}
    if a.prepare:prepare(cfg,a.out,a.protected_data)
    else:train(cfg,a)


if __name__=='__main__':main()
