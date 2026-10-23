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
from .coverage_data import CoverageGuard,keys_to_pairs,coverage,unique_tasks,hist,load_coverage_config
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


def pair_loss(model,env,pairs,device):
    states=np.unique(pairs);q=model.encode(torch.as_tensor(env.features(states),device=device))
    a,b=np.searchsorted(states,pairs).T
    return F.smooth_l1_loss(model.distance(q[a],q[b]),torch.as_tensor(env.distance(*pairs.T),device=device,dtype=q.dtype))


class Pool:
    def __init__(self,rows,seed,grouped=False):
        self.rows=rows;self.rng=np.random.default_rng(seed);self.grouped=grouped
        self.order=ordering(rows,self.rng,grouped);self.cursor=0
        self.seen=np.zeros(len(rows),dtype=np.int32)

    def draw(self,count):
        remaining=len(self.order)-self.cursor
        if count<=remaining:
            ids=self.order[self.cursor:self.cursor+count];self.cursor+=count
        else:
            tail=self.order[self.cursor:];self.order=ordering(self.rows,self.rng,self.grouped)
            ids=np.r_[tail,self.order[:count-remaining]];self.cursor=count-remaining
        np.add.at(self.seen,ids,1)
        return self.rows[ids]

    def state_dict(self):
        return dict(order=self.order,cursor=self.cursor,seen=self.seen,rng=self.rng.bit_generator.state)


class GoalPool:
    """Match goal exposure; optionally balance exact BFS-equivalent layers."""
    def __init__(self,rows,env,seed,by_layer):
        self.rows=rows;self.rng=np.random.default_rng(seed);self.seen=np.zeros(len(rows),np.int32)
        self.goals=np.unique(rows[:,1]);self.rng.shuffle(self.goals);self.drawn=0
        distance=env.distance(*rows.T) if by_layer else np.zeros(len(rows),int)
        self.layers={};self.cursors={};self.orders={};self.layer_cursors={}
        for goal in self.goals:
            layers=np.unique(distance[rows[:,1]==goal]);self.rng.shuffle(layers)
            self.layers[int(goal)]=layers;self.layer_cursors[int(goal)]=0
            for depth in layers:
                key=(int(goal),int(depth));ids=np.flatnonzero((rows[:,1]==goal)&(distance==depth))
                self.rng.shuffle(ids);self.orders[key]=ids;self.cursors[key]=0

    def draw(self,count):
        selected=[]
        for _ in range(count):
            goal=int(self.goals[self.drawn%len(self.goals)]);self.drawn+=1
            layers=self.layers[goal];depth=int(layers[self.layer_cursors[goal]%len(layers)])
            self.layer_cursors[goal]+=1;key=(goal,depth);order=self.orders[key];cursor=self.cursors[key]
            if cursor==len(order):self.rng.shuffle(order);cursor=0
            selected.append(order[cursor]);self.cursors[key]=cursor+1
        ids=np.asarray(selected);np.add.at(self.seen,ids,1)
        return self.rows[ids]

    def state_dict(self):
        return dict(goals=self.goals,drawn=self.drawn,layers=self.layers,layer_cursors=self.layer_cursors,
            orders=self.orders,cursors=self.cursors,seen=self.seen,rng=self.rng.bit_generator.state)


def run_factorial(args):
    cfg=load_coverage_config(args.config);options=cfg['training']
    torch.set_num_threads(options['cpu_threads']);torch.manual_seed(cfg['seed'])
    env=Cube2(args.oracle,metric='QTM_URF_6');d=np.load(args.data/'cohorts.npz')
    saved=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    model=QMap(env.feature_dim,**saved['model_config'],cap=saved['cap']).to(args.device)
    model.load_state_dict(saved['model']);optimizer=torch.optim.Adam(model.parameters())
    optimizer.load_state_dict(saved['optimizer'])
    for group in optimizer.param_groups:group['lr']=options['learning_rate']
    specification=cfg['arms'][args.base_arm or args.arm]
    options={**options,**{key:specification[key] for key in ('updates','evaluate_every') if key in specification}}
    batches=dict(specification['batches'])
    if args.arm in ('hard','random_aux'):
        batches['AuxHard' if args.arm=='hard' else 'AuxRandom']=[cfg['hard_followup']['extra_batch'],cfg['hard_followup']['extra_weight']]
    if args.arm in ('geometry','geometry_uniform'):
        batches['PairAngular' if args.arm=='geometry' else 'PairUniform']=[cfg['geometry']['pair_batch'],cfg['geometry']['pair_weight']]
    names=sorted(batches);pools={}
    guard=CoverageGuard(env,keys_to_pairs(d['excluded_relation_keys'],env.size),d['state_mask'],d['heldout_goals'])
    for index,name in enumerate(names):
        filename={'AuxHard':'Hard','AuxRandom':'Original'}.get(name,name)
        rows=np.load(args.data/(filename+'.npy'))
        if name.startswith('Pair'):
            if guard.contains(rows).any():raise ValueError('Protected auxiliary pair')
        elif guard.task_blocked(env,rows).any():raise ValueError('Incomplete/protected ranking pool')
        # Role-specific fixed seeds keep the original, cross and replay streams
        # identical between their edge/no-edge and hard/random controls.
        role_seed={'Original':0,'Cross':1,'OriginalReplay':2,'AuxHard':3,'AuxRandom':3,'PairAngular':3,'PairUniform':3,
            'Micro':1,'Layered':1,'UniformLayers':1}[name]
        pools[name]=(GoalPool(rows,env,options['seed']+role_seed,name=='Layered')
            if name in ('Layered','UniformLayers') else Pool(rows,options['seed']+role_seed,name=='Cross'))
    edge_rng=np.random.default_rng(options['seed']+4)
    allowed=np.load(args.data/'edge_allowed.npy',mmap_mode='r')
    edge_order=edge_rng.permutation(env.size) if specification['edges'] else np.empty(0,dtype=np.int64)
    edge_cursor=0;edge_seen=np.zeros(env.size,dtype=bool);edge_passes=0
    out=args.out;out.mkdir(parents=True,exist_ok=False)
    atomic_json(out/'config.json',dict(study=cfg,arm=args.arm,base_arm=args.base_arm,source_commit=args.source_commit))
    monitors={'validation':d['validation'],'restoration':d['original']}
    initial={k:inspect(model,env,tasks,args.device,32) for k,tasks in monitors.items()}
    best=(initial['validation']['rollout']['solved'],initial['restoration']['rollout']['solved'])
    eligible=best[1]>=385 and not specification.get('diagnostic_only',False)
    best_update=0;shutil.copy2(args.checkpoint,out/'best.pt')
    result=dict(source_commit=args.source_commit,arm=args.arm,base_arm=args.base_arm,initial=initial,evaluations=[],
        best_eligible=eligible,best_update=0,best_validation_and_restoration=best,initial_step=saved['step'])
    atomic_json(out/'summary.json',result);started=time.monotonic()
    fit_tasks={n:p.rows[np.random.default_rng(cfg['seed']).choice(len(p.rows),min(512,len(p.rows)),replace=False)] for n,p in pools.items()}
    if cfg.get('layer_pilot'):
        from .goal_layers import layer_report
        role=next(n for n in names if n!='OriginalReplay')
        result['initial_layers']=layer_report(model,env,args.data,role,args.device)
    fixed_edge_parents=np.random.default_rng(cfg['seed']+1).choice(env.size,512,replace=False)
    with torch.no_grad():
        result['initial_train_monitor']={}
        for name,tasks in fit_tasks.items():
            if name.startswith('Pair'):
                result['initial_train_monitor'][name]=dict(huber=float(pair_loss(model,env,tasks,args.device)),rank=None)
                continue
            batch=fixed_batch(env,tasks)
            h,r=batch_losses(model,torch.as_tensor(env.features(batch['states']),device=args.device),batch,cfg)
            result['initial_train_monitor'][name]=dict(huber=float(h),rank=float(r))
        el,_=edge_loss(model,env,fixed_edge_parents,allowed,args.device)
        result['initial_train_monitor']['edge_huber']=float(el)
    with (out/'loss.jsonl').open('w') as log:
        for update in range(1,options['updates']+1):
            model.train();optimizer.zero_grad(set_to_none=True);terms={}
            for name in names:
                count,weight=batches[name];tasks=pools[name].draw(count)
                if name.startswith('Pair'):
                    h=pair_loss(model,env,tasks,args.device);(weight*h).backward();terms[name]=[float(h.detach()),None]
                    continue
                batch=fixed_batch(env,tasks)
                h,r=batch_losses(model,torch.as_tensor(env.features(batch['states']),device=args.device),batch,cfg)
                (weight*(h+options['rank_weight']*r)).backward();terms[name]=[float(h.detach()),float(r.detach())]
            edge_count=0
            if specification['edges']:
                parents,edge_order,edge_cursor,passed=take_cycle(edge_order,edge_cursor,options['edge_parent_batch'],edge_rng)
                edge_passes+=passed;el,edge_count=edge_loss(model,env,parents,allowed,args.device)
                if el is not None:(options['edge_loss_weight']*el).backward()
                edge_seen[parents]=True;terms['edge']=None if el is None else float(el.detach())
            torch.nn.utils.clip_grad_norm_(model.parameters(),options['clip_norm']);optimizer.step()
            log.write(json.dumps(dict(update=update,losses=terms,effective_edge_labels=edge_count))+'\n')
            if update%options['evaluate_every'] and update!=options['updates']:continue
            log.flush();metrics={k:inspect(model,env,tasks,args.device,32) for k,tasks in monitors.items()}
            fit={}
            with torch.no_grad():
                for name,tasks in fit_tasks.items():
                    if name.startswith('Pair'):
                        fit[name]=dict(huber=float(pair_loss(model,env,tasks,args.device)),rank=None)
                        continue
                    batch=fixed_batch(env,tasks)
                    h,r=batch_losses(model,torch.as_tensor(env.features(batch['states']),device=args.device),batch,cfg)
                    fit[name]=dict(huber=float(h),rank=float(r))
                el,_=edge_loss(model,env,fixed_edge_parents,allowed,args.device)
                fit['edge_huber']=float(el)
            exposed=unique_tasks(np.concatenate([p.rows[p.seen>0] for n,p in pools.items() if not n.startswith('Pair')]),env.size)
            record=dict(updates=update,metrics=metrics,train_monitor=fit,
                roles={n:dict(exposures=int(p.seen.sum()),label_kind='distance_only' if n.startswith('Pair') else 'complete_star',
                    unique_decisions_exposed=None if n.startswith('Pair') else int((p.seen>0).sum()),
                    unique_distance_pairs_exposed=int((p.seen>0).sum()) if n.startswith('Pair') else None,
                    pool_rows=len(p.rows),minimum_exposure=int(p.seen.min()),maximum_exposure=int(p.seen.max())) for n,p in pools.items()},
                unique_complete_decisions=len(exposed),unique_candidate_action_labels=len(exposed)*6,
                unique_parents=len(np.unique(exposed[:,0])),unique_goals=len(np.unique(exposed[:,1])),
                true_distance_unique_decisions=hist(env.distance(*exposed.T)),
                unique_directed_edges_exposed=int(allowed[edge_seen].sum()),
                allowed_directed_edges=int(allowed.sum()),edge_parent_passes=edge_passes)
            if cfg.get('layer_pilot'):
                record['layers']=layer_report(model,env,args.data,role,args.device,full=update==options['updates'])
            score=(metrics['validation']['rollout']['solved'],metrics['restoration']['rollout']['solved'])
            record['eligible']=score[1]>=385 and not specification.get('diagnostic_only',False)
            result['evaluations'].append(record)
            payload=dict(model_type=MODEL_TYPE,task=env.name,actions=list(env.actions),model_config=saved['model_config'],
                cap=saved['cap'],model=model.state_dict(),optimizer=optimizer.state_dict(),config=cfg,
                step=saved['step']+update,initial_step=saved['step'],updates=update,source_commit=args.source_commit,
                arm=args.arm,base_arm=args.base_arm,pools={n:p.state_dict() for n,p in pools.items()},
                edge_order=edge_order,edge_cursor=edge_cursor,edge_seen=edge_seen,edge_passes=edge_passes,
                edge_rng=edge_rng.bit_generator.state,torch_rng=torch.get_rng_state(),data_directory=str(args.data),
                initial_checkpoint=str(args.checkpoint))
            path=out/f'update_{update:06d}.pt';torch.save(payload,path)
            if record['eligible'] and (not eligible or score>best):
                shutil.copy2(path,out/'best.pt');best=score;best_update=update;eligible=True
            result.update(best_update=best_update,best_validation_and_restoration=best,best_eligible=eligible,
                seconds=time.monotonic()-started)
            atomic_json(out/'summary.json',result)
            print(json.dumps(dict(update=update,solved=score,best=best,seconds=time.monotonic()-started,
                unique_decisions=len(exposed),unique_edges=record['unique_directed_edges_exposed'])),flush=True)
    atomic_json(out/'actual_coverage.json',coverage(env,exposed))
    atomic_json(out/'complete.json',dict(updates=update,source_commit=args.source_commit,best_update=best_update,
        sealed_tests_not_opened=True,all_allowed_edges_exposed=bool(edge_seen.all()) if specification['edges'] else None,
        unique_directed_edges_exposed=int(allowed[edge_seen].sum())))


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
    p.add_argument('--version',choices=['V1','V2','V2_disjoint','V3'])
    p.add_argument('--arm',choices=['original','edges','cross','both','hard','random_aux','geometry','geometry_uniform',
        'micro','layers','uniform_layers'])
    p.add_argument('--base-arm',choices=['original','edges','cross','both'])
    p.add_argument('--source-commit',required=True);p.add_argument('--device',default='cuda')
    p.add_argument('--no-edges',action='store_true')
    args=p.parse_args()
    if args.arm:run_factorial(args)
    elif args.version:run(args)
    else:p.error('Specify version or factorial arm')


if __name__=='__main__':main()
