"""Task-specific decision coverage, historical exposure and paired-goal pools."""
import argparse
from collections import Counter
import json
from pathlib import Path
import shutil

import numpy as np
import torch

from grounded_llm.core.files import atomic_json
from grounded_llm.tasks.cube2.environment import Cube2
from grounded_llm.tasks.cube2.model import QMap
from .full_state_joint import RelationGuard, task_pairs
from .near_goal_continuation import ProtectedRelations, training_tasks, axis_pairs


def hist(values):
    unique,counts=np.unique(values,return_counts=True)
    return {str(int(k)):int(v) for k,v in zip(unique,counts)}


def unique_tasks(tasks, size):
    keys=np.unique(tasks[:,0].astype(np.int64)*size+tasks[:,1])
    return np.column_stack((keys//size,keys%size))


def model_from(path,env,device):
    saved=torch.load(path,map_location='cpu',weights_only=False)
    model=QMap(env.feature_dim,**saved['model_config'],cap=saved['cap']).to(device)
    model.load_state_dict(saved['model']);model.eval()
    return model


@torch.no_grad()
def predicted_scores(model,env,tasks,device):
    rows=[]
    for begin in range(0,len(tasks),1024):
        chunk=tasks[begin:begin+1024];children=env.successors(chunk[:,0])
        states=np.unique(np.r_[children.ravel(),chunk[:,1]])
        q=model.encode(torch.as_tensor(env.features(states),device=device))
        rows.append(model.distance(q[np.searchsorted(states,children)],
                    q[np.searchsorted(states,chunk[:,1]),None]).cpu().numpy())
    return np.concatenate(rows)


def coverage(env,tasks):
    unique=unique_tasks(tasks,env.size)
    starts,counts=np.unique(unique[:,0],return_counts=True)
    output=dict(exposures=len(tasks),unique_goals=len(np.unique(unique[:,1])),
        unique_parents=len(starts),unique_state_goal_pairs=len(unique),
        unique_complete_decisions=len(unique),candidate_labels_exposed=len(tasks)*len(env.actions),
        unique_state_goal_action_labels=len(unique)*len(env.actions),parent_goals_histogram=hist(counts),
        goal_depth_exposures=hist(env.depth[tasks[:,1]]),
        parent_depth_unique_decisions=hist(env.depth[unique[:,0]]),
        true_distance_unique_decisions=hist(env.distance(*unique.T)),
        per_goal={})
    masks=[];unique_candidate_labels=0
    for goal in np.unique(unique[:,1]):
        rows=unique[unique[:,1]==goal];children=env.successors(rows[:,0])
        truth=env.distance(children,goal);d=env.distance(rows[:,0],goal)
        unique_candidate_labels+=len(np.unique(children))
        optimal=truth==d[:,None]-1
        masks.extend((optimal*(1<<np.arange(len(env.actions)))).sum(-1).tolist())
        region=(rows[:,0]//729*8//5040)*9+(rows[:,0]%729*9//729)
        output['per_goal'][str(int(goal))]=dict(depth=int(env.depth[goal]),
            unique_parents=len(rows),exposures=int((tasks[:,1]==goal).sum()),
            distances=hist(d),parent_depths=hist(env.depth[rows[:,0]]),
            permutation_twist_regions=hist(region),
            optimal_action_counts=optimal.sum(0).tolist(),
            best_action_ties=hist(optimal.sum(-1)),
            strict_order_constraints=int((truth[:,:,None]<truth[:,None,:]).sum()))
    output['unique_candidate_labels']=unique_candidate_labels
    output['optimal_action_mask_histogram']=hist(masks)
    # Count parents with disjoint optimal sets across targets: start-only action
    # cannot satisfy these decisions simultaneously.
    masks=np.asarray(masks) # recompute in original unique order below
    switch=0;multi=0
    for begin in range(0,len(unique),8192):
        part=unique[begin:begin+8192];child=env.successors(part[:,0])
        truth=env.distance(child,part[:,1,None]);d=env.distance(*part.T)
        masks_part=((truth==d[:,None]-1)*(1<<np.arange(len(env.actions)))).sum(-1)
        if begin==0:all_masks=masks_part
        else:all_masks=np.r_[all_masks,masks_part]
    boundaries=np.r_[0,np.flatnonzero(np.diff(unique[:,0]))+1,len(unique)]
    for first,last in zip(boundaries[:-1],boundaries[1:]):
        if last-first>1:
            multi+=1
            switch+=int(np.bitwise_and.reduce(all_masks[first:last])==0)
    output.update(multi_goal_parents=multi,parents_requiring_goal_dependent_action=switch)
    return output


def reconstruct(env,first_cohorts,second_cohorts,checkpoint1000,out):
    pools={d:np.flatnonzero(env.depth==d) for d in (1,2,3)}
    near=[];restore=[]
    rng=np.random.default_rng(2026102303)
    for phase,path in enumerate((first_cohorts,second_cohorts)):
        d=np.load(path);keys=d['excluded_relation_keys']
        if phase==0:
            # Original run stored protected keys as reconstructed later; its
            # npz has the same schema and historical mask.
            pass
        guard=ProtectedRelations(env,np.column_stack((keys//env.size,keys%env.size)),d['state_mask'])
        if phase:
            saved=torch.load(checkpoint1000,map_location='cpu',weights_only=False)
            assert rng.bit_generator.state==saved['rng'],'Historical RNG replay differs'
        for step in range(phase*1000+1,(phase+1)*1000+1):
            near.append(training_tasks(env,pools,256,rng,guard,(step-1)*256))
            restore.append(training_tasks(env,{0:np.array([0])},128,rng,guard,0))
    tasks=np.concatenate([*near,*restore])
    np.savez_compressed(out/'historical_tasks.npz',near=np.concatenate(near),restoration=np.concatenate(restore))
    atomic_json(out/'historical_coverage.json',coverage(env,tasks))
    return tasks


def goal_orbits(env):
    goals=np.flatnonzero((env.depth>=1)&(env.depth<=3))
    canonical=np.min(np.stack([env.axis_symmetry(goals,k) for k in range(6)]),axis=0)
    return goals,canonical


class CoverageGuard(ProtectedRelations):
    def __init__(self,env,pairs,state_mask,withheld):
        super().__init__(env,pairs,state_mask)
        self.withheld=np.asarray(withheld,dtype=np.int64)

    def contains(self,pairs):
        return super().contains(pairs)|np.isin(pairs,self.withheld).any(-1)


def closure_keys(env,tasks):
    pairs=axis_pairs(env,task_pairs(env,tasks))
    return np.unique(np.minimum(*pairs.T)*env.size+np.maximum(*pairs.T))


def keys_to_pairs(keys,size):
    return np.column_stack((keys//size,keys%size))


def sample_cohort(env,goals,count,rng,guard,previous):
    depths=sorted(np.unique(env.depth[goals]));rows={int(d):[] for d in depths};used=set()
    quotas={int(d):count//len(depths)+int(i<count%len(depths)) for i,d in enumerate(depths)}
    while sum(map(len,rows.values()))<count:
        starts=rng.integers(env.size,size=2048)
        targets=rng.choice(goals,size=len(starts))
        tasks=np.column_stack((starts,targets))
        keep=(starts!=targets)&~guard.task_blocked(env,tasks)&~previous.task_blocked(env,tasks)
        for s,g in tasks[keep]:
            depth=int(env.depth[g])
            if len(rows[depth])<quotas[depth] and (int(s),int(g)) not in used:
                rows[depth].append((int(s),int(g)));used.add((int(s),int(g)))
    return np.asarray([row for depth in depths for row in rows[int(depth)]],dtype=np.int64)


@torch.no_grad()
def mine_trajectories(model,env,roots,guard,device,known):
    current=roots[:,0].copy();goals=roots[:,1];seen=[{int(s)} for s in current]
    active=current!=goals;first_wrong=np.full(len(roots),-1);cycle_step=np.full(len(roots),-1)
    rows=[];details=[]
    for step in range(32):
        ids=np.flatnonzero(active)
        if not len(ids):break
        tasks=np.column_stack((current[ids],goals[ids]));children=env.successors(tasks[:,0])
        scores=predicted_scores(model,env,tasks,device);chosen=scores.argmin(-1)
        truth=env.distance(children,tasks[:,1,None]);before=env.distance(*tasks.T)
        wrong=truth[np.arange(len(ids)),chosen]!=before-1
        margin=np.sort(scores,axis=1)[:,1]-scores.min(-1)
        allowed=~guard.task_blocked(env,tasks)
        for index,trajectory in enumerate(ids):
            if wrong[index] and first_wrong[trajectory]<0:first_wrong[trajectory]=step
            if allowed[index]:
                s,g=map(int,tasks[index]);key=s*env.size+g
                rows.append((s,g));details.append((int(trajectory),step,bool(wrong[index]),float(margin[index]),key in known))
            child=int(children[index,chosen[index]])
            current[trajectory]=child
            if child==goals[trajectory]:active[trajectory]=False
            elif child in seen[trajectory]:
                cycle_step[trajectory]=step+1;active[trajectory]=False
            else:seen[trajectory].add(child)
    return np.asarray(rows,dtype=np.int64),dict(trajectories=len(roots),
        solved=int((current==goals).sum()),first_wrong=hist(first_wrong),cycle_entry=hist(cycle_step),
        per_decision=np.asarray(details,dtype=float),final_states=current,roots=roots)


def prepare(args):
    cfg=json.loads(args.config.read_text());torch.set_num_threads(2)
    out=args.out;out.mkdir(parents=True,exist_ok=False)
    env=Cube2(args.oracle,metric='QTM_URF_6');rng=np.random.default_rng(cfg['seed'])
    if args.audit_source:
        for name in ('historical_tasks.npz','historical_coverage.json','historical_label_keys.npy'):
            shutil.copy2(args.audit_source/name,out/name)
        old_tasks=np.load(out/'historical_tasks.npz')
        historical=np.r_[old_tasks['near'],old_tasks['restoration']]
        atomic_json(out/'historical_coverage.json',coverage(env,historical))
    else:
        historical=reconstruct(env,args.first_cohorts,args.second_cohorts,args.checkpoint1000,out)
    old=np.load(args.second_cohorts);oldkeys=old['excluded_relation_keys'];mask=old['state_mask']
    goals,canonical=goal_orbits(env);reserved=[]
    for depth in (2,3):
        orbits=np.unique(canonical[env.depth[goals]==depth]);rng.shuffle(orbits)
        reserved.extend(orbits[:max(1,round(len(orbits)*cfg['data']['withheld_goal_orbit_fraction']))])
    heldout_goals=goals[np.isin(canonical,reserved)];train_goals=goals[~np.isin(canonical,reserved)]
    atomic_json(out/'goal_inventory.json',dict(all_goals=goals.tolist(),canonical=canonical.tolist(),
        train_goals=train_goals.tolist(),heldout_goals=heldout_goals.tolist(),
        by_depth={str(d):dict(goals=int((env.depth[goals]==d).sum()),
            orbits=len(np.unique(canonical[env.depth[goals]==d])),train=int((env.depth[train_goals]==d).sum())) for d in (1,2,3)},
        limitation='One orbit at depth 1; unseen-goal testing is possible only at depth 2/3. Heldout goals may have inherited pretraining exposure.'))
    # Reconstructed historical labels exclude fresh initial decision labels.
    if args.audit_source:
        previous_keys=np.load(out/'historical_label_keys.npy')
    else:
        historical_keys=[]
        unique=unique_tasks(historical,env.size)
        for start in range(0,len(unique),16384):historical_keys.append(closure_keys(env,unique[start:start+16384]))
        previous_keys=np.unique(np.concatenate(historical_keys))
        np.save(out/'historical_label_keys.npy',previous_keys)
    guard=ProtectedRelations(env,keys_to_pairs(oldkeys,env.size),mask)
    previous=RelationGuard(env.size,keys_to_pairs(previous_keys,env.size))
    cohorts={}
    for name,pool in [('validation',train_goals),('test',train_goals),('novel_goal_test',heldout_goals)]:
        tasks=sample_cohort(env,pool,cfg['data'][name+'_tasks'],rng,guard,previous)
        cohorts[name]=tasks
        oldkeys=np.unique(np.r_[oldkeys,closure_keys(env,tasks)])
        guard=ProtectedRelations(env,keys_to_pairs(oldkeys,env.size),mask)
    np.savez_compressed(out/'cohorts.npz',**cohorts,state_mask=mask,excluded_relation_keys=oldkeys,
        heldout_goals=heldout_goals,train_goals=train_goals,original=old['original'])
    guard=CoverageGuard(env,keys_to_pairs(oldkeys,env.size),mask,heldout_goals)
    # One-step labels use the same endpoint/axis exclusions. Parents may have
    # partially protected edges: record the exact mask, never fabricate labels.
    edge_allowed=np.zeros((env.size,len(env.actions)),dtype=bool)
    for begin in range(0,env.size,32768):
        ids=np.arange(begin,min(begin+32768,env.size));children=env.successors(ids)
        pairs=np.stack((np.broadcast_to(ids[:,None],children.shape),children),axis=-1)
        edge_allowed[ids]=~guard.contains(pairs)
    np.save(out/'edge_allowed.npy',edge_allowed)
    atomic_json(out/'edge_inventory.json',dict(states=env.size,legal_directed_edges=env.size*len(env.actions),
        allowed_directed_edges=int(edge_allowed.sum()),allowed_fraction=float(edge_allowed.mean()),
        allowed_parents=int(edge_allowed.any(-1).sum()),
        complete_six_edge_parents=int(edge_allowed.all(-1).sum()),
        protected_labels=int((~edge_allowed).sum()),exclusion='historical state mask plus evaluation relation/axis closure and reserved goal endpoints'))
    # Absolute physical parents only; BFS near solved plus depth-stratified,
    # uniformly random states. No relative-state features or inference change.
    candidates=[]
    for depth in range(15):
        layer=np.flatnonzero(env.depth==depth)
        count=min(len(layer),cfg['data']['candidate_parents']//15)
        candidates.extend(rng.choice(layer,count,replace=False).tolist())
    candidates=np.unique(np.r_[candidates,rng.integers(env.size,size=cfg['data']['candidate_parents'])])
    rng.shuffle(candidates)
    model=model_from(args.checkpoint2000,env,args.device)
    groups=[];group_meta=[]
    allgoal=np.r_[0,train_goals]
    target_counts=Counter();category_counts=Counter()
    for parent in candidates:
        tasks=np.column_stack((np.full(len(train_goals),parent),train_goals))
        allowed=~guard.task_blocked(env,tasks)
        eligible=train_goals[allowed & (train_goals!=parent)]
        if len(eligible)<cfg['data']['goals_per_parent']:continue
        child=env.successors(int(parent));truth=env.distance(child[None],eligible[:,None])
        distance=env.distance(parent,eligible);good=truth==distance[:,None]-1
        actionmask=(good*(1<<np.arange(6))).sum(-1)
        # Cycle through desired parent/goal distances, preserve exhaustive goal
        # inventory rather than redrawing a few popular endpoints.
        selected=[]
        for position in range(cfg['data']['goals_per_parent']):
            depth=1+(len(groups)+position)%3
            indices=[i for i,g in enumerate(eligible) if env.depth[g]==depth and int(g) not in selected]
            if not indices:indices=[i for i,g in enumerate(eligible) if int(g) not in selected]
            order=sorted(indices,key=lambda i:(target_counts[int(eligible[i])],
                category_counts[(int(env.depth[parent]),int(distance[i]))],float(rng.random())))
            # Prefer incompatible optimal masks after broad target balancing.
            if selected:
                intersection=np.bitwise_and.reduce(actionmask[np.isin(eligible,selected)])
                switched=[i for i in order if not (int(actionmask[i]) & int(intersection))]
                minimum=target_counts[int(eligible[order[0]])]
                switched=[i for i in switched if target_counts[int(eligible[i])]<=minimum+2]
                if switched:order=switched+order
            i=order[0];goal=int(eligible[i]);selected.append(goal)
            target_counts[goal]+=1;category_counts[(int(env.depth[parent]),int(distance[i]))]+=1
        groups.append(np.column_stack((np.full(len(selected),parent),selected)))
        if len(groups)>=cfg['data']['v2_parents']:break
    if len(groups)!=cfg['data']['v2_parents']:raise ValueError('Insufficient admissible paired parents')
    paired=np.concatenate(groups)
    v1=paired[:cfg['data']['v1_parents']*cfg['data']['goals_per_parent']]
    # Disjoint control retains exact goal identities/marginals and (s,g)
    # distance/solved-depth bins, while replacing each repeated parent.
    disjoint=[];used=set();buckets={};fallbacks=0
    # Cache physical parents by the two exact distance strata. Tiny solved
    # layers cannot supply infinitely many disjoint parents; disclose reuse
    # instead of spending thousands of rejection attempts on impossible bins.
    control_candidates=np.unique(np.r_[candidates,rng.integers(env.size,size=262144)])
    for goal in np.unique(paired[:,1]):
        rows=np.column_stack((control_candidates,np.full(len(control_candidates),goal)))
        keep=(control_candidates!=goal)&~guard.task_blocked(env,rows)
        parents=control_candidates[keep];distances=env.distance(parents,goal)
        bins=env.depth[parents].astype(int)*16+distances
        for key in np.unique(bins):
            pool=parents[bins==key].copy();rng.shuffle(pool)
            buckets[(int(goal),int(key))]=list(map(int,pool))
    for row in paired:
        parent,goal=map(int,row);depth=int(env.depth[parent]);distance=int(env.distance(parent,goal))
        bucket=buckets.get((goal,depth*16+distance),[])
        while bucket and bucket[-1] in used:bucket.pop()
        if bucket:
            source=bucket.pop();used.add(source);disjoint.append((source,goal))
        else:
            disjoint.append((parent,goal));fallbacks+=1
    disjoint=np.asarray(disjoint,dtype=np.int64)
    atomic_json(out/'disjoint_matching.json',dict(rows=len(disjoint),fallback_reused_parents=fallbacks,
        exact_matched_parent_depth=True,exact_matched_goal_distance=True,
        unique_parents=len(np.unique(disjoint[:,0]))))
    # Common restoration replay pool, including paired-goal training parents
    # where solved-goal supervision is allowed. The pool is shared by all arms.
    replay_parents=np.unique(paired[:,0])
    replay_tasks=np.column_stack((replay_parents,np.zeros(len(replay_parents),dtype=int)))
    replay_tasks=replay_tasks[(replay_parents!=0)&~guard.task_blocked(env,replay_tasks)]
    rest=training_tasks(env,{0:np.array([0])},32768,rng,guard,0)
    replay_tasks=unique_tasks(np.r_[replay_tasks,rest],env.size)
    np.save(out/'restoration.npy',replay_tasks)
    atomic_json(out/'restoration_coverage.json',coverage(env,replay_tasks))
    # Real greedy failures only from the admissible training root pool.
    mine_roots=[]
    for goal in train_goals:
        pool=paired[paired[:,1]==goal]
        indices=rng.choice(len(pool),min(len(pool),cfg['data']['mining_trajectories_per_goal']),replace=False)
        mine_roots.extend(pool[indices].tolist())
    known=set((paired[:,0]*env.size+paired[:,1]).tolist())
    mined,trace=mine_trajectories(model,env,np.asarray(mine_roots),guard,args.device,known)
    detail=trace.pop('per_decision');np.savez_compressed(out/'training_trajectories.npz',decisions=mined,detail=detail,
        roots=trace.pop('roots'),final_states=trace.pop('final_states'))
    atomic_json(out/'trajectory_audit.json',trace)
    # Add novel trajectory states plus wrong/ambiguous training-pool candidates.
    extra=unique_tasks(mined,env.size)
    extra=extra[~np.isin(extra[:,0]*env.size+extra[:,1],list(known))]
    candidates_hard=np.column_stack((rng.choice(candidates,131072),rng.choice(train_goals,131072)))
    candidates_hard=candidates_hard[(candidates_hard[:,0]!=candidates_hard[:,1])&~guard.task_blocked(env,candidates_hard)]
    scores=predicted_scores(model,env,candidates_hard,args.device)
    truth=env.distance(env.successors(candidates_hard[:,0]),candidates_hard[:,1,None])
    before=env.distance(*candidates_hard.T);wrong=truth[np.arange(len(truth)),scores.argmin(-1)]!=before-1
    gap=np.sort(scores,axis=1)[:,1]-scores.min(-1)
    difficult=candidates_hard[wrong|(gap<cfg['data']['hard_margin'])]
    # Keep actual novel trajectory decisions first, then fill remaining slots
    # with novel wrong/close-margin decisions from the same training pool.
    rng.shuffle(extra);extra=extra[:cfg['data']['v3_extra_decisions']]
    trajectory_extra=len(extra)
    difficult=unique_tasks(difficult,env.size)
    existing=np.r_[list(known),extra[:,0]*env.size+extra[:,1]]
    difficult=difficult[~np.isin(difficult[:,0]*env.size+difficult[:,1],existing)]
    rng.shuffle(difficult)
    extra=np.r_[extra,difficult[:cfg['data']['v3_extra_decisions']-len(extra)]]
    if len(extra)!=cfg['data']['v3_extra_decisions']:raise ValueError('Insufficient novel hard decisions')
    atomic_json(out/'hard_pool_sources.json',dict(novel_actual_trajectory_decisions=trajectory_extra,
        novel_training_wrong_or_close_decisions=len(extra)-trajectory_extra))
    versions=dict(V1=v1,V2=paired,V2_disjoint=disjoint,V3=np.r_[paired,extra])
    reports={}
    for name,tasks in versions.items():
        if guard.task_blocked(env,tasks).any():raise ValueError('Protected task in pool')
        np.save(out/(name+'.npy'),tasks)
        reports[name]=coverage(env,tasks)
        atomic_json(out/(name+'_coverage.json'),reports[name])
    np.savez_compressed(out/'prepared.json.npz',version_names=list(versions),train_goals=train_goals)
    atomic_json(out/'ready.json',dict(versions={n:dict(rows=len(t),unique=len(unique_tasks(t,env.size)))
        for n,t in versions.items()},training_checkpoint=str(args.checkpoint2000),source_commit=args.source_commit,
        architecture='unchanged',sealed_tests=['test','novel_goal_test','old small_ood'],
        historical_rng_reconstructed=True))
    print(json.dumps({'stage':'ready','versions':{n:len(t) for n,t in versions.items()}}),flush=True)


def expand_cross(args):
    cfg=json.loads(args.config.read_text());env=Cube2(args.oracle,metric='QTM_URF_6')
    rng=np.random.default_rng(cfg['seed']);source=args.audit_source;out=args.out
    out.mkdir(parents=True,exist_ok=False);d=np.load(source/'cohorts.npz')
    guard=CoverageGuard(env,keys_to_pairs(d['excluded_relation_keys'],env.size),d['state_mask'],d['heldout_goals'])
    for name in ('cohorts.npz','edge_allowed.npy','edge_inventory.json','goal_inventory.json'):
        shutil.copy2(source/name,out/name)
    history=np.load(source/'historical_tasks.npz')
    pools={}
    for name,key in [('Original','near'),('OriginalReplay','restoration')]:
        tasks=unique_tasks(history[key],env.size)
        pools[name]=tasks[~guard.task_blocked(env,tasks)]
    allowed=np.load(source/'edge_allowed.npy',mmap_mode='r')
    parents=np.flatnonzero(allowed.all(-1));rng.shuffle(parents)
    goals=d['train_goals'];g1=goals[env.depth[goals]==1]
    assert len(g1)==cfg['data']['depth1_targets_per_parent']==6
    g2=goals[env.depth[goals]==2];g3=goals[env.depth[goals]==3]
    groups=[];count=0;needed=cfg['data']['cross_parents']
    for begin in range(0,len(parents),4096):
        s=parents[begin:begin+4096]
        targets=np.column_stack((np.tile(g1,(len(s),1)),rng.choice(g2,len(s)),rng.choice(g3,len(s))))
        tasks=np.stack((np.broadcast_to(s[:,None],targets.shape),targets),axis=-1)
        keep=~((tasks[:,:,0]==tasks[:,:,1])|guard.task_blocked(env,tasks.reshape(-1,2)).reshape(-1,8)).any(-1)
        selected=tasks[keep][:needed-count];groups.append(selected);count+=len(selected)
        if count==needed:break
    if count!=needed:raise ValueError('Insufficient guarded cross parents')
    pools['Cross']=np.concatenate(groups).reshape(-1,2)
    reports={}
    oldkeys=np.unique(history['near'][:,0]*env.size+history['near'][:,1])
    for name,tasks in pools.items():
        np.save(out/(name+'.npy'),tasks)
        reports[name]=coverage(env,tasks)
        if name=='Cross':
            reports[name]['new_decisions_vs_historical_near']=int((~np.isin(tasks[:,0]*env.size+tasks[:,1],oldkeys)).sum())
        atomic_json(out/(name+'_coverage.json'),reports[name])
    atomic_json(out/'ready.json',dict(source_commit=args.source_commit,source_data=str(source),
        pools={k:len(v) for k,v in pools.items()},complete_candidates_only=True,sealed_tests_not_opened=True))
    print(json.dumps(dict(stage='expanded',pools={k:len(v) for k,v in pools.items()})),flush=True)


def mine_expanded(args):
    cfg=json.loads(args.config.read_text());torch.set_num_threads(2)
    env=Cube2(args.oracle,metric='QTM_URF_6');out=args.out
    if (out/'Hard.npy').exists():raise ValueError('Do not overwrite frozen hard pool')
    d=np.load(out/'cohorts.npz');rng=np.random.default_rng(cfg['seed']+10)
    guard=CoverageGuard(env,keys_to_pairs(d['excluded_relation_keys'],env.size),d['state_mask'],d['heldout_goals'])
    cross=np.load(out/'Cross.npy');original=np.load(out/'Original.npy')
    known=np.unique(np.r_[cross[:,0]*env.size+cross[:,1],original[:,0]*env.size+original[:,1]])
    roots=[]
    for goal in d['train_goals']:
        pool=cross[cross[:,1]==goal]
        roots.extend(pool[rng.choice(len(pool),min(len(pool),cfg['hard_followup']['trajectories_per_goal']),replace=False)].tolist())
    model=model_from(args.mining_checkpoint,env,args.device)
    mined,trace=mine_trajectories(model,env,np.asarray(roots),guard,args.device,set(known.tolist()))
    detail=trace.pop('per_decision')
    np.savez_compressed(out/'hard_trajectories.npz',decisions=mined,detail=detail,
        roots=trace.pop('roots'),final_states=trace.pop('final_states'))
    extra=unique_tasks(mined,env.size)
    extra=extra[~np.isin(extra[:,0]*env.size+extra[:,1],known)]
    rng.shuffle(extra);extra=extra[:cfg['hard_followup']['hard_decisions']];trajectory_extra=len(extra)
    parents=np.unique(cross[:,0]);candidates=np.column_stack((rng.choice(parents,262144),rng.choice(d['train_goals'],262144)))
    candidates=unique_tasks(candidates,env.size)
    keep=(candidates[:,0]!=candidates[:,1])&~guard.task_blocked(env,candidates)
    keep&=~np.isin(candidates[:,0]*env.size+candidates[:,1],np.r_[known,extra[:,0]*env.size+extra[:,1]])
    candidates=candidates[keep];scores=predicted_scores(model,env,candidates,args.device)
    truth=env.distance(env.successors(candidates[:,0]),candidates[:,1,None]);before=env.distance(*candidates.T)
    wrong=truth[np.arange(len(candidates)),scores.argmin(-1)]!=before-1
    gap=np.sort(scores,axis=1)[:,1]-scores.min(-1)
    difficult=candidates[wrong|(gap<cfg['hard_followup']['hard_margin'])];rng.shuffle(difficult)
    extra=np.r_[extra,difficult[:cfg['hard_followup']['hard_decisions']-len(extra)]]
    if len(extra)!=cfg['hard_followup']['hard_decisions']:raise ValueError('Insufficient novel hard decisions')
    if guard.task_blocked(env,extra).any():raise ValueError('Protected hard decision')
    np.save(out/'Hard.npy',extra)
    atomic_json(out/'Hard_coverage.json',coverage(env,extra))
    atomic_json(out/'hard_mining.json',dict(**trace,source_commit=args.source_commit,
        checkpoint=str(args.mining_checkpoint),novel_actual_trajectory_decisions=trajectory_extra,
        novel_other_hard_decisions=len(extra)-trajectory_extra,all_decisions_novel_vs_core=True))
    print(json.dumps(dict(stage='hard_pool_ready',actual_trajectory_decisions=trajectory_extra,total=len(extra))),flush=True)


def main():
    p=argparse.ArgumentParser(__doc__)
    for name in ('config','oracle','first-cohorts','second-cohorts','checkpoint1000','checkpoint2000','out'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--source-commit',required=True);p.add_argument('--device',default='cuda')
    p.add_argument('--audit-source',type=Path)
    p.add_argument('--expand-cross',action='store_true')
    p.add_argument('--mine-expanded',action='store_true')
    p.add_argument('--mining-checkpoint',type=Path)
    args=p.parse_args()
    if args.mine_expanded:mine_expanded(args)
    elif args.expand_cross:expand_cross(args)
    else:prepare(args)


if __name__=='__main__':main()
