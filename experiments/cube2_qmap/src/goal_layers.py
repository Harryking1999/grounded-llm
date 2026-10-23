"""Small exact reverse-BFS layer pilot, with protected labels and diagnostics."""
import argparse
import json
from pathlib import Path
import shutil

import numpy as np
import torch

from grounded_llm.core.files import atomic_json
from grounded_llm.tasks.cube2.environment import Cube2
from .coverage_data import (CoverageGuard,keys_to_pairs,closure_keys,goal_orbits,
    load_coverage_config,coverage,hist,unique_tasks,predicted_scores,model_from)


def reverse_layers(env,goal):
    # The six QTM moves include every inverse, so outward BFS is reverse BFS.
    distance=np.full(env.size,255,dtype=np.uint8);distance[goal]=0;frontier=np.array([goal])
    depth=0
    while len(frontier):
        next_states=env.successors(frontier).ravel()
        frontier=np.unique(next_states[distance[next_states]==255]);depth+=1
        distance[frontier]=depth
    if (distance==255).any():raise ValueError('Reverse BFS did not cover the physical graph')
    return distance


@torch.no_grad()
def source_distances(model,env,tasks,device):
    result=[]
    for begin in range(0,len(tasks),1024):
        part=tasks[begin:begin+1024];states=np.unique(part)
        q=model.encode(torch.as_tensor(env.features(states),device=device));a,b=np.searchsorted(states,part).T
        result.append(model.distance(q[a],q[b]).cpu().numpy())
    return np.concatenate(result)


def real_path_edges(lengths,width):
    return np.arange(width-1)[None,:]<np.asarray(lengths)[:,None]


@torch.no_grad()
def switch_metrics(model,env,parents,device):
    goals=np.flatnonzero(env.depth==1);children=env.successors(parents)
    tasks=np.stack((np.broadcast_to(parents[:,None],(len(parents),6)),np.broadcast_to(goals,(len(parents),6))),axis=-1)
    actual=env.distance(tasks[:,:,0],tasks[:,:,1]);truth=env.distance(children[:,:,None],goals[None,None,:])
    optimal=truth==truth.min(1,keepdims=True);masks=(optimal*(1<<np.arange(6)[None,:,None])).sum(1)
    scores=predicted_scores(model,env,tasks.reshape(-1,2),device).reshape(len(parents),6,6).transpose(0,2,1)
    predicted=source_distances(model,env,tasks.reshape(-1,2),device).reshape(-1,6)
    chosen=scores.argmin(1);correct=optimal[np.arange(len(parents))[:,None],chosen,np.arange(6)]
    different_distance=actual.max(-1)!=actual.min(-1)
    different_sets=(masks!=masks[:,:1]).any(-1)
    disjoint=(masks[:,:,None]&masks[:,None,:])==0
    require_switch=disjoint.any((1,2));changes=(chosen!=chosen[:,:1]).any(-1)
    equal_predicted=predicted.max(-1)-predicted.min(-1)<1e-5
    p0,p1=np.triu_indices(6,1);true_different_pairs=actual[:,p0]!=actual[:,p1]
    missed_differences=(np.abs(predicted[:,p0]-predicted[:,p1])<1e-5)&true_different_pairs
    # Diagnose coordinates on wrong choices, not merely on equal true labels.
    qg=model.encode(torch.as_tensor(env.features(goals),device=device))
    qc=model.encode(torch.as_tensor(env.features(children.ravel()),device=device)).reshape(len(parents),6,-1)
    delta=(qc[:,:,None,:]-qg[None,None,:,:]).abs()
    selected=delta[torch.arange(len(parents),device=device)[:,None],torch.as_tensor(chosen,device=device),torch.arange(6,device=device)]
    winners=selected.argmax(-1).cpu().numpy();wrong=~correct
    examples=[]
    for index in np.flatnonzero(different_distance&different_sets&~correct.all(-1))[:8]:
        examples.append(dict(parent=int(parents[index]),goals=goals.tolist(),true_distances=actual[index].tolist(),
            predicted_distances=predicted[index].tolist(),optimal_action_masks=masks[index].tolist(),
            predicted_actions=chosen[index].tolist(),winning_coordinates=winners[index].tolist()))
    return dict(parents=len(parents),true_distance_different_parents=int(different_distance.sum()),
        true_distance_equal_parents=int((~different_distance).sum()),
        different_optimal_sets=int(different_sets.sum()),pairwise_disjoint_optimal_sets=int(require_switch.sum()),
        different_true_distance_pairs=int(true_different_pairs.sum()),missed_true_distance_differences=int(missed_differences.sum()),
        equal_predictions_despite_different_true_distances=int((equal_predicted&different_distance).sum()),
        required_switch_response_rate=float(changes[require_switch].mean()) if require_switch.any() else None,
        candidate_top1=float(correct.mean()),all_six_goals_correct_parents=int(correct.all(-1).sum()),
        wrong_decisions=int(wrong.sum()),wrong_decision_winning_coordinates=hist(winners[wrong]),examples=examples)


@torch.no_grad()
def layer_report(model,env,data,role,device,full=False,final_paths=False):
    model.eval();tasks=np.load(data/(role+'.npy'))
    if len(tasks)>1024 and not full:
        rng=np.random.default_rng(2026102409);distance=env.distance(*tasks.T)
        indices=np.concatenate([rng.choice(np.flatnonzero(distance==d),min(64,int((distance==d).sum())),replace=False)
            for d in np.unique(distance)])
        tasks=tasks[indices]
    def decisions(rows):
        scores=predicted_scores(model,env,rows,device);truth=env.distance(env.successors(rows[:,0]),rows[:,1,None])
        depth=env.distance(*rows.T);pred=source_distances(model,env,rows,device)
        right=truth[np.arange(len(rows)),scores.argmin(-1)]==depth-1
        ordered=truth[:,:,None]<truth[:,None,:];respected=scores[:,:,None]<scores[:,None,:]
        def summary(keep):
            required=ordered[keep];correct=respected[keep]&required
            return dict(count=int(keep.sum()),top1=float(right[keep].mean()),distance_mae=float(np.abs(pred[keep]-depth[keep]).mean()),
                strict_pair_order_accuracy=float(correct.sum()/required.sum()) if required.any() else None,
                all_candidate_ordering_rate=float((~required|respected[keep]).all((1,2)).mean()))
        return dict(rows=len(rows),distance_mae=float(np.abs(pred-depth).mean()),candidate_top1=float(right.mean()),
            strict_pair_order_accuracy=summary(np.ones(len(rows),bool))['strict_pair_order_accuracy'],
            all_candidate_ordering_rate=summary(np.ones(len(rows),bool))['all_candidate_ordering_rate'],
            by_layer={str(int(d)):summary(depth==d) for d in np.unique(depth)},
            by_goal_layer={str(int(g)):{str(int(d)):summary((rows[:,1]==g)&(depth==d))
                for d in np.unique(depth[rows[:,1]==g])} for g in np.unique(rows[:,1])})
    result=dict(training=decisions(tasks),micro_development=decisions(np.load(data/'MicroHoldout.npy')),
        switching=switch_metrics(model,env,np.load(data/'SwitchParents.npy'),device))
    micro=np.load(data/'Micro.npy');parents,counts=np.unique(micro[:,0],return_counts=True)
    shared=parents[counts==len(np.load(data/'goals.npy'))]
    g1=np.flatnonzero(env.depth==1);d=env.distance(shared[:,None],g1)
    shared=shared[d.max(-1)!=d.min(-1)]
    result['micro_shared_training_switching']=switch_metrics(model,env,shared,device)
    p=np.load(data/'paths.npz');states=p['states'];goals=p['goals'];lengths=p['lengths']
    valid=np.arange(states.shape[1])[None,:]<=lengths[:,None]
    rows=np.column_stack((states[valid],np.broadcast_to(goals[:,None],states.shape)[valid]))
    predicted=np.full(states.shape,np.nan);predicted[valid]=source_distances(model,env,rows,device)
    pairs=p['pair_allowed'];stars=p['star_allowed']
    edge_valid=real_path_edges(lengths,states.shape[1])&pairs[:,:-1]&pairs[:,1:]
    violation=(predicted[:,:-1]<=predicted[:,1:])&edge_valid
    first_wrong=[];complete_first_wrong=[];censored=0;fully=0
    for index,(goal,length) in enumerate(zip(goals,lengths)):
        live=stars[index,:length];complete=bool(live.all());fully+=int(complete)
        first=-1
        for step in range(int(length)):
            if not live[step]:censored+=1;break
            row=np.array([[states[index,step],goal]])
            scores=predicted_scores(model,env,row,device)[0]
            truth=env.distance(env.successors(int(states[index,step])),goal)
            if truth[scores.argmin()]!=length-step-1:first=int(length-step);break
        first_wrong.append(first)
        if complete:complete_first_wrong.append(first)
    result['shortest_paths']=dict(paths=len(lengths),fully_admissible_paths=fully,censored_before_first_error=censored,
        evaluable_monotonic_edges=int(edge_valid.sum()),protected_monotonic_edges_excluded=int((valid[:,1:]&~edge_valid).sum()),
        monotonic_violations=int(violation.sum()),strict_monotonic_rate=float(1-violation.sum()/edge_valid.sum()) if edge_valid.any() else None,
        first_wrong_layer_observed_prefix=hist(first_wrong),first_wrong_layer_fully_admissible=hist(complete_first_wrong),
        note='A -1 in censored prefixes is not a complete-path success. Protected labels never enter training or these ordering metrics.')
    if final_paths:
        # Final evaluation only, after model selection has been frozen. These
        # protected intermediate labels never fed training or adaptive mining.
        live=real_path_edges(lengths,states.shape[1]);assert int(live.sum())==int(lengths.sum())
        all_rows=np.column_stack((states[:,:-1][live],
            np.broadcast_to(goals[:,None],states[:,:-1].shape)[live]))
        scores=predicted_scores(model,env,all_rows,device)
        truth=env.distance(env.successors(all_rows[:,0]),all_rows[:,1,None])
        wrong=np.zeros(live.shape,bool)
        wrong[live]=truth[np.arange(len(scores)),scores.argmin(-1)]!=env.distance(*all_rows.T)-1
        first=[]
        for index,length in enumerate(lengths):
            error=np.flatnonzero(wrong[index,:length]);first.append(-1 if not len(error) else int(length-error[0]))
        violation=(predicted[:,:-1]<=predicted[:,1:])&live
        result['final_full_shortest_path_evaluation']=dict(paths=len(lengths),edges=int(live.sum()),
            strict_monotonic_rate=float(1-violation.sum()/live.sum()),monotonic_violations=int(violation.sum()),
            first_wrong_layer=hist(first),paths_with_no_candidate_error=int((np.asarray(first)==-1).sum()),
            note='Includes protected intermediate relations for final evaluation only, after freezing selection; never used to train, mine, or choose a checkpoint. Oracle path following is a diagnostic, not model search or free-rollout success.')
    return result


def prepare(args):
    cfg=load_coverage_config(args.config);out=args.out;out.mkdir(parents=True,exist_ok=False)
    env=Cube2(args.oracle,metric='QTM_URF_6');rng=np.random.default_rng(cfg['seed'])
    d=np.load(args.source/'cohorts.npz');keys=d['excluded_relation_keys']
    guard=CoverageGuard(env,keys_to_pairs(keys,env.size),d['state_mask'],d['heldout_goals'])
    all_goals,canonical=goal_orbits(env);goals=list(all_goals[env.depth[all_goals]==1])
    for depth in (2,3):
        candidates=np.unique(canonical[(env.depth[all_goals]==depth)&np.isin(all_goals,d['train_goals'])])
        orbit=next(int(c) for c in rng.permutation(candidates) if (canonical==c).sum()==6)
        goals.extend(all_goals[canonical==orbit].tolist())
    goals=np.asarray(goals);np.save(out/'goals.npy',goals)
    distances={};layer_inventory={}
    for goal in goals:
        distance=reverse_layers(env,int(goal));distances[int(goal)]=distance
        # Check exact graph labels against the existing task adapter, including
        # every state, rather than using the Q predictions as any truth.
        for begin in range(0,env.size,32768):
            ids=np.arange(begin,min(begin+32768,env.size))
            if not np.array_equal(distance[ids],env.distance(ids,goal)):raise ValueError('BFS/adapter disagreement')
        np.save(out/f'layers_{goal}.npy',distance)
        layer_inventory[str(int(goal))]=hist(distance)
        print(json.dumps(dict(goal=int(goal),stage='exact_reverse_layers')),flush=True)
    # Select shared parents whose goal-1 distances and required action sets
    # genuinely differ. Each parent supplies every pilot goal in micro fitting.
    candidates=np.unique(np.load(args.source/'Cross.npy')[:,0]);rng.shuffle(candidates)
    selected=[]
    for parent in candidates:
        rows=np.column_stack((np.full(len(goals),parent),goals))
        if guard.task_blocked(env,rows).any():continue
        truth=env.distance(env.successors(int(parent))[:,None],goals[None,:])
        masks=((truth==truth.min(0))*(1<<np.arange(6)[:,None])).sum(0)
        gd=env.distance(parent,goals[:6])
        if gd.max()==gd.min() or np.bitwise_and.reduce(masks[:6])!=0:continue
        selected.append(parent)
        if len(selected)==cfg['data']['micro_parents']:break
    if len(selected)!=cfg['data']['micro_parents']:raise ValueError('Insufficient goal-dependent micro parents')
    micro=np.stack((np.broadcast_to(np.asarray(selected)[:,None],(len(selected),len(goals))),
        np.broadcast_to(goals,(len(selected),len(goals)))),axis=-1).reshape(-1,2)
    # Include a tiny near-to-far seed field too: a far-only micro set could
    # diagnose an inactive-coordinate initialization rather than fitting of
    # a connected set of distance layers.
    seeds=[]
    for goal in goals:
        for depth in range(1,env.diameter+1):
            sources=np.flatnonzero(distances[int(goal)]==depth);rng.shuffle(sources)
            for begin in range(0,len(sources),2048):
                s=sources[begin:begin+2048];rows=np.column_stack((s,np.full(len(s),goal)))
                allowed=rows[~guard.task_blocked(env,rows)]
                if len(allowed):seeds.append(allowed[0]);break
    micro=unique_tasks(np.r_[micro,np.asarray(seeds)],env.size)
    np.save(out/'Micro.npy',micro)
    micro_nodes=np.unique(np.r_[micro[:,0],env.successors(micro[:,0]).ravel()])
    micro_nodes=np.unique(np.concatenate([env.axis_symmetry(micro_nodes,k) for k in range(6)]))
    # Development sources avoid micro training stars; relation/axis closures
    # are then protected in both main pilot arms as well as in micro fitting.
    holdout=[]
    for goal in goals:
        for depth in range(cfg['data']['micro_holdout_minimum_layer'],env.diameter+1):
            sources=np.flatnonzero(distances[int(goal)]==depth);rng.shuffle(sources)
            chosen=[]
            for batch in range(0,len(sources),2048):
                s=sources[batch:batch+2048];rows=np.column_stack((s,np.full(len(s),goal)))
                star=np.column_stack((s,env.successors(s)))
                keep=~guard.task_blocked(env,rows)&~np.isin(star,micro_nodes).any(-1)
                chosen.extend(rows[keep].tolist())
                if len(chosen)>=cfg['data']['micro_holdout_per_goal_depth']:break
            holdout.extend(chosen[:cfg['data']['micro_holdout_per_goal_depth']])
    holdout=np.asarray(holdout,dtype=np.int64);keys=np.unique(np.r_[keys,closure_keys(env,holdout)])
    guard=CoverageGuard(env,keys_to_pairs(keys,env.size),d['state_mask'],d['heldout_goals'])
    if guard.task_blocked(env,micro).any():raise ValueError('Micro development overlaps training labels')
    np.save(out/'MicroHoldout.npy',holdout)
    stored={name:d[name] for name in d.files};stored['excluded_relation_keys']=keys
    np.savez_compressed(out/'cohorts.npz',**stored)
    for name in ('OriginalReplay.npy','edge_allowed.npy'):shutil.copy2(args.source/name,out/name)
    # Recheck replay under the extra development closure, preserving no new
    # label that happens to belong to a protected micro development star.
    replay=np.load(out/'OriginalReplay.npy');replay=replay[~guard.task_blocked(env,replay)]
    np.save(out/'OriginalReplay.npy',replay)
    layered=[];available={}
    for goal in goals:
        available[str(int(goal))]={}
        for depth in range(1,env.diameter+1):
            sources=np.flatnonzero(distances[int(goal)]==depth);rng.shuffle(sources);chosen=[]
            for begin in range(0,len(sources),2048):
                s=sources[begin:begin+2048];rows=np.column_stack((s,np.full(len(s),goal)))
                chosen.extend(rows[~guard.task_blocked(env,rows)].tolist())
                if len(chosen)>=cfg['data']['layer_rows_per_goal_depth']:break
            chosen=chosen[:cfg['data']['layer_rows_per_goal_depth']];layered.extend(chosen)
            available[str(int(goal))][str(depth)]=len(chosen)
    layered=unique_tasks(np.asarray(layered),env.size)
    uniform=np.column_stack((rng.integers(env.size,size=len(layered)),layered[:,1]))
    for _ in range(10000):
        rowkeys=uniform[:,0]*env.size+uniform[:,1];_,first=np.unique(rowkeys,return_index=True)
        duplicate=np.ones(len(rowkeys),bool);duplicate[first]=False
        bad=guard.task_blocked(env,uniform)|(uniform[:,0]==uniform[:,1])|duplicate
        if not bad.any():break
        uniform[bad,0]=rng.integers(env.size,size=bad.sum())
    else:raise ValueError('Uniform control did not terminate')
    for name,rows in [('Layered',layered),('UniformLayers',uniform)]:
        stars=np.column_stack((rows[:,0],env.successors(rows[:,0])))
        for goal in goals:
            keep=rows[:,1]==goal
            if not np.array_equal(distances[int(goal)][stars[keep]],env.distance(stars[keep],goal)):
                raise ValueError('A candidate label differs from BFS truth')
        np.save(out/(name+'.npy'),rows);atomic_json(out/(name+'_coverage.json'),coverage(env,rows))
    switch=[]
    for parent in candidates:
        rows=np.column_stack((np.full(6,parent),goals[:6]))
        if guard.task_blocked(env,rows).any():continue
        gd=env.distance(parent,goals[:6]);truth=env.distance(env.successors(int(parent))[:,None],goals[:6][None,:])
        masks=((truth==truth.min(0))*(1<<np.arange(6)[:,None])).sum(0)
        if gd.max()!=gd.min() and (masks!=masks[:1]).any():switch.append(parent)
        if len(switch)==512:break
    np.save(out/'SwitchParents.npy',switch)
    paths=[];path_goals=[];lengths=[];pair_flags=[];star_flags=[]
    for goal in goals:
        for depth in cfg['data']['path_root_depths']:
            pool=layered[(layered[:,1]==goal)&(distances[int(goal)][layered[:,0]]==depth)]
            for source in pool[:cfg['data']['paths_per_goal_depth'],0]:
                path=[int(source)]
                while path[-1]!=goal:
                    children=env.successors(path[-1]);next_depth=distances[int(goal)][children]
                    path.append(int(rng.choice(children[next_depth==distances[int(goal)][path[-1]]-1])))
                length=len(path)-1;pad=np.pad(path,(0,env.diameter-length),constant_values=int(goal))
                rows=np.column_stack((pad,np.full(len(pad),goal)))
                paths.append(pad);path_goals.append(goal);lengths.append(length)
                pair_flags.append(~guard.contains(rows));star_flags.append(~guard.task_blocked(env,rows))
    np.savez_compressed(out/'paths.npz',states=paths,goals=path_goals,lengths=lengths,pair_allowed=pair_flags,star_allowed=star_flags)
    atomic_json(out/'ready.json',dict(source_commit=args.source_commit,goals=goals.tolist(),
        reverse_bfs_layers=layer_inventory,selected_complete_decisions_by_goal_layer=available,
        micro_rows=len(micro),micro_development_rows=len(holdout),layered_rows=len(layered),uniform_rows=len(uniform),
        layer0='one terminal node per goal; Q(g,g)=0 by definition, no terminal action decision counted',
        exact_adapter_and_all_candidate_labels_verified=True,edges_trained=0,sealed_tests_not_opened=True))


def main():
    p=argparse.ArgumentParser(__doc__)
    for name in ('config','source','oracle','out'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--source-commit',required=True)
    p.add_argument('--checkpoint',type=Path);p.add_argument('--role',default='Layered')
    p.add_argument('--device',default='cuda')
    p.add_argument('--final-path-evaluation',action='store_true')
    args=p.parse_args()
    if args.checkpoint:
        torch.set_num_threads(2);env=Cube2(args.oracle,metric='QTM_URF_6')
        model=model_from(args.checkpoint,env,args.device);args.out.mkdir(parents=True,exist_ok=False)
        atomic_json(args.out/'metrics.json',dict(source_commit=args.source_commit,checkpoint=str(args.checkpoint),
            role=args.role,metrics=layer_report(model,env,args.source,args.role,args.device,full=True,
                final_paths=args.final_path_evaluation)))
    else:prepare(args)


if __name__=='__main__':main()
