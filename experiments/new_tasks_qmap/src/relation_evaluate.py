"""Frozen exact-goal distance, ranking, reachability and matched greedy policies."""
import numpy as np
import torch
from grounded_llm.core.qmap.metrics import correlation, encode, average_ranks
from grounded_llm.core.qmap.model import soft_rank_loss


class QCache:
    def __init__(self,model,corpus,device):
        self.model,self.corpus,self.device=model,corpus,device
        self.cache={}

    @torch.no_grad()
    def q(self,ids):
        ids=np.asarray(ids,dtype=np.int64); flat=ids.ravel()
        missing=np.asarray([i for i in np.unique(flat) if int(i) not in self.cache],dtype=np.int64)
        if len(missing): self.cache.update(zip(map(int,missing),encode(self.model,self.corpus,missing,self.device)))
        return torch.stack([self.cache[int(i)] for i in flat]).reshape(*ids.shape,-1)

    @torch.no_grad()
    def scores(self,source,goal):
        source,goal=np.broadcast_arrays(source,goal)
        return self.model.distance(self.q(source),self.q(goal)).cpu().numpy()


def safe_scores(cache,children,goals):
    valid=children>=0; result=np.full(children.shape,np.inf,dtype=np.float32)
    result[valid]=cache.scores(children[valid],np.broadcast_to(goals[:,None],children.shape)[valid])
    return result


def calibrate(predictions,truth):
    dead=~np.isfinite(truth)
    if not dead.any() or dead.all(): return None
    thresholds=np.unique(np.quantile(predictions,np.linspace(0,1,201)))
    def quality(t):
        chosen=predictions>=t
        tpr=chosen[dead].mean(); fpr=chosen[~dead].mean()
        return (.5*(tpr+1-fpr),-fpr)
    return float(max(thresholds,key=quality))


def reachability(predictions,truth,threshold,proxy):
    dead=~np.isfinite(truth); finite=~dead
    auc=None
    if dead.any() and finite.any():
        ranks=average_ranks(predictions)
        auc=float((ranks[dead].sum()-dead.sum()*(dead.sum()+1)/2)/(dead.sum()*finite.sum()))
    return dict(proven_unreachable=int(dead.sum()),finite=int(finite.sum()),unknown=0,auroc=auc,
                validation_threshold=threshold,
                recall=float((predictions[dead]>=threshold).mean()) if dead.any() and threshold is not None else None,
                finite_false_positive=float((predictions[finite]>=threshold).mean()) if finite.any() and threshold is not None else None,
                recall_at_training_proxy=float((predictions[dead]>=proxy-.5).mean()) if dead.any() else None,
                note='Finite surrogate is not infinity; classification never filters planning candidates.')


def coordinate_statistics(source,goal,truth,predictions,cap,activation):
    """Diagnostic scaling only; physical distance labels remain unnormalized."""
    q=np.stack([source,goal],axis=1)/cap
    outside=((q<0)|(q>1)).mean(axis=(1,2))
    boundary=((q<1e-4)|(q>1-1e-4)).mean(axis=(1,2)) if activation!='linear' else None
    derivative=(q*(1-q) if activation=='sigmoid' else np.minimum(q,1-q)**2 if activation=='softsign' else np.ones_like(q))
    low_gradient=(derivative<1e-4).mean(axis=(1,2))
    def describe(mask):
        finite=mask & np.isfinite(truth)
        return dict(count=int(mask.sum()),
            finite_mae=float(np.abs(predictions[finite]-truth[finite]).mean()) if finite.any() else None,
            finite_bias=float((predictions[finite]-truth[finite]).mean()) if finite.any() else None,
            predicted_mean=float(predictions[mask].mean()),
            boundary_fraction=float(boundary[mask].mean()) if boundary is not None else None,
            low_output_derivative_fraction=float(low_gradient[mask].mean()),
            outside_sigmoid_range_fraction=float(outside[mask].mean()))
    bins={}; lower=-1
    for upper in (0,1,2,5,10,20,40,80,160,np.inf):
        mask=np.isfinite(truth) & (truth>lower) & (truth<=upper)
        if mask.any(): bins[f'({lower},{upper}]']=describe(mask)
        lower=upper
    if (~np.isfinite(truth)).any(): bins['proven_unreachable']=describe(~np.isfinite(truth))
    return dict(output_activation=activation,coordinate_divisor_for_diagnostics_only=cap,
        labels='physical unit-cost steps, without normalization or inverse normalization',
        boundary_tolerance_relative_to_cap=1e-4,derivative_tolerance_relative_to_cap=1e-4,
        all=describe(np.ones(len(truth),dtype=bool)),by_true_distance=bins)


def rollout(corpus,cache,rows,horizon,policy):
    current,goal=rows[:,0].copy(),rows[:,1].copy()
    exact=corpus.distance(current,goal)
    if not np.isfinite(exact).all(): raise ValueError('Rollout starts must have exact finite labels')
    steps=np.zeros(len(rows),dtype=int); active=current!=goal
    seen=[{int(s)} for s in current]; cycles=np.zeros(len(rows),bool); dead=np.zeros(len(rows),bool)
    no_action=np.zeros(len(rows),bool); traces=[[] for _ in rows]
    for _ in range(horizon):
        ids=np.flatnonzero(active)
        if not len(ids): break
        children=corpus.successors(current[ids]); valid=children>=0
        empty=~valid.any(axis=1)
        no_action[ids[empty]]=True; active[ids[empty]]=False
        ids=ids[~empty]; children=children[~empty]; valid=valid[~empty]
        if not len(ids): continue
        if policy=='q': scores=safe_scores(cache,children,goal[ids])
        else:
            scores=np.full(children.shape,np.inf)
            targets=np.broadcast_to(goal[ids,None],children.shape)[valid]
            scores[valid]=(corpus.distance(children[valid],targets) if policy=='oracle'
                           else corpus.baseline(children[valid],targets) if policy=='heuristic'
                           else np.zeros(valid.sum()))
        choices=np.argmin(1+scores,axis=1)
        next_states=children[np.arange(len(ids)),choices]
        # Oracle inspection occurs AFTER action selection and cannot prune/stop Q.
        dead[ids]|=~np.isfinite(corpus.distance(next_states,goal[ids]))
        for j,slot in zip(ids,choices): traces[j].append(corpus.action(int(current[j]),int(slot)))
        current[ids]=next_states; steps[ids]+=1
        for j in ids:
            if current[j]==goal[j]: active[j]=False
            elif int(current[j]) in seen[j]: cycles[j]=True; active[j]=False
            else: seen[j].add(int(current[j]))
    solved=current==goal; optimum=solved & (steps==exact)
    examples=[]
    for status in (True,False):
        for i in np.flatnonzero(solved==status)[:2]:
            examples.append(dict(start=corpus.describe(int(rows[i,0])),goal=corpus.describe(int(goal[i])),
                                 end=corpus.describe(int(current[i])),solved=bool(solved[i]),
                                 optimum=float(exact[i]),cost=int(steps[i]),cycle=bool(cycles[i]),
                                 entered_unreachable=bool(dead[i]),actions=traces[i]))
    per_scene={}
    if hasattr(corpus,'scenes'):
        scene_ids=corpus.locate(rows[:,0])
        per_scene={str(sid):dict(count=int((scene_ids==sid).sum()),solved=int(solved[scene_ids==sid].sum()),
                    success_rate=float(solved[scene_ids==sid].mean())) for sid in np.unique(scene_ids)}
    return dict(count=len(rows),solved=int(solved.sum()),success_rate=float(solved.mean()) if len(rows) else None,
                optimal=int(optimum.sum()),optimal_rate=float(optimum.mean()) if len(rows) else None,cycles=int(cycles.sum()),
                entered_unreachable=int(dead.sum()),no_legal_action=int(no_action.sum()),
                budget_exhausted=int(active.sum()),horizon=horizon,illegal_actions=0,
                mean_excess_cost_on_success=float((steps[solved]-exact[solved]).mean()) if solved.any() else None,
                mean_cost_ratio_on_success=float((steps[solved]/np.maximum(exact[solved],1)).mean()) if solved.any() else None,
                by_distance={str(d):dict(count=int((exact==d).sum()),solved=int(solved[exact==d].sum()),optimal=int(optimum[exact==d].sum())) for d in np.unique(exact)},
                by_scene=per_scene,examples=examples)


def planning_ids(corpus,rows,truth,count,rng):
    """Unique starts/queries; share the budget across scenes then distance layers."""
    pool=np.flatnonzero(np.isfinite(truth) & (truth>0))
    if len(pool)<=count: return pool
    scene=corpus.locate(rows[pool,0]) if hasattr(corpus,'scenes') else np.zeros(len(pool),int)
    groups=[pool[scene==sid] for sid in np.unique(scene)]
    chosen=[]
    for group in groups:
        layers=[rng.permutation(group[truth[group]==d]).tolist() for d in np.unique(truth[group])]
        order=[]
        while any(layers):
            for layer in layers:
                if layer: order.append(layer.pop())
        chosen.append(order)
    result=[]
    while len(result)<count:
        for group in chosen:
            if group and len(result)<count: result.append(group.pop(0))
    return np.asarray(result,dtype=np.int64)


@torch.no_grad()
def evaluate(model,corpus,data,config,cohorts=None,exposure=None,planning=True,threshold=None):
    model.eval(); cache=QCache(model,corpus,next(model.parameters()).device)
    names=cohorts or [key[6:] for key in data if key.startswith('tasks_') and key!='tasks_train']
    predictions={name:cache.scores(data['tasks_'+name][:,0],data['tasks_'+name][:,1]) for name in names}
    if threshold is None:
        validation=[name for name in names if 'validation' in name]
        if validation:
            threshold=calibrate(np.concatenate([predictions[n] for n in validation]),np.concatenate([data['distances_'+n] for n in validation]))
    results={}
    for name in names:
        rows=data['tasks_'+name]; truth=data['distances_'+name]; pred=predictions[name]
        finite=np.isfinite(truth); error=np.abs(pred[finite]-truth[finite]); signed=pred[finite]-truth[finite]
        children=data['children_'+name]; valid=children>=0
        scores=safe_scores(cache,children,rows[:,1]); child_truth=data['child_distances_'+name]
        decisions=finite & (truth>0) & valid.any(axis=1)
        ids=np.flatnonzero(decisions)
        if len(ids)>config['evaluation']['decision_count']:
            from .relation_data import balanced
            ids=balanced(ids[:,None],config['evaluation']['decision_count'],truth[ids],np.random.default_rng(config['seed']+len(name))).ravel()
        chosen=scores[ids].argmin(axis=1)
        optimal=data['optimal_'+name][ids]
        order=np.argsort(scores[ids],axis=1,kind='stable')
        actual=child_truth[ids]; ordered=(actual[:,:,None]<actual[:,None,:]) & valid[ids,:,None] & valid[ids,None,:]
        strict=(scores[ids,:,None]<scores[ids,None,:]-1e-7)
        reverse=cache.scores(rows[:,1],rows[:,0]); reverse_truth=corpus.distance(rows[:,0],rows[:,1],reverse=True)
        rank_loss=None
        if len(ids):
            device=next(model.parameters()).device
            target=np.where(np.isfinite(child_truth[ids]),child_truth[ids],config['training']['unreachable_target'])
            rank_loss=float(soft_rank_loss(torch.as_tensor(np.where(valid[ids],1+scores[ids],0),device=device),
                torch.as_tensor(1+target,device=device),torch.as_tensor(valid[ids],device=device),config['training']['rank_temperature']))
        bellman=pred[ids]-np.min(1+scores[ids],axis=1)
        result=dict(count=len(rows),finite_count=int(finite.sum()),finite_mae=float(error.mean()),
                    finite_huber=float(np.where(error<1,.5*error**2,error-.5).mean()),
                    finite_bias=float(signed.mean()),spearman=correlation(pred[finite],truth[finite]),
                    absolute_error_quantiles={str(q):float(np.quantile(error,q)) for q in (0,.5,.9,.95,1)},
                    within_half_step=float((error<=.5).mean()),
                    by_distance={str(d):dict(count=int((truth==d).sum()),mae=float(np.abs(pred[truth==d]-d).mean()),predicted_mean=float(pred[truth==d].mean())) for d in np.unique(truth[finite])},
                    decision_count=len(ids),shortest_action_rate=float(optimal[np.arange(len(ids)),chosen].mean()) if len(ids) else None,
                    top_k_optimal_hit={str(k):float(np.take_along_axis(optimal,order[:,:k],axis=1).any(axis=1).mean()) for k in (1,3,5)} if len(ids) else {},
                    strict_candidate_order_accuracy=float(strict[ordered].mean()) if ordered.any() else None,
                    dead_end_choice_rate=float((~np.isfinite(actual[np.arange(len(ids)),chosen])).mean()) if len(ids) else None,
                    candidate_soft_rank_loss=rank_loss,
                    bellman=dict(mae=float(np.abs(bellman).mean()) if len(ids) else None,upper_violation_rate=float((bellman>1e-3).mean()) if len(ids) else None),
                    directionality=dict(predicted_mean_absolute_asymmetry=float(np.abs(pred-reverse).mean()),
                        both_reachable=int((finite & np.isfinite(reverse_truth)).sum()),
                        forward_only=int((finite & ~np.isfinite(reverse_truth)).sum()),
                        reverse_finite_mae=float(np.abs(reverse[np.isfinite(reverse_truth)]-reverse_truth[np.isfinite(reverse_truth)]).mean()) if np.isfinite(reverse_truth).any() else None,
                        reverse_scope='reverse target may lie outside the successful-terminal training target domain'),
                    unreachable=reachability(pred,truth,threshold,config['training']['unreachable_target']))
        result['coordinate_statistics']=coordinate_statistics(cache.q(rows[:,0]).cpu().numpy(),cache.q(rows[:,1]).cpu().numpy(),
            truth,pred,model.cap,getattr(model,'output_activation','sigmoid'))
        if exposure is not None:
            seen_source=exposure[rows[:,0]]>0; seen_goal=exposure[rows[:,1]]>0
            result['actual_gradient_endpoint_exposure']=dict(both_seen=int((seen_source & seen_goal).sum()),
                one_unseen=int((seen_source ^ seen_goal).sum()),both_unseen=int((~seen_source & ~seen_goal).sum()),
                both_seen_finite_mae=float(np.abs(pred[finite & seen_source & seen_goal]-truth[finite & seen_source & seen_goal]).mean()) if (finite & seen_source & seen_goal).any() else None,
                any_unseen_finite_mae=float(np.abs(pred[finite & ~(seen_source & seen_goal)]-truth[finite & ~(seen_source & seen_goal)]).mean()) if (finite & ~(seen_source & seen_goal)).any() else None)
        if planning:
            rng=np.random.default_rng(config['seed']+len(name))
            selected=rows[planning_ids(corpus,rows,truth,config['evaluation']['rollout_count'],rng)]
            result['planning']={policy:rollout(corpus,cache,selected,config['evaluation']['horizon'],policy) for policy in ('q','oracle','heuristic','immediate_cost_only')}
        results[name]=result
    return dict(reachability_threshold=threshold,cohorts=results)


def main():
    import argparse
    import json
    from pathlib import Path
    from grounded_llm.core.files import atomic_json
    from grounded_llm.core.qmap.model import StateEncoder
    from .relation_data import load
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--checkpoint',required=True,type=Path)
    parser.add_argument('--run',required=True,type=Path)
    parser.add_argument('--out',required=True,type=Path)
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--cohorts',nargs='+')
    args=parser.parse_args()
    if args.out.exists(): raise FileExistsError('Keep previous frozen evaluations; choose a new output')
    saved=torch.load(args.checkpoint,map_location=args.device,weights_only=False)
    config=saved['config']; torch.set_num_threads(config['training']['cpu_threads'])
    corpus,data,_=load(config,args.run/'data')
    model=StateEncoder(input_dim=saved['input_dim'],**config['model']).to(args.device)
    model.load_state_dict(saved['model']); model.requires_grad_(False)
    result=evaluate(model,corpus,data,config,cohorts=args.cohorts,exposure=saved['exposure'])
    atomic_json(args.out,dict(checkpoint=str(args.checkpoint),step=saved['step'],source_commit=saved['source_commit'],metrics=result))
    print(json.dumps(dict(step=saved['step'],cohorts={name:row['planning']['q']['success_rate'] for name,row in result['cohorts'].items()})),flush=True)


if __name__=='__main__': main()
