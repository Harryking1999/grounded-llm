"""Frozen exact-goal distance, ranking, reachability and matched greedy policies."""
import numpy as np
import torch
from grounded_llm.core.qmap.metrics import correlation, encode, average_ranks
from grounded_llm.core.qmap.model import soft_rank_loss


def distance_strata(values):
    for low,high in ((1,1),(2,2),(3,5),(6,10),(11,20),(21,40),(41,80),(81,np.inf)):
        yield f'{low}_{high}',np.isfinite(values)&(values>=low)&(values<=high)
    yield 'proven_unreachable',~np.isfinite(values)


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
            positive_finite_predicted_zero_fraction=(float((predictions[finite & (truth>0)]<=1e-6).mean())
                if (finite & (truth>0)).any() else None),
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


def unit_cost_consistency(parent,children,optimal,valid):
    """Frozen distance-field diagnosis; never changes a candidate or its score."""
    children=np.where(valid,children,np.inf)
    chosen=children.argmin(axis=1); index=np.arange(len(parent))
    drop=parent-children[index,chosen]; residual=drop-1
    optimal=optimal & valid
    teacher_drop=np.broadcast_to(parent[:,None],children.shape)[optimal]-children[optimal]
    selected_optimal=optimal[index,chosen]
    return dict(mae=float(np.abs(residual).mean()) if len(parent) else None,
        upper_violation_rate=float((residual>1e-3).mean()) if len(parent) else None,
        lower_violation_rate=float((residual < -1e-3).mean()) if len(parent) else None,
        chosen_non_descent_rate=float((drop<=1e-6).mean()) if len(parent) else None,
        chosen_predicted_drop_quantiles={str(q):float(np.quantile(drop,q)) for q in (0,.1,.5,.9,1)} if len(parent) else {},
        chosen_true_optimal_count=int(selected_optimal.sum()),
        chosen_true_optimal_but_non_descent_count=int((selected_optimal & (drop<=1e-6)).sum()),
        teacher_optimal_edge_count=int(optimal.sum()),
        teacher_optimal_drop_mae_to_one=float(np.abs(teacher_drop-1).mean()) if len(teacher_drop) else None,
        teacher_optimal_non_descent_rate=float((teacher_drop<=1e-6).mean()) if len(teacher_drop) else None,
        teacher_optimal_predicted_drop_quantiles={str(q):float(np.quantile(teacher_drop,q)) for q in (0,.1,.5,.9,1)} if len(teacher_drop) else {},
        note='Reference decisions, not whole trajectories. Unit-cost teacher-optimal edges should have a true remaining-distance drop of1. Candidate order can be correct even when the parent-to-child predicted drop is wrong. No oracle is used to change model action selection.')


def rollout(corpus,cache,rows,horizon,policy,edge_records=None):
    current,goal=rows[:,0].copy(),rows[:,1].copy()
    exact=corpus.distance(current,goal)
    if not np.isfinite(exact).all(): raise ValueError('Rollout starts must have exact finite labels')
    steps=np.zeros(len(rows),dtype=int); active=current!=goal
    seen=[{int(s)} for s in current]; cycles=np.zeros(len(rows),bool); dead=np.zeros(len(rows),bool)
    no_action=np.zeros(len(rows),bool); traces=[[] for _ in rows]
    descent=[[] for _ in rows]; progress_counts=dict(executed_edges=0,predicted_decrease=0,predicted_plateau=0,
        predicted_increase=0,true_finite_edges=0,true_optimal_edges=0,predicted_decrease_but_not_true_optimal=0,
        entered_dead_edges=0,non_goal_predicted_zero=0)
    progress_layers={name:dict(executed_edges=0,true_optimal_edges=0,predicted_decrease=0,entered_dead_edges=0)
                     for name,_ in distance_strata(np.array([],dtype=float))}
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
        after_truth=corpus.distance(next_states,goal[ids])
        dead[ids]|=~np.isfinite(after_truth)
        if policy=='q':
            before=cache.scores(current[ids],goal[ids]); drop=before-scores[np.arange(len(ids)),choices]
            before_truth=corpus.distance(current[ids],goal[ids]); finite=np.isfinite(before_truth)&np.isfinite(after_truth)
            optimal=finite & (before_truth==1+after_truth); decreasing=drop>1e-6
            progress_counts['executed_edges']+=len(ids)
            progress_counts['predicted_decrease']+=int(decreasing.sum())
            progress_counts['predicted_plateau']+=int((np.abs(drop)<=1e-6).sum())
            progress_counts['predicted_increase']+=int((drop < -1e-6).sum())
            progress_counts['true_finite_edges']+=int(finite.sum())
            progress_counts['true_optimal_edges']+=int(optimal.sum())
            progress_counts['predicted_decrease_but_not_true_optimal']+=int((decreasing & ~optimal).sum())
            progress_counts['entered_dead_edges']+=int((np.isfinite(before_truth)&~np.isfinite(after_truth)).sum())
            progress_counts['non_goal_predicted_zero']+=int((before<=1e-6).sum())
            for name,mask in distance_strata(before_truth):
                progress_layers[name]['executed_edges']+=int(mask.sum())
                progress_layers[name]['true_optimal_edges']+=int((optimal&mask).sum())
                progress_layers[name]['predicted_decrease']+=int((decreasing&mask).sum())
                progress_layers[name]['entered_dead_edges']+=int((mask&np.isfinite(before_truth)&~np.isfinite(after_truth)).sum())
            for j,margin in zip(ids,drop): descent[j].append(float(margin))
            if edge_records is not None:
                child_truth=np.full(children.shape,np.inf,dtype=np.float32)
                child_truth[valid]=corpus.distance(children[valid],np.broadcast_to(goal[ids,None],children.shape)[valid])
                edge_records.append(dict(query_index=ids.copy(),step=steps[ids].copy(),
                    source=current[ids].copy(),goal=goal[ids].copy(),children=children.copy(),choice=choices.copy(),
                    parent_prediction=before.copy(),child_predictions=scores.copy(),
                    parent_distance=before_truth.copy(),child_distances=child_truth,
                    next_is_goal=next_states==goal[ids],
                    next_repeated=np.asarray([int(s) in seen[j] for j,s in zip(ids,next_states)])))
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
                                 entered_unreachable=bool(dead[i]),actions=traces[i],
                                 predicted_distance_drops=descent[i] if policy=='q' else None))
    per_scene={}
    if hasattr(corpus,'scenes'):
        scene_ids=corpus.locate(rows[:,0])
        per_scene={str(sid):dict(count=int((scene_ids==sid).sum()),solved=int(solved[scene_ids==sid].sum()),
                    success_rate=float(solved[scene_ids==sid].mean())) for sid in np.unique(scene_ids)}
    progress=None
    if policy=='q':
        drops=np.asarray([d for path in descent for d in path])
        all_down=np.asarray([all(d>1e-6 for d in path) for path in descent])
        progress=dict(counts=progress_counts,strict_descent_trajectory_count=int(all_down.sum()),
            strict_descent_success_count=int((all_down&solved).sum()),
            non_descent_trajectory_count=int((~all_down).sum()),
            predicted_drop_quantiles={str(q):float(np.quantile(drops,q)) for q in (0,.1,.5,.9,1)} if len(drops) else {},
            mean_abs_unit_descent_residual=float(np.abs(drops-1).mean()) if len(drops) else None,
            executed_edge_optimal_rate=progress_counts['true_optimal_edges']/len(drops) if len(drops) else None,
            predicted_strict_descent_edge_rate=progress_counts['predicted_decrease']/len(drops) if len(drops) else None,
            executed_edges_by_true_distance=progress_layers,
            note='Observed policy trajectories only, after action selection. A nondecreasing chosen edge also means no legal successor has a strictly lower predicted distance under this greedy policy. Strict predicted descent alone does not prove goal completion or true optimality; full rollouts remain the acceptance metric.')
    return dict(count=len(rows),unique_state_goal_pairs=len(np.unique(rows,axis=0)),
                solved=int(solved.sum()),success_rate=float(solved.mean()) if len(rows) else None,
                optimal=int(optimum.sum()),optimal_rate=float(optimum.mean()) if len(rows) else None,cycles=int(cycles.sum()),
                entered_unreachable=int(dead.sum()),no_legal_action=int(no_action.sum()),
                budget_exhausted=int(active.sum()),horizon=horizon,illegal_actions=0,
                mean_excess_cost_on_success=float((steps[solved]-exact[solved]).mean()) if solved.any() else None,
                mean_cost_ratio_on_success=float((steps[solved]/np.maximum(exact[solved],1)).mean()) if solved.any() else None,
                by_distance={str(d):dict(count=int((exact==d).sum()),solved=int(solved[exact==d].sum()),optimal=int(optimum[exact==d].sum())) for d in np.unique(exact)},
                by_scene=per_scene,examples=examples,predicted_progress=progress)


def planning_ids(corpus,rows,truth,count,rng,distance_bins=None):
    """Unique starts/queries; share the budget across scenes then distance layers."""
    pool=np.flatnonzero(np.isfinite(truth) & (truth>0))
    # Prepared distance-stratified evaluation rows may repeat a physical pair.
    # Keep their original weights for regression/checkpoint selection, but not
    # for deterministic whole-plan trials. Preserve first-occurrence order so
    # already unique cohorts retain their original planning queries.
    if len(pool):
        _,first=np.unique(rows[pool],axis=0,return_index=True)
        pool=pool[np.sort(first)]
    if len(pool)<=count: return pool
    scene=corpus.locate(rows[pool,0]) if hasattr(corpus,'scenes') else np.zeros(len(pool),int)
    groups=[pool[scene==sid] for sid in np.unique(scene)]
    chosen=[]
    for group in groups:
        if distance_bins is None:
            layers=[rng.permutation(group[truth[group]==d]).tolist() for d in np.unique(truth[group])]
        else:
            edges=np.asarray(distance_bins,dtype=float)
            if edges.ndim!=1 or not len(edges) or not np.isfinite(edges).all() or (np.diff(edges)<=0).any() or edges[0]<=0:
                raise ValueError('Positive increasing finite planning distance bounds required')
            if truth[group].max()>edges[-1]: raise ValueError('Planning bins omit a finite distance')
            bins=np.searchsorted(edges,truth[group])
            layers=[rng.permutation(group[bins==b]).tolist() for b in np.unique(bins)]
            # A small per-layout quota must sample all distance ranges across
            # layouts, rather than always selecting each layout's nearest pair.
            layers=[layers[i] for i in rng.permutation(len(layers))]
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
    planning_bins=config['evaluation'].get('planning_distance_bins')
    protocol=('unique_ordered_state_goal_pairs_distance_bins_v3' if planning_bins is not None
              else 'unique_ordered_state_goal_pairs_v2')
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
                    bellman=unit_cost_consistency(pred[ids],scores[ids],optimal,valid[ids]),
                    directionality=dict(predicted_mean_absolute_asymmetry=float(np.abs(pred-reverse).mean()),
                        both_reachable=int((finite & np.isfinite(reverse_truth)).sum()),
                        forward_only=int((finite & ~np.isfinite(reverse_truth)).sum()),
                        reverse_finite_mae=float(np.abs(reverse[np.isfinite(reverse_truth)]-reverse_truth[np.isfinite(reverse_truth)]).mean()) if np.isfinite(reverse_truth).any() else None,
                        reverse_scope='reverse target may lie outside the successful-terminal training target domain'),
                    unreachable=reachability(pred,truth,threshold,config['training']['unreachable_target']))
        result['coordinate_statistics']=coordinate_statistics(cache.q(rows[:,0]).cpu().numpy(),cache.q(rows[:,1]).cpu().numpy(),
            truth,pred,model.cap,getattr(model,'output_activation','sigmoid'))
        if 'solution_paths_'+name in data:
            from .solution_structure import path_metrics
            result['solution_paths']=path_metrics(cache,corpus,data['solution_paths_'+name])
        result['bellman']['by_true_distance']={name:dict(count=int(mask.sum()),
            shortest_action_rate=float(optimal[mask,chosen[mask]].mean()),
            **unit_cost_consistency(pred[ids][mask],scores[ids][mask],optimal[mask],valid[ids][mask]))
            for name,mask in distance_strata(truth[ids]) if mask.any()}
        if exposure is not None:
            seen_source=exposure[rows[:,0]]>0; seen_goal=exposure[rows[:,1]]>0
            result['actual_gradient_endpoint_exposure']=dict(both_seen=int((seen_source & seen_goal).sum()),
                one_unseen=int((seen_source ^ seen_goal).sum()),both_unseen=int((~seen_source & ~seen_goal).sum()),
                both_seen_finite_mae=float(np.abs(pred[finite & seen_source & seen_goal]-truth[finite & seen_source & seen_goal]).mean()) if (finite & seen_source & seen_goal).any() else None,
                any_unseen_finite_mae=float(np.abs(pred[finite & ~(seen_source & seen_goal)]-truth[finite & ~(seen_source & seen_goal)]).mean()) if (finite & ~(seen_source & seen_goal)).any() else None,
                counter_semantics='Legacy field name: exposure counts state inputs participating in loss computation, not proven nonzero parameter gradients. Identity pairs, one-candidate ranking or saturation may have zero gradient.')
        if planning:
            rng=np.random.default_rng(config['seed']+len(name))
            selected=rows[planning_ids(corpus,rows,truth,config['evaluation']['rollout_count'],rng,planning_bins)]
            unique_selected=len(np.unique(selected,axis=0))
            if unique_selected!=len(selected): raise ValueError('Duplicate physical planning query')
            eligible=np.isfinite(truth)&(truth>0)
            result['planning_query_selection']=dict(protocol=protocol,distance_bin_upper_bounds=planning_bins,
                eligible_rows=int(eligible.sum()),unique_eligible_pairs=len(np.unique(rows[eligible],axis=0)),
                maximum_queries=config['evaluation']['rollout_count'],selected_queries=len(selected),
                unique_selected_pairs=unique_selected,
                scope='Exact ordered(S,G) duplicates removed before scene/distance allocation. Different complete goals at one source remain distinct. Regression/ranking row weights and validation checkpoint selection are unchanged.')
            result['planning']={policy:rollout(corpus,cache,selected,config['evaluation']['horizon'],policy) for policy in ('q','oracle','heuristic','immediate_cost_only')}
        results[name]=result
    return dict(reachability_threshold=threshold,cohorts=results,
                planning_protocol=protocol if planning else None)


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
    parser.add_argument('--planning-distance-bins',nargs='+',type=float)
    args=parser.parse_args()
    if args.out.exists(): raise FileExistsError('Keep previous frozen evaluations; choose a new output')
    saved=torch.load(args.checkpoint,map_location=args.device,weights_only=False)
    config=saved['config']; torch.set_num_threads(config['training']['cpu_threads'])
    if args.planning_distance_bins is not None:
        config['evaluation']['planning_distance_bins']=args.planning_distance_bins
    if config['training'].get('solution_structure'):
        from .solution_structure import prepare as prepare_structure
        corpus,data,_=prepare_structure(config,args.run/'data')
    else:
        corpus,data,_=load(config,args.run/'data')
    model=StateEncoder(input_dim=saved['input_dim'],**config['model']).to(args.device)
    model.load_state_dict(saved['model']); model.requires_grad_(False)
    result=evaluate(model,corpus,data,config,cohorts=args.cohorts,exposure=saved['exposure'])
    atomic_json(args.out,dict(checkpoint=str(args.checkpoint),step=saved['step'],source_commit=saved['source_commit'],
                             evaluation_override=dict(planning_distance_bins=args.planning_distance_bins),metrics=result))
    print(json.dumps(dict(step=saved['step'],cohorts={name:row['planning']['q']['success_rate'] for name,row in result['cohorts'].items()})),flush=True)


if __name__=='__main__': main()
