"""Teacher-only geometry probes; these vectors are never inference features."""
import argparse
import csv
import gzip
import json
from pathlib import Path
import numpy as np
from grounded_llm.core.files import atomic_json
from .relation_data import load


def _popcount(values):
    table=np.asarray([i.bit_count() for i in range(256)],dtype=np.uint8)
    return int(table[values.view(np.uint8)].sum(dtype=np.int64))


def isometry_feature_view(corpus,task,variant,seed):
    """One query-independent input isometry, with transported candidate order."""
    from types import SimpleNamespace
    if not 0<=variant<8: raise ValueError('Expected one of eight fixed diagnostic variants')
    mapping={}
    if task=='rush_hour':
        for sid,scene in enumerate(corpus.scenes):
            used=np.array([ord(piece[0])-ord('A') for piece in scene.env.pieces[1:]])
            permutation=np.arange(26)
            if variant and len(used)>1:
                shuffled=np.random.default_rng(seed+104729*variant+sid).permutation(used)
                if np.array_equal(shuffled,used): shuffled=np.roll(shuffled,1)
                permutation[used]=shuffled
            mapping[sid]=permutation
    elif task=='token_swapping':
        permutation=np.arange(corpus.n)
        if variant:
            permutation=np.random.default_rng(seed+104729*variant).permutation(corpus.n)
            if np.array_equal(permutation,np.arange(corpus.n)): permutation=np.roll(permutation,1)
        mapping[0]=permutation
    elif task!='sokoban': raise ValueError('Unsupported exact-pair task')
    def features(ids):
        ids=np.asarray(ids,dtype=np.int64); original=corpus.features(ids)
        if task=='sokoban':
            values=original.reshape(len(ids),4,10,10)
            values=np.rot90(values,variant%4,axes=(-2,-1))
            if variant//4: values=np.flip(values,axis=-1)
            return np.ascontiguousarray(values).reshape(len(ids),-1)
        if task=='token_swapping':
            values=original.reshape(len(ids),corpus.n,corpus.n); result=np.empty_like(values)
            result[:,:,mapping[0]]=values
        else:
            values=original.reshape(len(ids),26,36); result=np.empty_like(values)
            scenes=corpus.locate(ids)
            for sid in np.unique(scenes):
                mask=scenes==sid
                result[mask]=values[mask][:,np.argsort(mapping[int(sid)]),:]
        return result.reshape(len(ids),-1)
    return SimpleNamespace(features=features),{str(sid):permutation.tolist() for sid,permutation in mapping.items()}


def isometry_probe(model,corpus,data,config,cohorts,count):
    from .relation_evaluate import QCache,planning_ids,rollout,safe_scores
    result={}; archive={}; mappings={}
    for name in cohorts:
        rows=data['tasks_'+name]; truth=corpus.distance(rows[:,0],rows[:,1])
        rows=rows[planning_ids(corpus,rows,truth,count,np.random.default_rng(config['seed']+len(name)))]; archive['queries_'+name]=rows
        truth=corpus.distance(rows[:,0],rows[:,1]); children=corpus.successors(rows[:,0]); valid=children>=0
        child_truth=np.full(children.shape,np.inf,dtype=np.float32)
        child_truth[valid]=corpus.distance(children[valid],np.broadcast_to(rows[:,1,None],children.shape)[valid])
        optimal=valid & (child_truth+1==truth[:,None]); index=np.arange(len(rows)); result[name]={}
        base_prediction=base_choice=None
        for variant in range(8):
            view,mapping=isometry_feature_view(corpus,config['task'],variant,config['seed'])
            mappings[str(variant)]=mapping
            cache=QCache(model,view,'cpu'); predicted=cache.scores(rows[:,0],rows[:,1]); scores=safe_scores(cache,children,rows[:,1])
            choice=scores.argmin(axis=1)
            if variant==0: base_prediction,base_choice=predicted,choice
            plan=rollout(corpus,cache,rows,config['evaluation']['horizon'],'q')
            result[name][str(variant)]=dict(finite_mae=float(np.abs(predicted-truth).mean()),
                prediction_shift_mae_from_identity=float(np.abs(predicted-base_prediction).mean()),
                prediction_shift_max_from_identity=float(np.abs(predicted-base_prediction).max()),
                chosen_action_agreement_with_identity=float((choice==base_choice).mean()),
                true_optimal_action_rate=float(optimal[index,choice].mean()),
                planning={key:plan[key] for key in ('count','unique_state_goal_pairs','solved','success_rate','optimal','optimal_rate','cycles','entered_unreachable')})
            archive['predictions_'+name+'_'+str(variant)]=predicted
            archive['choices_'+name+'_'+str(variant)]=choice
    return dict(cohorts=result,channel_mappings=mappings,
        condition=('Validation-only input-isometry diagnosis. Sokoban variant k rotates the complete board90*(k mod4) degrees counterclockwise, then reflects columns for k>=4; all four channels move together. Rush Hour variants jointly rename only existing ordinary vehicle labels, preserving A. Token variants jointly rename token identities on the fixed labelled tree. Variant0 is identity. Each feature transform is independent of query goal. Physics, oracle distances, complete targets and candidate multiplicity are preserved. Candidate order is transported from the original query to isolate encoding; this need not match a freshly enumerated transformed board tie order. These are diagnostic variations, not acceptance results or training augmentation.')),archive


def near_endpoint_cases(corpus,partition,seed,count):
    """Fixed positive-distance validation cases, never extra training rows."""
    from grounded_llm.tasks.token_swapping.relation_data import rank
    if hasattr(corpus,'scenes') or count<5 or len(partition)!=corpus.size:
        raise ValueError('Token endpoint partition and at least five near queries required')
    pools={fold:np.flatnonzero(partition==fold) for fold in (0,1)}
    if any(not len(pool) for pool in pools.values()): raise ValueError('Both training and validation endpoint pools required')
    rng=np.random.default_rng(seed+1009); groups={}
    for name,source_fold,goal_fold in (('near_unseen_start_validation',1,0),('near_unseen_goal_validation',0,1),('near_unseen_both_validation',1,1)):
        pieces=[]
        for distance in range(1,6):
            quota=count//5+(distance<=count%5); shell=np.flatnonzero(corpus.distances==distance)
            if not len(shell): raise ValueError(('Missing exact near-distance shell',distance))
            candidates=np.empty((0,2),dtype=np.int64)
            for _ in range(64):
                missing=quota-len(candidates)
                if missing<=0: break
                goal=rng.choice(pools[goal_fold],max(1024,missing*24))
                relative=corpus.states[rng.choice(shell,len(goal))]
                source=rank(np.take_along_axis(corpus.states[goal],relative,axis=1))
                keep=partition[source]==source_fold
                candidates=np.unique(np.concatenate([candidates,np.column_stack([source[keep],goal[keep]])]),axis=0)
            if len(candidates)<quota: raise ValueError(('Insufficient unique held-endpoint near relations',name,distance))
            rows=candidates[rng.choice(len(candidates),quota,replace=False)]
            if not (corpus.distance(rows[:,0],rows[:,1])==distance).all(): raise ValueError('Relative near-distance proposal changed label')
            pieces.append(rows)
        rows=np.concatenate(pieces); rng.shuffle(rows)
        if len(np.unique(rows,axis=0))!=count: raise ValueError('Repeated near validation query')
        if not ((partition[rows[:,0]]==source_fold)&(partition[rows[:,1]]==goal_fold)).all(): raise ValueError('Near endpoint fold changed')
        groups[name]=rows
    return groups


def policy_boundaries(records):
    """Actual policy occupancy and cycle entrances; labels never select actions."""
    if not records: return {},{}
    edges={key:np.concatenate([batch[key] for batch in records]) for key in records[0]}
    valid=edges['children']>=0; index=np.arange(len(valid)); chosen=edges['choice']
    truth=edges['parent_distance']; child_truth=edges['child_distances']
    optimal=valid & np.isfinite(child_truth) & (child_truth+1==truth[:,None])
    optimal&=np.isfinite(truth)[:,None]
    score=edges['child_predictions']; predicted=edges['parent_prediction']
    drop=predicted-score[index,chosen]; selected_optimal=optimal[index,chosen]
    best_optimal=np.where(optimal,score,np.inf).min(axis=1)
    best_other=np.where(valid & ~optimal,score,np.inf).min(axis=1)
    margin=np.full(len(truth),np.nan,dtype=np.float32)
    np.subtract(best_other,best_optimal,out=margin,where=np.isfinite(best_other)&np.isfinite(best_optimal))
    non_descent=drop<=1e-6
    first=np.zeros(len(truth),dtype=bool)
    # Records are generated in execution order, interleaved across queries.
    _,first_ids=np.unique(edges['query_index'][non_descent],return_index=True)
    first[np.flatnonzero(non_descent)[first_ids]]=True
    def quantiles(values):
        return {str(q):float(np.quantile(values,q)) for q in (0,.1,.5,.9,1)} if len(values) else {}
    def describe(mask):
        finite=mask & np.isfinite(truth); comparable=finite & np.isfinite(margin)
        return dict(executed_edges=int(mask.sum()),
            unique_state_goal_pairs=len(np.unique(np.column_stack([edges['source'][mask],edges['goal'][mask]]),axis=0)),
            represented_queries=len(np.unique(edges['query_index'][mask])),
            true_distance_quantiles=quantiles(truth[finite]),
            finite_distance_mae=float(np.abs(predicted[finite]-truth[finite]).mean()) if finite.any() else None,
            true_optimal_action_rate=float(selected_optimal[mask].mean()) if mask.any() else None,
            predicted_non_descent_rate=float(non_descent[mask].mean()) if mask.any() else None,
            chosen_predicted_drop_quantiles=quantiles(drop[mask]),
            comparable_optimal_vs_other_edges=int(comparable.sum()),
            optimal_prediction_margin_quantiles=quantiles(margin[comparable]),
            optimal_prediction_margin_nonpositive_rate=float((margin[comparable]<=1e-6).mean()) if comparable.any() else None,
            optimal_prediction_margin_abs_le_half_rate=float((np.abs(margin[comparable])<=.5).mean()) if comparable.any() else None,
            entered_proven_unreachable=int((mask & np.isfinite(truth) & ~np.isfinite(child_truth[index,chosen])).sum()))
    from .relation_evaluate import distance_strata
    subsets=dict(all_executed=np.ones(len(truth),dtype=bool),first_non_descent=first,
        all_non_descent=non_descent,cycle_closing=edges['next_repeated'] & ~edges['next_is_goal'])
    result={name:dict(**describe(mask),by_true_distance={layer:describe(mask & layer_mask)
        for layer,layer_mask in distance_strata(truth) if (mask & layer_mask).any()}) for name,mask in subsets.items()}
    result['interpretation']='Execution-weighted records include repeated visits across queries; unique pair counts are also reported. First non-descent is the first visited state with no strictly lower predicted legal successor. A cycle-closing action is inspected after selection. Positive optimal-vs-other margin favors the best teacher-optimal child; ties depend on the unchanged candidate order. No evaluation record is added to training.'
    return result,edges


def pair_incompatibility_gaps(corpus,rows):
    """Positive gap means two undirected pairs cannot share a tight Linf axis."""
    a,b=rows[:,0],rows[:,1]; length=corpus.distance(a,b)
    cross_same=corpus.distance(a[:,None],a[None,:])+corpus.distance(b[:,None],b[None,:])
    cross_opposite=corpus.distance(a[:,None],b[None,:])+corpus.distance(b[:,None],a[None,:])
    return length[:,None]+length[None,:]-np.maximum(cross_same,cross_opposite)


def linf_witness(corpus,config,count):
    """Exact finite metric obstruction, independent of learned encoder weights."""
    if config['task']!='token_swapping' or count<2: raise ValueError('At least two fixed-tree Token pairs required')
    from grounded_llm.tasks.token_swapping.relation_data import rank
    seed=config['seed']+721; rng=np.random.default_rng(seed)
    diameter=int(corpus.distances.max()); far=np.flatnonzero(corpus.distances==diameter)
    source=rng.choice(corpus.size,count,replace=False)
    relative=corpus.states[rng.choice(far,count)]
    goals=rank(np.take_along_axis(corpus.states[source],np.argsort(relative,axis=1),axis=1))
    rows=np.column_stack([source,goals])
    if not (corpus.distance(source,goals)==diameter).all(): raise ValueError('Witness is not at the exact diameter')
    gap=pair_incompatibility_gaps(corpus,rows); results={}
    for margin in (1,2,4,6,8):
        adjacent=gap>=margin; np.fill_diagonal(adjacent,False); best=[]
        for trial in range(16):
            candidates=rng.permutation(count); selected=[]
            if trial==0: candidates=candidates[np.argsort(-adjacent[candidates].sum(axis=1),kind='stable')]
            while len(candidates):
                chosen=int(candidates[0]); selected.append(chosen); candidates=candidates[adjacent[chosen,candidates]]
            if len(selected)>len(best): best=selected
        if len(best)<2: continue
        verified=gap[np.ix_(best,best)].copy(); np.fill_diagonal(verified,np.inf)
        minimum=float(verified.min())
        if minimum<margin: raise ValueError('Witness clique failed exact recheck')
        results[str(margin)]=dict(pairs=len(best),minimum_verified_gap=minimum,
            uniform_absolute_error_lower_bound_for_fewer_dimensions=minimum/4,
            witness_pair_ids=rows[best].tolist())
    return dict(seed=seed,diameter=diameter,proposed_pairs=count,results=results,
        proof='For pairs (a,b),(c,d) with lengths D1,D2, a shared tight absolute-max coordinate requires D1+D2 <= max(d(a,c)+d(b,d),d(a,d)+d(b,c)). Otherwise each orientation contradicts a cross-distance bound. With uniform additive error epsilon, the right side can gain at most4epsilon. A positive-gap clique of M pairs requires at least M coordinates for an exact fit; with fewer dimensions its minimum gap/4 bounds the uniform absolute error below.',
        scope='Exact finite lower-bound witness for independent Q and L-infinity only. Not an average-error bound, not a learned-map result, and not a proof that greedy80 percent is impossible. Oracle is used offline only; all witness endpoints and their cross-distances are part of the bounded domain.')


def stream_statistics(corpus,data,run,out):
    """Actual saved batches, not proposed sampler quotas or whole-graph priors."""
    if not hasattr(corpus,'scenes') or any(len(s.goals)>32 for s in corpus.scenes):
        raise ValueError('This exact goal-bit coverage probe is for the Sokoban goal bank')
    from .relation_data import keys
    reserved=data['reserved_relations']
    def forbidden(values):
        if not len(reserved): return np.zeros(values.shape,dtype=bool)
        at=np.searchsorted(reserved,values)
        return (at<len(reserved)) & (reserved[np.minimum(at,len(reserved)-1)]==values)
    columns=np.full(corpus.size,-1,dtype=np.int8)
    point_index=np.full(corpus.size,-1,dtype=np.int32)
    point_index[data['pairs_train'][:,0]]=np.arange(len(data['pairs_train']))
    eligible=np.zeros(corpus.size,dtype=np.uint32); seen=np.zeros_like(eligible)
    for sid,scene in enumerate(corpus.scenes):
        offset=corpus.offsets[sid]; columns[scene.goals+offset]=np.arange(len(scene.goals))
        if scene.metadata['split']!='train': continue
        nt=scene.metadata['training_goals']; blocked=scene.goals[nt:]
        safe=~np.isin(np.arange(len(scene.states)),blocked) & ~np.isin(scene.children,blocked).any(axis=1)
        for j,g in enumerate(scene.goals[:nt]):
            s=np.flatnonzero(safe & np.isfinite(scene.distances[j]) & (scene.distances[j]>0))+offset
            children=corpus.successors(s); valid=children>=0; target=np.full(len(s),g+offset)
            bad=forbidden(keys(s,target,corpus.size))
            relation=np.full(children.shape,-1,dtype=np.int64)
            relation[valid]=keys(children[valid],np.broadcast_to(target[:,None],children.shape)[valid],corpus.size)
            bad|=forbidden(relation).any(axis=1)
            eligible[s[~bad]]|=np.uint32(1)<<np.uint32(j)
    files=sorted(Path(run).glob('samples/attempt_*/*.npz'))
    if not files: raise ValueError('No actual saved training sample shards')
    csv_path=Path(out).with_suffix('.csv.gz'); csv_path.parent.mkdir(parents=True,exist_ok=True)
    fields=['attempt','step','point_rows','point_finite','point_infinite','rank_rows','legal_candidates',
            'candidate_finite','candidate_infinite','boundary_decisions','optimal_candidates','rows_without_optimal',
            'point_initial_row_finite','initial_finite_to_actual_unreachable','initial_unreachable_to_actual_finite',
            'point_initial_row_near_1_5','initial_near_to_actual_zero','initial_near_to_actual_near_1_5','initial_near_to_actual_far','initial_near_to_actual_unreachable']
    intervals=[(0,0),(1,1),(2,2),(3,5),(6,10),(11,20),(21,40),(41,80),(81,np.inf)]
    bin_names=[f'{a}_{b}' for a,b in intervals]
    fields += ['point_distance_'+name for name in bin_names]+['rank_distance_'+name for name in bin_names]
    totals={key:0 for key in fields if key not in ('attempt','step')}
    proportions={'point_finite':[],'boundary_decisions':[],'candidate_infinite':[]}
    steps=[]; attempts=set()
    with gzip.open(csv_path,'wt',newline='',compresslevel=1) as stream:
        writer=csv.DictWriter(stream,fieldnames=fields); writer.writeheader()
        for path in files:
            with np.load(path) as archive: x={key:archive[key] for key in ('point_step','point_distance','pairs','rank_step','tasks','children','child_distance','parent_distance','optimal')}
            children=x['children']; valid=children>=0
            source_rows=point_index[x['pairs'][:,0]]
            if (source_rows<0).any(): raise ValueError(('Point source outside original training rows',path))
            initial=data['pair_distances_train'][source_rows]
            if not np.array_equal(children,corpus.successors(x['tasks'][:,0])): raise ValueError(('Missing legal training candidate',path))
            if not np.array_equal(x['point_distance'],corpus.distance(x['pairs'][:,0],x['pairs'][:,1])): raise ValueError(('Point labels changed scale or identity',path))
            if not np.array_equal(x['child_distance'][valid],corpus.distance(children[valid],np.broadcast_to(x['tasks'][:,1,None],children.shape)[valid])): raise ValueError(('Child labels changed scale or identity',path))
            parent=corpus.distance(x['tasks'][:,0],x['tasks'][:,1])
            if not np.array_equal(x['parent_distance'],parent): raise ValueError(('Parent labels changed scale or identity',path))
            if not np.array_equal(x['optimal'],valid & np.isfinite(x['child_distance']) & (x['child_distance']+1==parent[:,None])):
                raise ValueError(('Optimal candidate mask changed',path))
            goals=columns[x['tasks'][:,1]]
            if (goals<0).any(): raise ValueError('Unknown complete training goal')
            np.bitwise_or.at(seen,x['tasks'][:,0],np.left_shift(np.uint32(1),goals.astype(np.uint32)))
            p_steps,p_start,p_count=np.unique(x['point_step'],return_index=True,return_counts=True)
            r_steps,r_start,r_count=np.unique(x['rank_step'],return_index=True,return_counts=True)
            if not np.array_equal(p_steps,r_steps): raise ValueError('Point and ranking batch identities differ')
            for step,p0,np_,r0,nr in zip(p_steps,p_start,p_count,r_start,r_count):
                pd=x['point_distance'][p0:p0+np_]; cd=x['child_distance'][r0:r0+nr]; mask=valid[r0:r0+nr]
                rd=x['parent_distance'][r0:r0+nr]; optimum=x['optimal'][r0:r0+nr] & mask
                initial_pd=initial[p0:p0+np_]; initial_finite=np.isfinite(initial_pd)
                initial_near=initial_finite & (initial_pd>=1) & (initial_pd<=5)
                finite=np.isfinite(cd) & mask; dead=~np.isfinite(cd) & mask
                row=dict(attempt=path.parent.name,step=int(step),point_rows=int(np_),point_finite=int(np.isfinite(pd).sum()),
                    point_infinite=int((~np.isfinite(pd)).sum()),rank_rows=int(nr),legal_candidates=int(mask.sum()),
                    candidate_finite=int(finite.sum()),candidate_infinite=int(dead.sum()),
                    boundary_decisions=int((finite.any(axis=1)&dead.any(axis=1)).sum()),
                    optimal_candidates=int(optimum.sum()),rows_without_optimal=int((~optimum.any(axis=1)).sum()),
                    point_initial_row_finite=int(initial_finite.sum()),
                    initial_finite_to_actual_unreachable=int((initial_finite & ~np.isfinite(pd)).sum()),
                    initial_unreachable_to_actual_finite=int((~initial_finite & np.isfinite(pd)).sum()),
                    point_initial_row_near_1_5=int(initial_near.sum()),
                    initial_near_to_actual_zero=int((initial_near & (pd==0)).sum()),
                    initial_near_to_actual_near_1_5=int((initial_near & (pd>=1) & (pd<=5)).sum()),
                    initial_near_to_actual_far=int((initial_near & np.isfinite(pd) & (pd>5)).sum()),
                    initial_near_to_actual_unreachable=int((initial_near & ~np.isfinite(pd)).sum()))
                for name,(a,b) in zip(bin_names,intervals):
                    row['point_distance_'+name]=int((np.isfinite(pd)&(pd>=a)&(pd<=b)).sum())
                    row['rank_distance_'+name]=int((np.isfinite(rd)&(rd>=a)&(rd<=b)).sum())
                writer.writerow(row)
                for key in totals: totals[key]+=row[key]
                proportions['point_finite'].append(row['point_finite']/row['point_rows'])
                proportions['boundary_decisions'].append(row['boundary_decisions']/row['rank_rows'])
                proportions['candidate_infinite'].append(row['candidate_infinite']/row['legal_candidates'])
                steps.append(int(step)); attempts.add(path.parent.name)
    if np.bitwise_and(seen,np.bitwise_not(eligible)).any(): raise ValueError('Ranking relation outside the declared training closure')
    available=_popcount(eligible); actual=_popcount(seen)
    return dict(actual_batches=len(steps),first_step=min(steps),last_step=max(steps),attempts=sorted(attempts),
        per_batch_csv=str(csv_path),totals=totals,
        point_finite_fraction=totals['point_finite']/totals['point_rows'],
        boundary_decision_fraction=totals['boundary_decisions']/totals['rank_rows'],
        unique_ranking_source_goal_pairs=actual,eligible_ranking_source_goal_pairs=available,
        ranking_relation_coverage=actual/available,
        initial_row_label_comparison='Initial labels belong to the saved source-row placeholder goals used by the baseline strata, not necessarily to the online-resampled actual goal. These transition counts quantify that mismatch without inferring which source draws came from the broad branch.',
        all_actual_point_and_child_labels_equal_physical_oracle=True,all_legal_candidates_preserved=True,
        per_batch_fractions={key:dict(mean=float(np.mean(values)),quantiles={str(q):float(np.quantile(values,q)) for q in (0,.1,.5,.9,1)}) for key,values in proportions.items()},
        note='Each CSV row is one saved actual optimizer batch. Repeated steps across attempts are retained explicitly. No historical per-step gradient was recorded or reconstructed here.')


def batch_gradient_probe(model,corpus,batch,config):
    """Loss components on one actual batch at fixed weights, without an update."""
    import torch
    from torch.nn import functional as F
    from grounded_llm.core.qmap.model import soft_rank_loss
    device=next(model.parameters()).device; model.eval()
    pairs=batch['pairs']; tasks=batch['tasks']; children=batch['children']; valid=children>=0
    ids=np.concatenate([pairs[:,0],pairs[:,1],children[valid],tasks[:,1]])
    unique,inverse=np.unique(ids,return_inverse=True)
    vectors=model.encode(torch.as_tensor(corpus.features(unique),device=device))
    b=len(pairs); nchild=int(valid.sum()); options=config['training']
    predictions=model.distance(vectors[inverse[:b]],vectors[inverse[b:2*b]])
    finite=np.isfinite(batch['point_distance']); mask=torch.as_tensor(finite,device=device)
    targets=np.where(finite,batch['point_distance'],options['unreachable_target']).astype(np.float32)
    per_point=F.smooth_l1_loss(predictions,torch.as_tensor(targets,device=device),reduction='none')
    child=vectors[inverse[2*b:2*b+nchild]]; goal=vectors[inverse[2*b+nchild:]]
    legal=torch.as_tensor(valid,device=device); scores=torch.zeros(children.shape,device=device)
    scores[legal]=1+model.distance(child,goal[:,None].expand(-1,children.shape[1],-1)[legal])
    truth=np.where(np.isfinite(batch['child_distance']),batch['child_distance'],options['unreachable_target'])
    rank=soft_rank_loss(scores,torch.as_tensor(1+truth,device=device),legal,options['rank_temperature'])
    losses=dict(finite_point=per_point[mask].sum()/b,unreachable_point=per_point[~mask].sum()/b,
                ranking=options['rank_weight']*rank)
    parameters=list(model.parameters()); gradients={}
    for name,loss in losses.items():
        parts=torch.autograd.grad(loss,parameters,retain_graph=True,allow_unused=True)
        gradients[name]=torch.cat([(torch.zeros_like(p) if g is None else g).reshape(-1) for p,g in zip(parameters,parts)]).detach()
    total=sum(gradients.values()); squared=float(total.square().sum())
    metrics={}
    for name,g in gradients.items():
        norm=float(g.norm()); dot=float((g*total).sum())
        metrics[name]=dict(weighted_loss=float(losses[name].detach()),gradient_l2=norm,
            cosine_to_total=dot/(norm*squared**.5) if norm and squared else None,
            projection_on_total=dot/squared if squared else None)
    result=dict(point_count=b,finite_count=int(finite.sum()),components=metrics,total_gradient_l2=squared**.5,
        component_cosines={a+':'+b:float(torch.nn.functional.cosine_similarity(gradients[a],gradients[b],dim=0))
                           for a,b in (('finite_point','unreachable_point'),('finite_point','ranking'),('unreachable_point','ranking'))})
    with torch.no_grad():
        residual=(predictions-torch.as_tensor(targets,device=device)).abs()
        derivative=residual.clamp_max(1)
        result['point_huber_abs_derivative_mean']={name:float(derivative[m].mean()) if m.any() else None
            for name,m in (('finite',mask),('unreachable',~mask))}
        positive=mask & (torch.as_tensor(targets,device=device)>0)
        result['positive_finite_predicted_zero_fraction']=float((predictions[positive]<=1e-6).float().mean()) if positive.any() else None
    return result


def gradient_probe(model,corpus,run,config,checkpoint_step,count):
    # Frozen checkpoint on recorded batches ending at its step. This is not a
    # reconstruction of optimizer-time gradients and never changes weights.
    rows=[]; first=max(1,checkpoint_step-count+1)
    for path in sorted(Path(run).glob('samples/attempt_*/*.npz')):
        with np.load(path) as archive:
            steps=archive['point_step']; selected=np.unique(steps[(steps>=first)&(steps<=checkpoint_step)])
            if not len(selected): continue
            x={key:archive[key] for key in ('pairs','tasks','children','point_distance','child_distance','rank_step')}
        for step in selected:
            point=steps==step; rank=x['rank_step']==step
            batch={key:x[key][point if key in ('pairs','point_distance') else rank]
                   for key in ('pairs','tasks','children','point_distance','child_distance')}
            rows.append(dict(attempt=path.parent.name,batch_step=int(step),
                             **batch_gradient_probe(model,corpus,batch,config)))
    if len(rows)!=count: raise ValueError(('Expected unique contiguous actual batches',first,checkpoint_step,len(rows)))
    return dict(checkpoint_step=checkpoint_step,actual_batches=rows,
        note='Frozen weights on the specified recorded actual batches; no optimizer step. Components use the original full-batch denominator and rank weight. Gradient norms are not additive shares; signed projections onto the total sum to one and may be negative. Historical optimizer-time gradients were not recorded.')


def encoder_axes(model,corpus,data,config,cohorts):
    """Observed max-coordinate usage is diagnostic, not an impossibility proof."""
    import torch
    from .relation_evaluate import QCache
    model.eval(); cache=QCache(model,corpus,next(model.parameters()).device)
    rng=np.random.default_rng(config['seed']+219); result={}
    for name in cohorts:
        rows=data['tasks_'+name]
        if len(rows)>config['evaluation']['decision_count']:
            rows=rows[rng.choice(len(rows),config['evaluation']['decision_count'],replace=False)]
        rows=rows[rows[:,0]!=rows[:,1]]
        with torch.no_grad():
            source=cache.q(rows[:,0]).cpu().numpy(); goal=cache.q(rows[:,1]).cpu().numpy()
        difference=source-goal
        if config['model']['metric']=='linf': difference=np.abs(difference)
        elif config['model']['metric']!='directed_max': raise ValueError('Axis probe requires max-coordinate metric')
        counts=np.bincount(difference.argmax(axis=1),minlength=source.shape[1]); weights=counts[counts>0]/counts.sum()
        endpoints=np.concatenate([source,goal]); std=endpoints.std(axis=0)
        result[name]=dict(pairs=len(rows),winning_dimension_counts=counts.tolist(),
            active_winning_dimensions=int((counts>0).sum()),
            effective_winning_dimensions=float(np.exp(-(weights*np.log(weights)).sum())),
            top_four_winner_fraction=float(np.sort(counts)[-4:].sum()/counts.sum()),
            coordinate_standard_deviation_quantiles={str(q):float(np.quantile(std,q)) for q in (0,.5,.9,1)},
            output_activation=model.output_activation,
            fraction_near_sigmoid_boundary=(float(((endpoints<model.cap*1e-4)|(endpoints>model.cap*(1-1e-4))).mean())
                                           if model.output_activation=='sigmoid' else None),
            interpretation='Winning-coordinate counts across these queries do not measure full network capacity or prove a dimension limit. They can reveal sparse use or saturation of the max-distance readout.')
    return result


def geometry(corpus,data,config,cohorts):
    """Compare exact shortest-path landmark coordinates to the model's metric.

    With every query goal included, d(s,landmark) coordinates give exact
    directed-max (or symmetric L-infinity) distances to that goal. This is an
    existence check for a scene's target domain, not an oracle-free encoder.
    A 128-landmark subset gives only a lower-bound probe when goals exceed it.
    """
    if config['model']['metric'] not in ('linf','directed_max'):
        raise ValueError('Landmark geometry probe is defined for Linf or directed max only')
    budget=config['evaluation'].get('landmarks',config['model']['dimension'])
    canonical=config.get('task')=='sokoban' and budget>=100
    rng=np.random.default_rng(config['seed']+211)
    landmarks={}
    if hasattr(corpus,'scenes'):
        for sid,scene in enumerate(corpus.scenes):
            goals=scene.goals+corpus.offsets[sid]
            landmarks[sid]=goals if len(goals)<=budget else rng.choice(goals,budget,replace=False)
    else:
        landmarks[0]=rng.choice(corpus.size,min(budget,corpus.size),replace=False)
    result={}
    for name in cohorts:
        rows=data['tasks_'+name]
        truth=data['distances_'+name]
        take=np.arange(len(rows))
        if len(take)>config['evaluation']['decision_count']:
            take=rng.choice(take,config['evaluation']['decision_count'],replace=False)
        rows=rows[take]; truth=truth[take]
        pred=np.empty(len(rows),dtype=np.float32)
        candidate_scores=np.full(data['children_'+name][take].shape,np.inf,dtype=np.float32)
        query_in_bank=np.zeros(len(rows),bool)
        scene_ids=corpus.locate(rows[:,0]) if hasattr(corpus,'scenes') else np.zeros(len(rows),int)
        for sid in np.unique(scene_ids):
            ids=np.flatnonzero(scene_ids==sid); bank=landmarks[int(sid)]
            children=data['children_'+name][take[ids]]; valid=children>=0
            states=np.concatenate([rows[ids].ravel(),children[valid]])
            unique,inverse=np.unique(states,return_inverse=True)
            width=100 if canonical else len(bank)
            profiles=np.zeros((len(unique),width),dtype=np.float32)
            for j,g in enumerate(bank):
                d=corpus.distance(unique,np.full(len(unique),g,dtype=np.int64))
                axis=int(corpus.scenes[int(sid)].states[g-corpus.offsets[int(sid)],0]) if canonical else j
                profiles[:,axis]=np.where(np.isfinite(d),d,config['training']['unreachable_target'])
            b=len(ids); parent=profiles[inverse[:2*b]].reshape(b,2,-1)
            difference=parent[:,0]-parent[:,1]
            pred[ids]=(np.maximum(difference.max(axis=1),0) if config['model']['metric']=='directed_max'
                       else np.abs(difference).max(axis=1))
            child=profiles[inverse[2*b:]]
            goals=np.broadcast_to(parent[:,1,None],(*children.shape,width))[valid]
            difference=child-goals
            scores=np.full(children.shape,np.inf,dtype=np.float32)
            scores[valid]=(np.maximum(difference.max(axis=1),0) if config['model']['metric']=='directed_max'
                           else np.abs(difference).max(axis=1))
            candidate_scores[ids]=scores
            query_in_bank[ids]=np.isin(rows[ids,1],bank)
        finite=np.isfinite(truth); decision=finite & (truth>0)
        chosen=candidate_scores.argmin(axis=1)
        optimal=data['optimal_'+name][take]
        result[name]=dict(count=len(rows),finite_count=int(finite.sum()),
            finite_mae=float(np.abs(pred[finite]-truth[finite]).mean()),
            finite_exact_fraction=float((np.abs(pred[finite]-truth[finite])<1e-5).mean()),
            query_goal_in_landmarks=int(query_in_bank.sum()),
            shortest_action_rate=float(optimal[np.flatnonzero(decision),chosen[decision]].mean()) if decision.any() else None,
            dimensions_per_scene={str(sid):100 if canonical else len(landmarks[int(sid)]) for sid in np.unique(scene_ids)},
            coordinate_assignment='Common100-cell player-position axes across every layout' if canonical else 'Per-scene landmark order',
            interpretation=('Oracle profiles only. In Sokoban a complete success goal is uniquely determined within a layout by its player cell, so100 globally shared axes suffice for the capped labels, without a query-goal-conditioned state vector. Padding to128 and a common coordinate shift fit the open Sigmoid output range. This does not establish that the fixed-width MLP or CNN can compute or learn those profiles, nor cover arbitrary intermediate goals or true infinity.' if canonical else
                'Oracle distance profiles only. Exact fit with all query goals in the bank establishes within-scene representability; it does not establish that the shared MLP learned or can generalize this map. Subset error is not an impossibility proof.'))
    return result


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--cohorts',nargs='+',default=['validation'])
    parser.add_argument('--checkpoint',type=Path)
    parser.add_argument('--stream-statistics',action='store_true')
    parser.add_argument('--gradient-batches',type=int,default=0)
    parser.add_argument('--linf-witness-count',type=int,default=0)
    parser.add_argument('--near-endpoint-count',type=int,default=0)
    parser.add_argument('--policy-boundary-count',type=int,default=0)
    parser.add_argument('--isometry-count',type=int,default=0)
    args=parser.parse_args(); config=json.loads(args.config.read_text())
    if args.out.exists(): raise FileExistsError('Keep previous diagnostics; choose a new output')
    corpus,data,_=load(config,args.run/'data')
    if args.isometry_count:
        if args.isometry_count<1 or not args.checkpoint: raise ValueError('Positive query count and checkpoint required')
        if any('validation' not in name for name in args.cohorts): raise ValueError('Isometry diagnosis uses validation cohorts only')
        import torch
        from grounded_llm.core.qmap.model import StateEncoder
        torch.set_num_threads(config['training']['cpu_threads'])
        saved=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
        if saved['config']!=config: raise ValueError('Isometry checkpoint configuration mismatch')
        model=StateEncoder(input_dim=corpus.input_dim,**config['model']); model.load_state_dict(saved['model']); model.requires_grad_(False); model.eval()
        result,archive=isometry_probe(model,corpus,data,config,args.cohorts,args.isometry_count)
        target=args.out.with_suffix('.queries.npz'); temporary=target.with_suffix('.tmp'); target.parent.mkdir(parents=True,exist_ok=True)
        with temporary.open('wb') as stream: np.savez_compressed(stream,**archive)
        temporary.replace(target)
        atomic_json(args.out,dict(checkpoint_step=saved['step'],training_source_commit=saved['source_commit'],queries_archive=str(target),**result)); return
    if args.policy_boundary_count:
        if args.policy_boundary_count<1 or not args.checkpoint: raise ValueError('Positive query count and checkpoint required')
        if any('validation' not in name for name in args.cohorts): raise ValueError('Policy diagnosis uses validation cohorts only')
        import torch
        from grounded_llm.core.qmap.model import StateEncoder
        from .relation_evaluate import QCache,planning_ids,rollout
        torch.set_num_threads(config['training']['cpu_threads'])
        saved=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
        if saved['config']!=config: raise ValueError('Policy diagnostic checkpoint configuration mismatch')
        model=StateEncoder(input_dim=corpus.input_dim,**config['model']); model.load_state_dict(saved['model']); model.requires_grad_(False); model.eval()
        cache=QCache(model,corpus,'cpu'); groups={}; archive={}
        for name in args.cohorts:
            rows=data['tasks_'+name]; truth=corpus.distance(rows[:,0],rows[:,1])
            rows=rows[planning_ids(corpus,rows,truth,args.policy_boundary_count,np.random.default_rng(config['seed']+len(name)))]
            records=[]; metrics=rollout(corpus,cache,rows,config['evaluation']['horizon'],'q',edge_records=records)
            summary,edges=policy_boundaries(records)
            groups[name]=dict(planning=metrics,boundaries=summary)
            archive['queries_'+name]=rows
            archive.update({key+'_'+name:value for key,value in edges.items()})
        target=args.out.with_suffix('.edges.npz'); temporary=target.with_suffix('.tmp'); target.parent.mkdir(parents=True,exist_ok=True)
        with temporary.open('wb') as stream: np.savez_compressed(stream,**archive)
        temporary.replace(target)
        atomic_json(args.out,dict(checkpoint_step=saved['step'],training_source_commit=saved['source_commit'],
            planning_protocol='unique_ordered_state_goal_pairs_v2',edges_archive=str(target),cohorts=groups)); return
    if args.near_endpoint_count:
        if config['task']!='token_swapping' or not args.checkpoint: raise ValueError('Token checkpoint required for near endpoint diagnosis')
        import torch
        from grounded_llm.core.qmap.model import StateEncoder
        from .relation_data import materialize
        from .relation_evaluate import evaluate
        torch.set_num_threads(config['training']['cpu_threads'])
        saved=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
        if saved['config']!=config: raise ValueError('Near endpoint checkpoint configuration mismatch')
        partition=np.load(args.run/'data'/'endpoint_partition.npy')
        groups=near_endpoint_cases(corpus,partition,config['seed'],args.near_endpoint_count)
        diagnostic={key+'_'+name:value for name,rows in groups.items() for key,value in materialize(corpus,rows).items()}
        queries=args.out.with_suffix('.queries.npz'); temporary=queries.with_suffix('.tmp')
        queries.parent.mkdir(parents=True,exist_ok=True)
        with temporary.open('wb') as stream:
            np.savez_compressed(stream,**diagnostic,**{'endpoint_folds_'+name:partition[rows] for name,rows in groups.items()})
        temporary.replace(queries)
        model=StateEncoder(input_dim=corpus.input_dim,**config['model']); model.load_state_dict(saved['model']); model.requires_grad_(False)
        metrics=evaluate(model,corpus,diagnostic,config,cohorts=list(groups),exposure=saved['exposure'])
        atomic_json(args.out,dict(checkpoint_step=saved['step'],training_source_commit=saved['source_commit'],seed=config['seed']+1009,
            queries_archive=str(queries),
            condition='Validation-only near endpoints: exact distances1-5 with equal quotas; same deterministic cases for every model. At least one endpoint is validation fold, none is test fold. Full legal successors remain available, as in the original endpoint cohorts. Not a replacement acceptance set or training supervision.',
            cases={name:dict(count=len(rows),unique_pairs=len(np.unique(rows,axis=0)),
                by_true_distance={str(d):int((diagnostic['distances_'+name]==d).sum()) for d in range(1,6)}) for name,rows in groups.items()},
            metrics=metrics)); return
    if args.linf_witness_count:
        atomic_json(args.out,linf_witness(corpus,config,args.linf_witness_count)); return
    if args.stream_statistics:
        atomic_json(args.out,stream_statistics(corpus,data,args.run,args.out)); return
    if args.gradient_batches:
        if not args.checkpoint or args.gradient_batches<1: raise ValueError('Positive batch count and checkpoint required')
        import torch
        from grounded_llm.core.qmap.model import StateEncoder
        torch.set_num_threads(config['training']['cpu_threads'])
        saved=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
        if saved['config']!=config: raise ValueError('Gradient probe checkpoint configuration mismatch')
        model=StateEncoder(input_dim=corpus.input_dim,**config['model']); model.load_state_dict(saved['model'])
        atomic_json(args.out,gradient_probe(model,corpus,args.run,config,saved['step'],args.gradient_batches)); return
    result=dict(teacher_only=True,used_by_encoder_or_planner=False,geometry=geometry(corpus,data,config,args.cohorts))
    if args.checkpoint:
        import torch
        from grounded_llm.core.qmap.model import StateEncoder
        saved=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
        if saved['config']!=config: raise ValueError('Diagnostic checkpoint configuration mismatch')
        torch.set_num_threads(config['training']['cpu_threads'])
        model=StateEncoder(input_dim=corpus.input_dim,**config['model']); model.load_state_dict(saved['model'])
        model.requires_grad_(False)
        result.update(checkpoint_step=saved['step'],encoder_axes=encoder_axes(model,corpus,data,config,args.cohorts))
    atomic_json(args.out,result)


if __name__=='__main__': main()
