"""Read-only convergence checks for an unchanged mixed Sokoban training run."""
import json
from pathlib import Path
import numpy as np
import torch
from grounded_llm.core.files import atomic_json
from grounded_llm.core.qmap.model import soft_rank_loss
from .relation_evaluate import QCache,rollout
from .solution_structure import exact_path,path_metrics
from .solution_global_order import global_order_metrics
from .solution_migration import decision_outcomes,paired_counts,chosen_branch_categories


def clean(value):
    if isinstance(value,float) and not np.isfinite(value):return None
    if isinstance(value,dict):return {str(k):clean(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [clean(v) for v in value]
    return value


def exposure_counts(draws):
    draws=np.asarray(draws)
    return dict(relations=len(draws),draws=int(draws.sum()),unique=int((draws>0).sum()),
        unique_fraction=float((draws>0).mean()),mean_draws=float(draws.mean()),minimum=int(draws.min()),
        quantiles={str(q):float(np.quantile(draws,q)) for q in [.1,.5,.9,.99]},
        interpretation='Mean draws is not a completed epoch; minimum zero means some relations remain unseen.')


class ConvergenceProbe:
    def __init__(self,corpus,data,config,out):
        self.corpus,self.data,self.config=corpus,data,config
        self.options=config['convergence'];self.root=Path(out)/'convergence';self.root.mkdir(exist_ok=True)
        self.np,self.nr=int(data['broad_point_count']),int(data['broad_rank_count'])
        self.graph={k:v[self.np if k in ('pairs_train','pair_distances_train') else self.nr:]
                    for k,v in data.items() if k.endswith('_train')}
        self.tasks=self.graph['tasks_train'];self.pairs=self.graph['pairs_train'];self.goals=np.unique(self.tasks[:,1])
        self.point_keys=self.pairs[:,0]*corpus.size+self.pairs[:,1];self.order=np.argsort(self.point_keys)
        self.sorted_keys=self.point_keys[self.order]
        self.parent_index=self.point_indices(self.tasks)
        children=self.graph['children_train'];valid=children>=0
        self.child_index=np.full(children.shape,-1,np.int64)
        self.child_index[valid]=self.point_indices(np.column_stack([children[valid],np.broadcast_to(self.tasks[:,1,None],children.shape)[valid]]))
        self.ids=np.unique(self.pairs)
        self.pair_index=np.searchsorted(self.ids,self.pairs)
        cache_root=Path(out)/'data'
        if (cache_root/'features_packed.npy').exists():
            cache_ids=np.load(cache_root/'feature_ids.npy');at=np.searchsorted(cache_ids,self.ids)
            if not np.array_equal(cache_ids[at],self.ids):raise ValueError('Probe feature identity changed')
            self.packed=np.load(cache_root/'features_packed.npy',mmap_mode='r')[at]
        else:
            self.packed=np.concatenate([np.packbits(corpus.features(self.ids[i:i+32768]).astype(np.uint8),axis=1,bitorder='little')
                                       for i in range(0,len(self.ids),32768)])
        self.dense=dict(np.load(cache_root/'dense_validation.npz')) if (cache_root/'dense_validation.npz').exists() else {}
        self.retention=dict(np.load(self.options['retention_bank'])) if self.options.get('retention_bank') else None
        bank=self.root/'queries.npz'
        if bank.exists():self.queries=dict(np.load(bank))
        else:
            self.queries=self.make_queries();np.savez_compressed(bank,**self.queries)
        self.best_file=self.root/'best_validation.json'
        self.best=json.loads(self.best_file.read_text()) if self.best_file.exists() else {}

    def point_indices(self,rows):
        keys=rows[:,0]*self.corpus.size+rows[:,1];at=np.searchsorted(self.sorted_keys,keys)
        if (at>=len(self.sorted_keys)).any() or not np.array_equal(self.sorted_keys[at],keys):
            raise ValueError('Training decision lacks complete candidate point supervision')
        return self.order[at]

    def make_queries(self):
        rng=np.random.default_rng(self.options['seed']);roots=[];alternative=[];recoverable=[];paths=[];order_ids=[];rejected=0
        rank_lookup={int(s)*self.corpus.size+int(g):i for i,(s,g) in enumerate(self.tasks)}
        distance=self.graph['distances_train'];children=self.graph['children_train'];truth=self.graph['child_distances_train']
        for goal in self.goals:
            ids=np.flatnonzero(self.tasks[:,1]==goal);labels=distance[ids]
            for lo,hi in zip([0]+self.options['path_distance_bins'][:-1],self.options['path_distance_bins']):
                pool=ids[(labels>lo)&(labels<=hi)]
                if not len(pool):continue
                accepted=0
                for index in rng.permutation(pool):
                    path=exact_path(self.corpus,self.tasks[index,0],goal,self.config['evaluation']['horizon'])
                    if not all(int(s)*self.corpus.size+int(goal) in rank_lookup for s in path[:-1]):
                        rejected+=1;continue
                    roots.append(self.tasks[index]);paths.append(path)
                    valid=children[index]>=0;optimal=valid&(truth[index]+1==distance[index])
                    def eligible(mask):
                        return [int(c) for c,d,take in zip(children[index],truth[index],mask)
                                if take and 0<d<=self.config['evaluation']['horizon'] and int(c)*self.corpus.size+int(goal) in rank_lookup]
                    choices=eligible(optimal)
                    # Any other optimal branch than the fixed exact teacher's
                    # first move becomes an ordinary greedy starting state.
                    choices=[c for c in choices if c!=path[1]]
                    if choices:alternative.append([rng.choice(choices),goal])
                    choices=eligible(valid&np.isfinite(truth[index])&~optimal)
                    if choices:recoverable.append([rng.choice(choices),goal])
                    accepted+=1
                    if accepted==self.options['queries_per_goal_layer']:break
            bins=self.options['order_distance_bins']
            for lo,hi in zip([0]+bins[:-1],bins):
                pool=ids[(labels>lo)&(labels<=hi)]
                order_ids.extend(rng.choice(pool,min(self.options['order_per_layer'],len(pool)),replace=False))
        padded=np.full((len(paths),self.config['evaluation']['horizon']+1),-1,np.int64)
        for i,path in enumerate(paths):padded[i,:len(path)]=path
        result=dict(teacher_paths=padded,order_ids=np.asarray(order_ids,np.int64))
        for name,rows in [('root',roots),('alternative_optimal',alternative),('recoverable_suboptimal',recoverable)]:
            result['tasks_'+name]=np.unique(np.asarray(rows,np.int64).reshape(-1,2),axis=0)
        atomic_json(self.root/'query_scope.json',dict(targets=len(self.goals),
            counts={k:len(v) for k,v in result.items()},teacher_paths_rejected_at_protected_boundaries=rejected,
            scope='Only accepted training graph targets and starts. Teacher paths wholly inside accepted training contexts; branch entries are normal starts for unchanged greedy32, with no forced inference action. No validation feedback.'))
        return result

    def keep_best(self,name,value,step,checkpoint):
        if value is None or (name in self.best and value<=self.best[name]['value']):return
        from .relations import retain_checkpoint
        target=self.root/f'best_{name}.pt';retain_checkpoint(checkpoint,target)
        self.best[name]=dict(value=float(value),step=int(step),checkpoint=str(target))
        atomic_json(self.best_file,self.best)

    def retain_policy_bests(self,checkpoint,step,validation):
        for name in ('broad_goal_validation','broad_layout_validation'):
            row=validation['cohorts'].get(name,{})
            self.keep_best('greedy_'+name,row.get('planning',{}).get('q',{}).get('success_rate'),step,checkpoint)

    @torch.no_grad()
    def inspect(self,model,step,pair_draws,rank_draws,validation,checkpoint=None):
        model.eval();device=next(model.parameters()).device
        vectors=[]
        for start in range(0,len(self.ids),32768):
            values=np.unpackbits(self.packed[start:start+32768],axis=1,count=self.corpus.input_dim,bitorder='little')
            vectors.append(model.encode(torch.as_tensor(values,dtype=torch.float32,device=device)))
        vectors=torch.cat(vectors);prediction=np.empty(len(self.pairs),np.float32)
        for start in range(0,len(self.pairs),16384):
            at=self.pair_index[start:start+16384]
            prediction[start:start+len(at)]=model.distance(vectors[at[:,0]],vectors[at[:,1]]).cpu().numpy()
        cache=QCache(model,self.corpus,device);cache.cache.update(zip(map(int,self.ids),vectors))
        valid=self.child_index>=0;scores=np.full(valid.shape,np.inf,np.float32);scores[valid]=prediction[self.child_index[valid]]
        outcomes=decision_outcomes(self.graph,scores)
        exact=self.graph['pair_distances_train'];finite=np.isfinite(exact)
        target=np.where(finite,exact,self.config['training']['unreachable_target']);error=np.abs(prediction-target)
        truth=self.graph['child_distances_train'];rank_loss=0
        for start in range(0,len(scores),16384):
            end=min(start+16384,len(scores));legal=valid[start:end]
            pred=np.where(legal,1+scores[start:end],0)
            labels=1+np.where(np.isfinite(truth[start:end]),truth[start:end],self.config['training']['unreachable_target'])
            rank_loss+=float(soft_rank_loss(torch.as_tensor(pred,device=device),torch.as_tensor(labels,device=device),
                torch.as_tensor(legal,device=device),self.config['training']['rank_temperature']))*(end-start)
        point_draws=pair_draws[self.np:];decision_draws=rank_draws[self.nr:]
        def fit(point_mask,rank_mask):
            eligible=point_mask&finite
            return dict(points=int(point_mask.sum()),decisions=int(rank_mask.sum()),
                finite_mae=float(np.abs(prediction[eligible]-exact[eligible]).mean()) if eligible.any() else None,
                shortest_action_rate=float(outcomes['shortest_action'][rank_mask].mean()) if rank_mask.any() else None,
                complete_candidate_order_rate=float(outcomes['complete_candidate_order'][rank_mask].mean()) if rank_mask.any() else None,
                point_exposure=exposure_counts(point_draws[point_mask]) if point_mask.any() else None,
                decision_exposure=exposure_counts(decision_draws[rank_mask]) if rank_mask.any() else None)
        training=fit(np.ones(len(exact),bool),np.ones(len(self.tasks),bool))
        training.update(distance_huber=float(np.where(error<1,.5*error**2,error-.5).mean()),
            complete_candidate_soft_rank_loss=rank_loss/len(scores),
            by_target={str(int(g)):fit(self.pairs[:,1]==g,self.tasks[:,1]==g) for g in self.goals},
            by_distance={str(int(d)):fit(exact==d,self.graph['distances_train']==d)
                         for d in np.unique(np.r_[exact[finite],self.graph['distances_train']])})
        before=self.root/'baseline_outcomes.npz'
        if not before.exists():np.savez_compressed(before,**outcomes)
        old=dict(np.load(before));training['error_migration_from_baseline']={k:paired_counts(old[k],v) for k,v in outcomes.items()}
        _,_,training['chosen_action_types']=chosen_branch_categories(self.corpus,self.graph,scores)
        np.savez_compressed(self.root/f'predictions-{step}.npz',point_predictions=prediction,**outcomes)
        ordered=self.queries['order_ids'];train_order=global_order_metrics(self.corpus,self.tasks[ordered],
            self.graph['distances_train'][ordered],prediction[self.parent_index[ordered]])
        train_paths=path_metrics(cache,self.corpus,self.queries['teacher_paths'])
        train_rollouts={name[6:]:rollout(self.corpus,cache,rows,self.config['evaluation']['horizon'],'q')
                        for name,rows in self.queries.items() if name.startswith('tasks_') and len(rows)}
        dense={}
        for name,rows in self.dense.items():
            if not name.startswith('tasks_'):continue
            cohort=name[6:];dense[cohort]=global_order_metrics(self.corpus,rows,self.dense['distances_'+cohort],cache.scores(rows[:,0],rows[:,1]))
        retention=None
        if self.retention is not None:
            rows=self.retention['pairs'];pred=cache.scores(rows[:,0],rows[:,1]);tasks=self.retention['tasks'];children=self.retention['children'];legal=children>=0
            cs=np.full(children.shape,np.inf,np.float32);cs[legal]=cache.scores(children[legal],np.broadcast_to(tasks[:,1,None],children.shape)[legal])
            correct=self.retention['child_distances'][np.arange(len(tasks)),cs.argmin(axis=1)]+1==self.retention['distances']
            retention=dict(finite_mae=float(np.abs(pred-self.retention['truth']).mean()),shortest_action_rate=float(correct.mean()),
                scope='Same fixed allowed original-training relationships, excluding the 16 small-study layouts; no checkpoint selection here.')
        witnesses={}
        witness_queries=[('training',self.queries['tasks_root'],None)]
        if 'tasks_broad_goal_validation' in self.data:
            witness_queries.append(('new_layout_new_goal',self.data['tasks_broad_goal_validation'],self.data['distances_broad_goal_validation']))
        for name,rows,labels in witness_queries:
            if not len(rows):continue
            if labels is None:labels=self.corpus.distance(rows[:,0],rows[:,1])
            candidates=np.flatnonzero((labels>=21)&(labels<=32));index=int(candidates[0]) if len(candidates) else 0
            edges=[];rollout(self.corpus,cache,rows[[index]],self.config['evaluation']['horizon'],'q',edges)
            witnesses[name]=dict(query=rows[index].tolist(),true_distance=float(labels[index]),
                steps=[{k:v.tolist() for k,v in edge.items()} for edge in edges])
        report=dict(step=step,training=training,training_global_order=train_order,training_teacher_paths=train_paths,
            training_greedy=train_rollouts,validation=validation,validation_global_order=dense,retention=retention,
            witnesses=witnesses,total_point_draws=int(pair_draws.sum()),total_decision_draws=int(rank_draws.sum()),
            scope='Read-only diagnostics; original exact loss, complete candidates, and greedy32 unchanged. No metrics, queries or labels change training sampling or optimizer state. Null witness costs mean proven unreachable.')
        atomic_json(self.root/f'{step}.json',clean(report))
        if checkpoint is not None and Path(checkpoint).exists() and 'broad_goal_validation' in dense:
            metrics=dense['broad_goal_validation']['aggregate']
            self.keep_best('global_order',metrics['all']['accuracy'],step,checkpoint)
            self.keep_best('adjacent_order',metrics['gap_1']['accuracy'],step,checkpoint)
        print(json.dumps(dict(convergence_step=step,training_mae=training['finite_mae'],
            training_action_rate=training['shortest_action_rate'],graph_unique_fraction=training['decision_exposure']['unique_fraction'],
            training_solved={k:v['success_rate'] for k,v in train_rollouts.items()},
            heldout_goal_success=validation['cohorts'].get('broad_goal_validation',{}).get('planning',{}).get('q',{}).get('success_rate'))),flush=True)
