"""Frozen fixed-tree candidates: protected regression and fresh held inputs."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import numpy as np
import torch
from grounded_llm.core.files import atomic_json
from grounded_llm.core.qmap.model import StateEncoder
from .relation_data import closure,keys,load,materialize
from .relation_evaluate import evaluate


def fresh_queries(corpus,partition,goal_partition,count,bounds,rng,excluded):
    """Teacher-only distance layers; source inputs are always held test states."""
    source_pool=np.flatnonzero(partition==2);selected=[];used=set(map(int,excluded))
    for layer in range(len(bounds)):
        quota=count//len(bounds)+(layer<count%len(bounds));rows=[]
        for _ in range(100):
            if len(rows)==quota:break
            size=max(256,(quota-len(rows))*32)
            source=rng.choice(source_pool,size)
            goal=corpus.sample_layer_goals(source,np.asarray(bounds),np.full(size,layer),rng)
            candidates=np.column_stack([source,goal])
            candidates=candidates[(partition[goal]==goal_partition)&(source!=goal)]
            for row in candidates:
                key=int(keys(*row,corpus.size))
                if key in used:continue
                used.add(key);rows.append(row)
                if len(rows)==quota:break
        if len(rows)!=quota:raise ValueError('Insufficient fresh held-input queries in a distance layer')
        selected.extend(rows)
    result=np.asarray(selected,dtype=np.int64);rng.shuffle(result)
    if not ((partition[result[:,0]]==2).all() and (partition[result[:,1]]==goal_partition).all()):
        raise ValueError('Fresh suite violates endpoint holdout')
    return result


def validation_efficiency_candidates(config,runs,device):
    """Choose among terminal snapshots using validation trajectories only."""
    candidates=[];expected_scope=None
    for name in config['candidates']:
        ref=runs/name
        if not (ref/'summary.json').exists():raise ValueError('Candidate training is not terminal')
        seen_steps=set()
        for filename in config.get('candidate_checkpoints',['best.pt']):
            if filename not in ('best.pt','last.pt'):raise ValueError('Expected a complete saved candidate checkpoint')
            saved=torch.load(ref/filename,map_location='cpu',weights_only=False)
            if saved['step'] in seen_steps:continue
            seen_steps.add(saved['step']);original=saved['config']
            torch.set_num_threads(original['training']['cpu_threads'])
            corpus,data,_=load(original,ref/'data')
            try:
                if expected_scope is None:expected_scope=deepcopy(corpus.scope)
                elif corpus.scope!=expected_scope:raise ValueError('Efficiency candidates use different underlying trees')
                partition=np.load(ref/'data/endpoint_partition.npy');exposure=saved['exposure']
                if not (exposure[partition!=0]==0).all():raise ValueError('Held endpoint entered candidate loss inputs')
                names=[key[6:] for key in data if key.startswith('tasks_') and 'validation' in key]
                if not names:raise ValueError('No validation trajectories for efficiency selection')
                evaluation=deepcopy(original);evaluation['evaluation'].update(decision_count=config['selection_queries'],
                    rollout_count=config['selection_queries'],horizon=config['horizon'])
                model=StateEncoder(input_dim=saved['input_dim'],**original['model']).to(device)
                model.load_state_dict(saved['model']);model.requires_grad_(False)
                metrics=evaluate(model,corpus,data,evaluation,cohorts=names,exposure=exposure)
                plans=[v['planning']['q'] for v in metrics['cohorts'].values()]
                long=[]
                for p in plans:
                    rows=[v for d,v in p['by_distance'].items() if float(d)>config['long_distance_above']]
                    n=sum(v['count'] for v in rows)
                    if n:long.append(sum(v['optimal'] for v in rows)/n)
                optimal=[p['optimal_rate'] for p in plans]+long
                ratios=[p['mean_cost_ratio_on_success'] for p in plans]
                candidates.append(dict(run=name,checkpoint=filename,step=saved['step'],
                    score=float(np.mean([v['finite_huber']+v['candidate_soft_rank_loss'] for v in metrics['cohorts'].values()])),
                    full_point_source_coverage=bool((saved['pair_draws']>0).all()),
                    minimum_validation_completion=min(p['success_rate'] for p in plans),
                    minimum_validation_optimal_including_long=min(optimal),mean_validation_optimal=float(np.mean(optimal)),
                    mean_validation_success_cost_ratio=float(np.mean(ratios)) if all(x is not None for x in ratios) else None,
                    selection_cohorts=names,validation_query_count=config['selection_queries']))
                del model,metrics
            finally:corpus.close()
    eligible=[v for v in candidates if v['full_point_source_coverage']
        and v['minimum_validation_completion']>=config['minimum_completion_rate']]
    if not eligible:raise ValueError('No efficiency candidate preserves the validation completion guard')
    chosen=min(eligible,key=lambda v:(-v['minimum_validation_optimal_including_long'],-v['mean_validation_optimal'],
        v['mean_validation_success_cost_ratio'],v['score'],v['run'],v['checkpoint']))
    return candidates,chosen


def run(config,runs,device,source_commit):
    out=runs/config['output']
    if out.exists():raise FileExistsError('Do not overwrite a frozen acceptance suite')
    out.mkdir();candidates=[]
    if config.get('validation_efficiency_selection'):
        candidates,chosen=validation_efficiency_candidates(config,runs,device)
    else:
        for name in config['candidates']:
            ref=runs/name
            if not (ref/'summary.json').exists():raise ValueError('Candidate training is not terminal')
            saved=torch.load(ref/'best.pt',map_location='cpu',weights_only=False)
            candidates.append(dict(run=name,score=float(saved['best_score']),step=saved['step'],
                full_point_source_coverage=bool((saved['pair_draws']>0).all())))
        eligible=[r for r in candidates if r['full_point_source_coverage']]
        if not eligible:raise ValueError('No candidate has full prepared point-source coverage')
        chosen=min(eligible,key=lambda r:(r['score'],r['run']))
    atomic_json(out/'freeze_choice.json',dict(candidates=candidates,chosen=chosen,rule=config['selection'],
        independent_suite_used_for_choice=False))
    ref=runs/chosen['run'];saved=torch.load(ref/chosen.get('checkpoint','best.pt'),map_location=device,weights_only=False)
    original=saved['config'];torch.set_num_threads(original['training']['cpu_threads'])
    corpus,data,_=load(original,ref/'data')
    try:
        partition=np.load(ref/'data/endpoint_partition.npy');exposure=saved['exposure']
        held_clean=bool((exposure[partition!=0]==0).all())
        if not held_clean:raise ValueError('Held input states participated in a training loss')
        rng=np.random.default_rng(config['seed']);groups={}
        rows=np.unique(data['tasks_test'],axis=0)
        rows=rows[corpus.distance(*rows.T)>0]
        if len(rows)<config['queries_per_cohort']:raise ValueError('Too few protected regression queries')
        groups['protected_relation_test']=rows[rng.choice(len(rows),config['queries_per_cohort'],replace=False)]
        protected=groups['protected_relation_test']
        if not (partition[protected]==0).all() or not np.isin(closure(corpus,protected),data['reserved_relations']).all():
            raise ValueError('Regression queries are not protected known-input relations')
        prior=np.concatenate([v for k,v in data.items() if k.startswith('tasks_') and k!='tasks_train'])
        excluded=np.unique(keys(prior[:,0],prior[:,1],corpus.size))
        for filename in config.get('exclude_suites',[]):
            previous=json.loads((runs/filename).read_text())
            previous_rows=np.vstack([np.asarray(v,dtype=np.int64) for v in previous['groups'].values()])
            if ((previous_rows<0)|(previous_rows>=corpus.size)).any():raise ValueError('Previous suite exceeds this tree')
            excluded=np.unique(np.r_[excluded,keys(previous_rows[:,0],previous_rows[:,1],corpus.size)])
        for name,goal_partition in [('fresh_unseen_endpoint_test',0),('fresh_unseen_both_test',2)]:
            groups[name]=fresh_queries(corpus,partition,goal_partition,config['queries_per_cohort'],
                config['distance_bin_upper_bounds'],rng,excluded)
        if 'diagnostic_reference' in config:
            with np.load(runs/config['diagnostic_reference']) as archive:
                groups['diagnostic']=archive['tasks_diagnostic'].copy()
            if ((groups['diagnostic']<0)|(groups['diagnostic']>=corpus.size)).any():
                raise ValueError('Transported diagnostic permutations exceed this tree size')
        else:groups['diagnostic']=data['tasks_diagnostic']
        atomic_json(out/'suite.json',dict(config=config,groups={k:v.tolist() for k,v in groups.items()},
            source_scope=corpus.scope,held_input_training_exposure_zero=held_clean,
            protected_relation_scope='Original frozen regression pool; not a new independent relation split.',
            diagnostic_scope=('Transported known permutation pairs, relabelled costs/actions use this tree; not an independent input-holdout claim.'
                if 'diagnostic_reference' in config else 'Original known difficult permutation pairs.'),
            fresh_scope='New pairs with held test source inputs; balanced exact distance layers; never added to training.'))
        materialized={}
        for name,rows in groups.items():materialized.update({k+'_'+name:v for k,v in materialize(corpus,rows).items()})
        evaluation=deepcopy(original);evaluation['evaluation'].update(decision_count=config['queries_per_cohort'],
            rollout_count=config['queries_per_cohort'],horizon=config['horizon'])
        model=StateEncoder(input_dim=saved['input_dim'],**original['model']).to(device)
        model.load_state_dict(saved['model']);model.requires_grad_(False)
        metrics=evaluate(model,corpus,materialized,evaluation,exposure=exposure)
        plans={k:r['planning']['q'] for k,r in metrics['cohorts'].items()};long={}
        for name in ['fresh_unseen_endpoint_test','fresh_unseen_both_test']:
            rows=[v for d,v in plans[name]['by_distance'].items() if float(d)>config['long_distance_above']]
            count=sum(v['count'] for v in rows);solved=sum(v['solved'] for v in rows)
            optimal=sum(v['optimal'] for v in rows)
            long[name]=dict(count=count,solved=solved,success_rate=solved/count if count else None,
                optimal=optimal,optimal_rate=optimal/count if count else None)
        oracle_ok=all(r['planning']['oracle']['optimal_rate']==1 for r in metrics['cohorts'].values())
        coverage=bool((exposure[partition==0]>0).all())
        passed=oracle_ok and coverage and all(p['success_rate']>=config['minimum_completion_rate'] for p in plans.values())
        passed=passed and all(p['count']>=config['minimum_long_queries'] and p['success_rate']>=config['minimum_completion_rate'] for p in long.values())
        efficiency=None
        if 'minimum_optimal_rate' in config:
            primary={name:p for name,p in plans.items() if name!='diagnostic'}
            efficiency=dict(minimum_optimal_rate=config['minimum_optimal_rate'],
                maximum_mean_cost_ratio_on_success=config['maximum_mean_cost_ratio_on_success'],
                passed=all(p['optimal_rate']>=config['minimum_optimal_rate']
                    and p['mean_cost_ratio_on_success'] is not None
                    and p['mean_cost_ratio_on_success']<=config['maximum_mean_cost_ratio_on_success'] for p in primary.values())
                    and all(p['optimal_rate'] is not None and p['optimal_rate']>=config['minimum_optimal_rate'] for p in long.values()),
                target_reached=all(p['optimal_rate']>=config['target_optimal_rate'] for p in primary.values())
                    and all(p['optimal_rate'] is not None and p['optimal_rate']>=config['target_optimal_rate'] for p in long.values()))
            passed=passed and efficiency['passed']
        result=dict(source_commit=source_commit,checkpoint_source_commit=saved['source_commit'],step=saved['step'],
            chosen=chosen,oracle_ok=oracle_ok,full_training_input_coverage=coverage,held_input_training_exposure_zero=held_clean,
            long_distance=long,efficiency=efficiency,passed=passed,
            target_reached=passed and all(p['success_rate']>=config['target_completion_rate'] for p in plans.values())
                and (efficiency is None or efficiency['target_reached']),metrics=metrics)
        atomic_json(out/'result.json',result)
        print(json.dumps(dict(passed=passed,chosen=chosen,long_distance=long,
            cohorts={n:{k:p[k] for k in ['count','solved','optimal','cycles']} for n,p in plans.items()})),flush=True)
    finally:corpus.close()


def main():
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('--config',required=True,type=Path)
    parser.add_argument('--runs',required=True,type=Path);parser.add_argument('--device',default='cpu')
    parser.add_argument('--source-commit',required=True);args=parser.parse_args()
    run(json.loads(args.config.read_text()),args.runs,args.device,args.source_commit)


if __name__=='__main__':main()
