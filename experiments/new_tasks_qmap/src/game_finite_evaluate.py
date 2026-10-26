"""Exhaustive numerical-state fitting checks for the finite distinct-source DAGs."""
import argparse
from copy import deepcopy
import json
from itertools import combinations
from pathlib import Path
import numpy as np
import torch
from grounded_llm.core.files import atomic_json
from grounded_llm.core.qmap.model import StateEncoder
from grounded_llm.tasks.game24.relation_data import load
from .relation_data import materialize
from .relation_evaluate import evaluate,QCache,safe_scores,rollout


def numerical_queries(corpus,targets,exposure=None):
    """Deduplicate by exact reduced rationals, retaining one closed physical DAG.

    Future arithmetic depends on the remaining multiset, not its root question.
    Repeated occurrences must agree on every target's exact reachability/cost.
    """
    representatives={};roots=[];all_values=set();seen_values=set()
    for sid,scene in enumerate(corpus.scenes):
        offset=int(corpus.offsets[sid]);columns=[];goals=[]
        for target in targets:
            columns.append(next(j for j,g in enumerate(scene.goals) if scene.values[g]==(target,)))
            goals.append(int(scene.goals[columns[-1]])+offset)
        root=int(scene.metadata['original_state']);roots.append((offset+root,goals))
        for local,value in enumerate(scene.values):
            all_values.add(value)
            if exposure is not None and exposure[offset+local]>0:seen_values.add(value)
            if len(value) not in (2,3):continue
            truth=tuple(float(scene.distances[j,local]) for j in columns)
            previous=representatives.get(value)
            if previous is None:representatives[value]=(offset+local,goals,truth)
            elif previous[2]!=truth:raise ValueError('Identical numerical states have different exact target costs')
    groups={}
    for column,target in enumerate(targets):
        groups[f'root4_to_{target}']=np.asarray([[s,g[column]] for s,g in roots],dtype=np.int64)
        for arity in (2,3):
            groups[f'suffix{arity}_to_{target}']=np.asarray([[s,g[column]] for v,(s,g,_) in representatives.items() if len(v)==arity],dtype=np.int64)
    return groups,dict(physical_states=corpus.size,exact_numerical_states=len(all_values),
        numerical_states_seen_in_loss=len(seen_values) if exposure is not None else None,
        numerical_input_coverage=len(seen_values)/len(all_values) if exposure is not None else None,
        duplicate_cost_consistency_verified=True,
        deduplication='Exact sorted reduced Fraction tuples; one representative per2/3-number multiset, separately for each singleton target.')


def target_switch_metrics(model,corpus,groups,horizon):
    """Same physical roots, two goals whose optimal action sets are disjoint."""
    first,second=groups['root4_to_21'],groups['root4_to_24']
    if not np.array_equal(first[:,0],second[:,0]):raise ValueError('Goal-switch roots differ')
    children=corpus.successors(first[:,0]);valid=children>=0;optimal=[]
    for rows in [first,second]:
        truth=corpus.distance(*rows.T);child=np.full(children.shape,np.inf)
        child[valid]=corpus.distance(children[valid],np.broadcast_to(rows[:,1,None],children.shape)[valid])
        optimal.append(valid & np.isfinite(truth[:,None]) & (child+1==truth[:,None]))
    selected=optimal[0].any(1)&optimal[1].any(1)&~(optimal[0]&optimal[1]).any(1)
    n=int(selected.sum());cache=QCache(model,corpus,next(model.parameters()).device)
    if not n:return dict(count=0,different_first_actions=0,both_first_actions_optimal=0,both_complete=0,missing=True)
    choices=[];completed=[]
    for rows in [first[selected],second[selected]]:
        scores=safe_scores(cache,corpus.successors(rows[:,0]),rows[:,1]);choices.append(scores.argmin(1))
        records=[];plan=rollout(corpus,cache,rows,horizon,'q',records);done=np.zeros(n,dtype=bool)
        for edge in records:done[edge['query_index'][edge['next_is_goal']]]=True
        if done.sum()!=plan['solved']:raise ValueError('Goal-switch trajectory outcomes disagree')
        completed.append(done)
    index=np.arange(n);both=optimal[0][selected][index,choices[0]]&optimal[1][selected][index,choices[1]]
    return dict(count=n,different_first_actions=int((choices[0]!=choices[1]).sum()),
        both_first_actions_optimal=int(both.sum()),both_first_actions_optimal_rate=float(both.mean()),
        both_complete=int((completed[0]&completed[1]).sum()),both_complete_rate=float((completed[0]&completed[1]).mean()),
        scope='Same root, both singleton goals solvable and exact optimal FIRST physical action sets disjoint. All candidates retained; teacher checks follow model action selection.')


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--protocol',type=Path,required=True)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--device',default='cpu')
    args=parser.parse_args()
    if args.out.exists():raise FileExistsError('Preserve previous frozen evaluations')
    protocol=json.loads(args.protocol.read_text(encoding='utf-8'))
    summary=json.loads((args.run/'summary.json').read_text())
    saved=torch.load(args.run/'best.pt',map_location='cpu',weights_only=False)
    config=saved['config'];spec=config['data']
    if config['task']!='game24' or spec.get('initial_sampling')!='exhaustive_distinct':
        raise ValueError('This protocol requires the complete finite distinct-source training domain')
    if spec['initial_numbers']!=4 or spec['source_values']!=protocol['source_values']:
        raise ValueError('Finite source domain differs from the declared protocol')
    if saved['step']!=summary['evaluated_step']:raise ValueError('Final report and frozen selected checkpoint differ')
    torch.set_num_threads(config['training']['cpu_threads'])
    corpus=load(args.run/'data/graphs',feature_encoding=config.get('input_encoding','identity'))
    try:
        roots=[tuple(s.metadata['source']['numbers']) for s in corpus.scenes]
        if len(set(roots))!=len(roots) or set(roots)!=set(combinations(protocol['source_values'],4)):
            raise ValueError('Physical corpus does not contain exactly all distinct four-number roots')
        groups,coverage=numerical_queries(corpus,protocol['targets'],saved['exposure'])
        for name,count in protocol['expected_queries'].items():
            if len(groups[name])!=count:raise ValueError(('Finite query domain incomplete',name,len(groups[name]),count))
        # These existing query holdouts are local to their question. A numerical
        # goal may have been trained in another question, so no global unseen
        # target or held input claim is made.
        missing=[]
        with np.load(args.run/'data/relations.npz') as archive:
            for split in ('validation','test'):
                key='tasks_unseen_goal_'+split
                if key not in archive:
                    if split=='test':missing.extend([f'held_intermediate{arity}_query_test' for arity in (2,3)])
                    continue
                rows=archive[key];goal_arity=np.empty(len(rows),dtype=np.int8)
                for sid,mask in corpus._scene_groups(rows[:,1]):
                    scene=corpus.scenes[sid];offset=corpus.offsets[sid]
                    goal_arity[mask]=[len(scene.values[g-offset]) for g in rows[mask,1]]
                for arity in (2,3):
                    selected=rows[goal_arity==arity];name=f'held_intermediate{arity}_query_{split}'
                    if len(selected):groups[name]=selected
                    elif split=='test':missing.append(name)
        materialized={}
        for name,rows in groups.items():
            materialized.update({k+'_'+name:v for k,v in materialize(corpus,rows).items()})
        evaluation=deepcopy(config)
        evaluation['evaluation'].update(decision_count=max(map(len,groups.values())),
            rollout_count=max(map(len,groups.values())),horizon=4)
        model=StateEncoder(input_dim=saved['input_dim'],**config['model']).to(args.device)
        model.load_state_dict(saved['model']);model.requires_grad_(False)
        metrics=evaluate(model,corpus,materialized,evaluation,exposure=saved['exposure'],
            threshold=summary['metrics']['reachability_threshold'])
        switches=target_switch_metrics(model,corpus,groups,4) if protocol.get('target_switch_required') else None
        if switches is not None and switches['count']!=protocol.get('expected_target_switch_count',switches['count']):
            raise ValueError('Target-switch query domain differs from the frozen count')
        plans={name:value['planning']['q'] for name,value in metrics['cohorts'].items()}
        oracle_ok=all(value['planning']['oracle']['optimal_rate']==1 for value in metrics['cohorts'].values())
        passed=not missing and oracle_ok and coverage['numerical_input_coverage']==1
        passed=passed and all(p['success_rate'] is not None and p['success_rate']>=protocol['minimum_completion_rate'] for p in plans.values())
        if switches is not None:
            passed=passed and switches['count']>0 and switches['both_first_actions_optimal_rate']>=protocol['minimum_completion_rate'] and switches['both_complete_rate']>=protocol['minimum_completion_rate']
        reachability_passed=all(
            value['unreachable']['recall'] is not None and value['unreachable']['recall']>=protocol['minimum_unreachable_recall']
            and value['unreachable']['finite_false_positive'] is not None
            and value['unreachable']['finite_false_positive']<=protocol['maximum_finite_false_positive']
            for name,value in metrics['cohorts'].items() if name in protocol['expected_queries'])
        result=dict(protocol=protocol,checkpoint=str(args.run/'best.pt'),step=saved['step'],source_commit=saved['source_commit'],
            checkpoint_selection='Existing validation objective only; this suite never selects a checkpoint.',
            scope='Exhaustive finite-domain fitting, not unseen initial-question generalization.2/3-number suffixes are deduplicated across roots. Unsolvable queries remain in distance/reachability evaluation; completion uses every positive-distance solvable query. Local held query goals may occur as training goals or state inputs in other questions.',
            coverage=coverage,missing_required_cohorts=missing,target_switch=switches,oracle_ok=oracle_ok,completion_passed=passed,reachability_passed=reachability_passed,
            passed=passed and reachability_passed,
            target_reached=passed and reachability_passed and all(p['success_rate']>=protocol['target_completion_rate'] for p in plans.values())
                and (switches is None or (switches['both_first_actions_optimal_rate']>=protocol['target_completion_rate']
                    and switches['both_complete_rate']>=protocol['target_completion_rate'])),metrics=metrics)
        atomic_json(args.out,result)
        print(json.dumps(dict(output=str(args.out),passed=result['passed'],coverage=coverage,
            cohorts={name:{k:p[k] for k in ['count','solved','optimal','cycles','entered_unreachable']} for name,p in plans.items()})),flush=True)
    finally:
        for scene in corpus.scenes:
            for name in ('states','children','distances','reverse_distances'):
                getattr(scene,name)._mmap.close()


if __name__=='__main__':main()
