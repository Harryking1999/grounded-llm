"""Paired current-checkpoint content probe on fixed held-out reference states."""

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import random

import numpy as np
import torch

from .counterfactual import reverse_candidate_q
from .diagnostic_eval import append_history, load_evaluation, scored_answer
from .trajectory_dataset import load_record
from .trajectory_eval import GenerationSession
from .trajectory_metrics import score_turn


def select_records(records, config):
    selected = []
    rng = random.Random(config['seed'])
    for origin in sorted({r['group'] for r in records}):
        for length, quota in config['tasks_per_origin_by_shortest_length'].items():
            pool = sorted((r for r in records if r['group']==origin and r['shortest_moves']==int(length)),
                          key=lambda r:r['trajectory_id'])
            if len(pool)<quota:
                raise ValueError('Insufficient tasks in prescribed stratum')
            selected.extend(rng.sample(pool,quota))
    if len(selected)!=config['sample_tasks']:
        raise ValueError('Prescribed sample size differs')
    return selected


def probe_turns(demo, environment, stages=None):
    stages = stages or ('first', 'later_solvable_nearest_decision_4')
    if not set(stages) <= {'first', 'later_solvable_nearest_decision_4'}:
        raise ValueError('Unknown probe stage')
    eligible = [i for i,t in enumerate(demo.turns) if t.supervise and not t.step.done and
                environment.remaining(t.step.current,t.step.goal,t.executed_path)>0]
    if not eligible or eligible[0]!=0:
        raise ValueError('Ordinary task lacks a live first decision')
    selected = {0:'first'} if 'first' in stages else {}
    later = [i for i in eligible if i>0]
    if later and 'later_solvable_nearest_decision_4' in stages:
        selected[min(later,key=lambda i:(abs(i-3),i))]='later_solvable_nearest_decision_4'
    return selected


def aggregate_pairs(cases):
    groups = defaultdict(list)
    for case in cases:
        for row in case['turns']:
            for group in ('all',row['stage'],'origin/'+case['group'],'length/'+str(case['shortest_moves'])):
                groups[group].append(row)
    result = {}
    for group,rows in groups.items():
        normal_safe = sum(r['normal']['action_keeps_goal_reachable'] for r in rows)
        reverse_safe = sum(r['reverse']['action_keeps_goal_reachable'] for r in rows)
        paired = Counter(('safe' if r['normal']['action_keeps_goal_reachable'] else 'unsafe')+'/'+
                         ('safe' if r['reverse']['action_keeps_goal_reachable'] else 'unsafe') for r in rows)
        result[group] = dict(pairs=len(rows),normal_safe=normal_safe,reverse_safe=reverse_safe,
            normal_legal=sum(r['normal'].get('legal_action',False) for r in rows),
            reverse_legal=sum(r['reverse'].get('legal_action',False) for r in rows),
            normal_shortest=sum(r['normal']['action_environment_shortest'] for r in rows),
            reverse_shortest=sum(r['reverse']['action_environment_shortest'] for r in rows),
            changed_physical_action=sum(r['changed_physical_action'] for r in rows),
            normal_original_map_minimum=sum(r['normal'].get('action_map_minimum',False) for r in rows),
            reverse_original_map_minimum=sum(r['reverse'].get('action_map_minimum',False) for r in rows),
            reverse_presented_map_minimum=sum(r['reverse_presented_map_minimum'] for r in rows),
            normal_saved_action_agreement=sum(r['normal_saved_action_agreement'] for r in rows),
            distance_order_changed=sum(r['distance_order_changed'] for r in rows),paired_safe_counts=dict(paired))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('manifest','checkpoint','model-path','source-path-map','saved-reference','config','out'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--device',default='cuda:0')
    args = parser.parse_args()
    torch.set_num_threads(1)
    protocol = json.loads(args.config.read_text())
    ready = json.loads(args.checkpoint.with_name('evaluation_ready.json').read_text())
    if ready['step'] not in protocol['checkpoint_steps']:
        raise ValueError('Checkpoint is outside the prescribed comparison')
    if (args.out/'summary.json').exists():
        raise FileExistsError('Completed content probe already exists')
    manifest,config,reader,tokenizer,device,environment = load_evaluation(
        args.manifest,args.checkpoint,args.model_path,source_path_map=args.source_path_map,device=args.device)
    records = [r for r in manifest['records'] if r['split']==protocol['split']]
    if len(records)!=protocol['cohort_tasks']:
        raise ValueError('Reference cohort differs')
    selected = select_records(records,protocol)
    baselines = {}
    for p in args.saved_reference.glob('*/cases.jsonl'):
        if '.interrupted' in p.parent.name or not (p.parent/'summary.json').exists():
            continue
        for line in p.read_text().splitlines():
            case=json.loads(line)
            if case['trajectory_id'] in baselines:
                raise ValueError('Duplicate saved reference task')
            baselines[case['trajectory_id']]=case
    args.out.mkdir(parents=True,exist_ok=True)
    results=[]
    with (args.out/'cases.jsonl').open('w') as handle:
        for record in selected:
            demo=load_record(args.manifest.parent/'trajectories',record,config,0)
            turns=probe_turns(demo,environment,protocol['stages'])
            normal=GenerationSession(reader,tokenizer,config,device)
            reverse=GenerationSession(reader,tokenizer,config,device)
            rows=[]
            for i,turn in enumerate(demo.turns):
                if i > max(turns):
                    break
                if i not in turns:
                    append_history(normal,turn.user_text,turn.step,turn.answer_text)
                    append_history(reverse,turn.user_text,turn.step,turn.answer_text)
                    continue
                step=turn.step
                saved=baselines[record['trajectory_id']]['turns'][i]
                if int(saved['current'])!=step.current or int(saved['goal'])!=step.goal or saved['candidates']!=len(step.candidate_actions):
                    raise ValueError('Saved baseline differs from prepared physical state')
                altered,sources=reverse_candidate_q(step)
                if not torch.equal(step.map_batch.vectors[:,:2],altered.map_batch.vectors[:,:2]):
                    raise AssertionError('Intervention changed current/goal content')
                if not all(torch.equal(getattr(step.map_batch,k),getattr(altered.map_batch,k)) for k in ('roles','candidate_ids','valid')):
                    raise AssertionError('Intervention changed slot addresses')
                before=environment.remaining(step.current,step.goal,turn.executed_path)
                reachable=environment.reachable_candidates(step,turn.executed_path,before)
                answer,usage=normal.ask(turn.user_text,step)
                other,other_usage=reverse.ask(turn.user_text,altered)
                score=scored_answer(step,answer,usage,environment,turn.executed_path,before,reachable)
                reverse_score=scored_answer(step,other,other_usage,environment,turn.executed_path,before,reachable)
                shown=score_turn(altered,other,reported=config['data'].get('reported_candidates'))
                rows.append(dict(turn=i,stage=turns[i],current=str(step.current),goal=str(step.goal),
                    candidates=len(step.candidate_actions),normal=score,reverse=reverse_score,
                    normal_answer=answer,reverse_answer=other,source_candidate_ids=[x+1 for x in sources],
                    changed_physical_action=score['chosen_action']!=reverse_score['chosen_action'],
                    normal_saved_action_agreement=score['chosen_action']==saved.get('chosen_action'),
                    reverse_presented_map_minimum=shown.get('action_map_minimum',False),
                    distance_order_changed=any(not np.isclose(a,b,rtol=1e-10,atol=1e-12)
                        for a,b in zip(step.candidate_map_distances,altered.candidate_map_distances))))
                # Both next decisions receive the original historical map and gold response.
                reverse.steps[-1]=step
                normal.accept(turn.answer_text);reverse.accept(turn.answer_text)
            case=dict(trajectory_id=record['trajectory_id'],group=record['group'],
                      shortest_moves=record['shortest_moves'],turns=rows)
            results.append(case);handle.write(json.dumps(case)+'\n');handle.flush()
            print(json.dumps({'completed_tasks':len(results),'pairs':sum(len(c['turns']) for c in results)}),flush=True)
    summary=dict(contract=protocol,step=ready['step'],epoch=ready['epoch'],tasks=len(results),
        recorded_utc=datetime.now(timezone.utc).isoformat(),strata=aggregate_pairs(results))
    if summary['strata']['all']['pairs']>protocol['maximum_pairs_per_checkpoint']:
        raise ValueError('Probe exceeded prescribed state budget')
    (args.out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')


if __name__=='__main__':
    main()
