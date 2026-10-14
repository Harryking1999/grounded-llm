"""Sample map Top-k policies on fixed tasks; exact solver only scores choices."""
import argparse
from collections import Counter
import json
from pathlib import Path
import random
import statistics
import time
import torch
from experiments.blocks_distance_map.src.oracle import DistanceOracle
from experiments.flamingo_map_reader.src.blocks import FrozenBoardMap
from .common import sorted_step, atomic_json
from .prepare import CachedMap


def choose_candidate(distances, top_k, rng):
    if top_k < 1 or not distances:
        raise ValueError('A positive shortlist and legal candidates are required')
    order=sorted(range(1,len(distances)+1),key=lambda i:(distances[i-1],i))
    return rng.choice(order[:top_k])


def rollout(record, qmap, top_k, seed, budget, oracle):
    state,goal=int(record['start']),int(record['goal'])
    # Independent per-task stream: reordering the cohort cannot change a task.
    rng=random.Random(f"{seed}:{record['trajectory_id']}")
    actions=[];states=[str(state)];first={}
    for _ in range(budget):
        if state==goal:break
        step=sorted_step(qmap,state,goal)
        if not step.candidate_actions:break
        candidate=choose_candidate(step.candidate_map_distances,top_k,rng)
        _,after=step.execute(candidate)
        if not actions:
            before=oracle.distance(state,goal);remaining=oracle.distance(after,goal)
            first=dict(first_safe=before>0 and remaining>=0,first_optimal=before>0 and remaining==before-1)
        actions.append(candidate);state=after;states.append(str(state))
    return dict(trajectory_id=record['trajectory_id'],arrived=state==goal,
        group=record.get('group'),steps=len(actions),actions=actions,states=states,**first)


def summarize(rows):
    n=len(rows)
    return dict(tasks=n,arrival=sum(r['arrived'] for r in rows)/n,
        first_safe=sum(r.get('first_safe',False) for r in rows)/n,
        first_optimal=sum(r.get('first_optimal',False) for r in rows)/n,
        by_group={g:sum(r['arrived'] for r in rows if r['group']==g)/count
                  for g,count in Counter(r['group'] for r in rows).items()})


def main():
    parser=argparse.ArgumentParser()
    for name in ('config','data','q-checkpoint','out','evaluation-root'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--source-commit',required=True)
    args=parser.parse_args();cfg=json.loads(args.config.read_text())
    if args.out.exists():raise ValueError('Use a fresh output directory')
    records=json.loads((args.data/'evaluation.json').read_text())
    ids={r['trajectory_id'] for r in records}
    if len(records)!=cfg['tasks'] or len(ids)!=len(records):raise ValueError('Wrong evaluation cohort')
    torch.set_num_threads(cfg['cpu_threads'])
    q=CachedMap(FrozenBoardMap.load(args.q_checkpoint));oracle=DistanceOracle(cache_limit=2000000,seconds=3600)
    args.out.mkdir(parents=True);began=time.monotonic();results={};task_probabilities={}
    for policy in cfg['policies']:
        seeds=policy.get('seeds',list(range(policy.get('seed_start',0),policy.get('seed_start',0)+policy.get('repeats',0))))
        summaries=[];success=Counter()
        with (args.out/(policy['name']+'_cases.jsonl')).open('w') as stream:
            for seed in seeds:
                rows=[rollout(r,q,policy['top_k'],seed,cfg['action_budget'],oracle) for r in records]
                for row in rows:
                    success[row['trajectory_id']]+=row['arrived']
                    stream.write(json.dumps(dict(row,seed=seed))+'\n')
                stream.flush();metrics=summarize(rows);summaries.append(dict(seed=seed,**metrics))
                print(json.dumps(dict(policy=policy['name'],seed=seed,**metrics)),flush=True)
                atomic_json(args.out/'progress.json',dict(policy=policy['name'],completed_repeats=len(summaries),total_repeats=len(seeds),seconds=time.monotonic()-began))
        probabilities={tid:success[tid]/len(seeds) for tid in ids};task_probabilities[policy['name']]=probabilities
        arrivals=[r['arrival'] for r in summaries]
        results[policy['name']]=dict(repeats=len(seeds),mean_arrival=statistics.mean(arrivals),
            std_across_seeds=statistics.stdev(arrivals) if len(arrivals)>1 else 0.,
            minimum=min(arrivals),maximum=max(arrivals),mean_first_safe=statistics.mean(r['first_safe'] for r in summaries),
            mean_first_optimal=statistics.mean(r['first_optimal'] for r in summaries),runs=summaries)
        if policy['top_k']==1 and round(arrivals[0]*len(records))!=cfg['expected_top1_arrivals']:
            raise ValueError('Top-1 does not reproduce the accepted map baseline')
    comparison=cfg['comparison'];cp=args.evaluation_root/comparison['baseline']/f"checkpoint-{comparison['step']}"
    llm=[json.loads(line) for path in cp.glob('RO_*/cases.jsonl') if (path.parent/'summary.json').exists() for line in path.read_text().splitlines()]
    if len(llm)!=len(ids) or {r['trajectory_id'] for r in llm}!=ids:
        raise ValueError('LLM comparison must cover the same complete tasks')
    matched={}
    for name,prob in task_probabilities.items():
        matched[name]=dict(llm_only=statistics.mean(float(r['arrived'])*(1-prob[r['trajectory_id']]) for r in llm),
            policy_only=statistics.mean(float(not r['arrived'])*prob[r['trajectory_id']] for r in llm),
            llm_minus_policy=statistics.mean(float(r['arrived'])-prob[r['trajectory_id']] for r in llm))
    atomic_json(args.out/'summary.json',dict(config=cfg,source_commit=args.source_commit,q_checkpoint=str(args.q_checkpoint),
        tasks=len(ids),policies=results,llm=dict(**comparison,arrival=sum(r['arrived'] for r in llm)/len(llm)),
        paired=matched,elapsed_seconds=time.monotonic()-began))
    atomic_json(args.out/'task_success_probabilities.json',task_probabilities)


if __name__=='__main__':main()
