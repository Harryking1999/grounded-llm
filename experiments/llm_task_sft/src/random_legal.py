"""Exact arrival probabilities for uniform legal-action policies, without an LLM."""

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import time

from experiments.gcml_counterexamples.src import blocks as rules
from .files import atomic_json


def exact_probabilities(start, goal, budget, placements=None, max_states=2000000, seconds=120):
    """Evaluate the removal DAG; losing branches keep their probability mass.

    Policy A chooses uniformly among every geometrically legal placement.
    Policy B also rejects placements touching the goal. Both stop at the goal.
    Local support prunes only provably zero-probability subtrees; it never
    changes either policy's candidate denominator or selects a safe action.
    """
    started=time.monotonic()
    if goal & start != goal:
        return dict(all_legal=0.,goal_preserving=0.,states=0,seconds=0.,initial_candidates=0,initial_goal_preserving=0)
    tiles=[tile for tile in (placements if placements is not None else [p[0] for p in rules.PLACEMENTS])
           if tile & start == tile]
    if any(tile <= 0 for tile in tiles):
        raise ValueError('Each removal must clear at least one occupied cell')
    extra=start ^ goal
    active_initial=(1<<len(tiles))-1
    safe_initial=sum(1<<i for i,tile in enumerate(tiles) if not tile & goal)
    touches=defaultdict(int)
    for i,tile in enumerate(tiles):
        for cell in rules.cells(tile & extra):
            touches[cell] |= 1<<i
    disables=[0]*len(tiles)
    for i,tile in enumerate(tiles):
        disabled=0
        for cell in rules.cells(tile & extra): disabled |= touches[cell]
        disables[i]=disabled
    minimum=min((tile.bit_count() for tile in tiles),default=1)
    nonbinding=extra.bit_count()//minimum <= budget
    cache={}
    visits=0
    def probability(remaining,active,left):
        nonlocal visits
        if not remaining: return 1.,1.
        if left==0: return 0.,0.
        key=remaining if nonbinding else (remaining,left)
        if key in cache: return cache[key]
        visits+=1
        if len(cache)>=max_states or (visits%2048==0 and time.monotonic()-started>seconds):
            raise RuntimeError('Exact probability budget exhausted; result is unknown, not zero')
        safe=active & safe_initial
        if not safe or any(not (touches[cell] & safe) for cell in rules.cells(remaining)):
            cache[key]=(0.,0.);return cache[key]
        count_all,count_safe=active.bit_count(),safe.bit_count()
        total_all,total_safe=0.,0.
        while safe:
            bit=safe & -safe;safe ^= bit;i=bit.bit_length()-1
            child_all,child_safe=probability(remaining ^ tiles[i],active & ~disables[i],left-1)
            total_all+=child_all;total_safe+=child_safe
        result=total_all/count_all,total_safe/count_safe
        cache[key]=result
        return result
    all_legal,goal_preserving=probability(extra,active_initial,budget)
    return dict(all_legal=all_legal,goal_preserving=goal_preserving,states=len(cache),
                seconds=time.monotonic()-started,initial_candidates=len(tiles),
                initial_goal_preserving=safe_initial.bit_count())


def evaluate_record(payload):
    record,spec=payload
    values=exact_probabilities(int(record['start']),int(record['goal']),spec['action_budget'],
                              max_states=spec['max_states_per_task'],seconds=spec['seconds_per_task'])
    return dict(trajectory_id=record['trajectory_id'],initial_goal=record['start']==record['goal'],
                shortest_moves=record['certified_shortest_moves'],goal_empty=int(record['goal'])==0,**values)


def aggregate(rows):
    count=len(rows)
    return dict(tasks=count,
        expected_arrivals_all_legal=sum(row['all_legal'] for row in rows),
        arrival_rate_all_legal=sum(row['all_legal'] for row in rows)/count if count else None,
        expected_arrivals_goal_preserving=sum(row['goal_preserving'] for row in rows),
        arrival_rate_goal_preserving=sum(row['goal_preserving'] for row in rows)/count if count else None,
        mean_initial_candidates=sum(row['initial_candidates'] for row in rows)/count if count else None,
        mean_initial_goal_preserving=sum(row['initial_goal_preserving'] for row in rows)/count if count else None)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--smoke-tasks',type=int)
    args=parser.parse_args()
    spec=json.loads(args.config.read_text())
    if (args.out/'summary.json').exists(): raise FileExistsError('A completed diagnostic already exists')
    records=[record for record in json.loads((args.data/'evaluation.json').read_text()) if record['split']==spec['split']]
    if args.smoke_tasks: records=records[:args.smoke_tasks]
    args.out.mkdir(parents=True,exist_ok=True)
    started=time.monotonic();rows=[]
    with ProcessPoolExecutor(max_workers=spec['workers']) as pool,(args.out/'cases.jsonl').open('w') as stream:
        for row in pool.map(evaluate_record,[(record,spec) for record in records],chunksize=1):
            rows.append(row);stream.write(json.dumps(row)+'\n');stream.flush()
            if len(rows)%50==0 or len(rows)==1:
                atomic_json(args.out/'progress.json',dict(tasks=len(rows),total=len(records),elapsed_seconds=time.monotonic()-started))
    ordinary=[row for row in rows if not row['initial_goal']]
    zero=[row for row in rows if row['initial_goal']]
    groups=defaultdict(list)
    for row in ordinary: groups[str(row['shortest_moves'])].append(row)
    dataset=json.loads((args.data/'dataset.json').read_text())
    starts={task['start'] for task in dataset['tasks']};goals={task['goal'] for task in dataset['tasks']}
    pairs={(task['start'],task['goal']) for task in dataset['tasks']}
    ordinary_records=[record for record in records if record['start']!=record['goal']]
    result=dict(config=spec,smoke=bool(args.smoke_tasks),ordinary=aggregate(ordinary),initial_goal=aggregate(zero),
        by_shortest_moves={key:aggregate(value) for key,value in sorted(groups.items())},
        by_goal_type={key:aggregate([row for row in ordinary if row['goal_empty']==empty]) for key,empty in [('empty',True),('nonempty',False)]},
        source_groups=dict(Counter(record['group'] for record in ordinary_records)),
        initial_training_exposure=dict(start_seen=sum(record['start'] in starts for record in ordinary_records),
            goal_seen=sum(record['goal'] in goals for record in ordinary_records),
            pair_seen=sum((record['start'],record['goal']) in pairs for record in ordinary_records)),
        elapsed_seconds=time.monotonic()-started,total_evaluated_states=sum(row['states'] for row in rows),
        definitions=dict(all_legal='Uniform choice over all geometric candidates, no premature stop or format error.',
            goal_preserving='Uniform choice after rejecting goal-touching removals; no reachability or optimality filter.',
            terminal='Perfect recognition and stopping at the exact goal; initially solved tasks are reported separately.',
            probability='Dynamic programming sums all successful routes; no Monte Carlo sampling or retry selection.'))
    atomic_json(args.out/'summary.json',result)
    print(json.dumps(result),flush=True)


if __name__=='__main__': main()
