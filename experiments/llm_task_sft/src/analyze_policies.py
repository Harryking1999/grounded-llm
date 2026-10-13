"""Compare saved map/SFT policies by physical outcomes and first fatal decisions."""

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path

from experiments.blocks_distance_map.src.oracle import DistanceOracle
from experiments.gcml_counterexamples.src import blocks as rules
from .files import atomic_json


def read_cases(root):
    cases = []
    for path in sorted(root.glob('*/cases.jsonl')):
        if '.interrupted' in path.parent.name or not (path.parent/'summary.json').exists():
            continue
        cases.extend(json.loads(line) for line in path.read_text().splitlines())
    result = {c['trajectory_id']: c for c in cases}
    if len(result) != len(cases):
        raise ValueError(f'Duplicate task IDs in {root}')
    return result


def physical_action(action):
    if action is None:
        return None
    if isinstance(action, int):
        action = rules.PLACEMENTS[action][1]
    return tuple(action[k] for k in ('shape_id', 'row', 'col'))


def action_dict(action):
    return dict(zip(('shape_id', 'row', 'col'), action))


def normalize(case, record, kind):
    """Keep actual execution distinct from an unexecuted proposed action."""
    moves = case['executed_actions' if kind == 'sft' else 'actual_actions']
    moves = [physical_action(action) for action in moves]
    start, goal = int(record['start']), int(record['goal'])
    states = [start]
    for move in moves:
        states.append(rules.apply(states[-1], action_dict(move)))
    arrived = goal in states
    reported = case['arrived' if kind == 'sft' else 'reached_goal']
    if arrived != reported:
        raise ValueError('Saved arrival differs from physical action replay')
    rows = []
    for row in case['turns']:
        index = row['index' if kind == 'sft' else 'turn']
        before = int(row['before_state' if kind == 'sft' else 'current'])
        if before != states[min(index, len(moves))]:
            raise ValueError('Saved turn differs from physical action replay')
        remaining = row['before_distance' if kind == 'sft' else 'remaining_shortest']
        error = row.get('control_error') if kind == 'sft' else None
        if kind == 'map':
            if row.get('context_exhausted'):
                error = 'context_overflow'
            elif row.get('declared_no_solution'):
                error = 'false_no_solution'
            elif not row['valid_control']:
                error = 'control_format'
            elif row.get('premature_done'):
                error = 'premature_stop'
            elif not row['done'] and not row['legal_action']:
                error = 'illegal_action'
        if case.get('failure') in ('action_budget', 'context_budget') and index == len(moves):
            error = case['failure']
        executed = moves[index] if index < len(moves) else None
        proposed = (physical_action(row.get('action')) if kind == 'sft'
                    else physical_action(row.get('chosen_action')))
        rows.append(dict(step=index+1, before=before, after=states[index+1] if executed else before,
                         remaining=remaining, live=remaining > 0, goal=before == goal,
                         action=executed, proposed=proposed, error=error,
                         safe=bool(row.get('reachability' if kind == 'sft' else 'action_keeps_goal_reachable')),
                         shortest=bool(row.get('shortest_progress' if kind == 'sft' else 'action_environment_shortest')),
                         reason_error=row.get('reason_error'), text=row.get('text',row.get('answer'))))
    return dict(id=case['trajectory_id'], group=record['group'], length=record['shortest_moves'],
                start=start, goal=goal, moves=moves, rows=rows, arrived=arrived,
                stopped=case['arrived_and_stopped' if kind == 'sft' else 'success'],
                arrival_step=states.index(goal) if arrived else None, ending=case.get('failure'))


def first_failure(case, oracle):
    """A later stop on an already-dead board is not a second causal failure."""
    for row in case['rows']:
        if not row['live']:
            continue
        if oracle.distance(row['before'], case['goal']) != row['remaining']:
            raise ValueError('Saved live-state distance disagrees with exact oracle')
        if row['action'] is not None:
            after = row['after']
            distance = oracle.distance(after, case['goal'])
            if row['safe'] != (distance >= 0):
                raise ValueError('Saved reachability disagrees with exact oracle')
            if distance < 0:
                kind = 'remove_goal_cells' if after & case['goal'] != case['goal'] else 'untileable_residue'
                # Local support separates an obvious orphan from globally incompatible tiling.
                detail = None if kind == 'remove_goal_cells' else (
                    'unsupported_cell' if not rules.locally_supported(after ^ case['goal']) else 'global_tiling_conflict')
                return dict(step=row['step'], kind=kind, detail=detail)
        elif row['error']:
            return dict(step=row['step'], kind=row['error'], detail=None)
        elif not row['safe']:
            raise ValueError('Live failed decision lacks an action or control failure')
    if not case['arrived']:
        raise ValueError('Failed episode has no identifiable first failure')
    return None


def rate(numerator, denominator):
    return dict(numerator=numerator, denominator=denominator,
                rate=numerator/denominator if denominator else None)


def summarize(cases, oracle):
    fatal, roots, details, endings, rounds = {}, Counter(), Counter(), Counter(), defaultdict(list)
    arrival_steps, first_loss, reason_context = Counter(), Counter(), defaultdict(Counter)
    for case in cases.values():
        failure = first_failure(case, oracle)
        fatal[case['id']] = failure
        if failure:
            roots[failure['kind']] += 1
            first_loss[failure['step']] += 1
            if failure['detail']:
                details[failure['detail']] += 1
        if case['arrived']:
            arrival_steps[case['arrival_step']] += 1
        endings[case['ending'] or 'success'] += 1
        for row in case['rows']:
            rounds[row['step']].append(row)
            context = 'live' if row['live'] else 'goal' if row['goal'] else 'already_dead'
            reason_context[context]['turns'] += 1
            if row['reason_error']:
                reason_context[context][row['reason_error']] += 1
    n = len(cases)
    if sum(roots.values()) + sum(c['arrived'] for c in cases.values()) != n:
        raise ValueError('First failures and arrivals do not partition the cohort')
    steps = {}
    for step, rows in sorted(rounds.items()):
        live = [r for r in rows if r['live']]
        lost = sum(v for k,v in first_loss.items() if k <= step)
        reached = sum(v for k,v in arrival_steps.items() if k <= step)
        types = Counter(f['kind'] for f in fatal.values() if f and f['step']==step)
        steps[str(step)] = dict(live=len(live), safe=sum(r['safe'] for r in live),
            shortest=sum(r['shortest'] for r in live), first_fatal=first_loss[step], fatal_types=dict(types),
            cumulative_fatal=lost, cumulative_arrived=reached, active_after=n-lost-reached,
            already_dead_turns=sum(r['remaining']<0 for r in rows),
            reason_fact_errors_live=sum(r['live'] and r['reason_error']=='fact' for r in rows))
    live = [r for c in cases.values() for r in c['rows'] if r['live']]
    first = [c['rows'][0] for c in cases.values()]
    reached = sum(c['arrived'] for c in cases.values())
    result = dict(tasks=n, arrival=rate(reached,n), arrival_and_stop=rate(sum(c['stopped'] for c in cases.values()),n),
        first_safe=rate(sum(r['safe'] for r in first),n), live_safe=rate(sum(r['safe'] for r in live),len(live)),
        live_shortest=rate(sum(r['shortest'] for r in live),len(live)), root_errors=dict(roots),
        residue_errors=dict(details), ending_labels=dict(endings), by_step=steps,
        reason_errors_by_context={k:dict(v) for k,v in reason_context.items()},
        mean_live_decisions=len(live)/n,
        mean_actions_arrived=sum(len(c['moves']) for c in cases.values() if c['arrived'])/reached if reached else None,
        by_length={}, by_group={})
    for field, target in (('length','by_length'),('group','by_group')):
        for value in sorted({c[field] for c in cases.values()}):
            group = [c for c in cases.values() if c[field]==value]
            result[target][str(value)] = dict(tasks=len(group), arrived=sum(c['arrived'] for c in group),
                root_errors=dict(Counter(fatal[c['id']]['kind'] for c in group if fatal[c['id']])) )
    return result, fatal


def compare(left, right, failures_left, failures_right):
    if left.keys() != right.keys():
        raise ValueError('Policy comparison requires exactly the same task IDs')
    ids = set(left)
    a = {k for k in ids if left[k]['arrived']}
    b = {k for k in ids if right[k]['arrived']}
    both = a & b
    first_eligible = [k for k in ids if left[k]['moves'] and right[k]['moves']]
    same_first = sum(left[k]['moves'][0]==right[k]['moves'][0] for k in first_eligible)
    detail = Counter()
    prefix = 0
    for k in ids:
        if (left[k]['start'],left[k]['goal']) != (right[k]['start'],right[k]['goal']):
            raise ValueError('Matched task IDs have different physical boards')
        for x,y in zip(left[k]['moves'],right[k]['moves']):
            if x != y:
                break
            prefix += 1
        if k not in a|b:
            fa,fb = failures_left[k],failures_right[k]
            detail[f"{fa['kind']} / {fb['kind']}"] += 1
    strata = {}
    for length in sorted({c['length'] for c in left.values()}):
        selected = {k for k in ids if left[k]['length']==length}
        strata[str(length)] = dict(tasks=len(selected), both_arrive=len(selected&both),
            left_only=len(selected&(a-b)),right_only=len(selected&(b-a)),neither=len(selected-(a|b)))
    return dict(tasks=len(ids), both_arrive=len(both), left_only=len(a-b), right_only=len(b-a),
        neither=len(ids-(a|b)), success_set_jaccard=len(both)/len(a|b) if a|b else None,
        identical_first_physical_action=rate(same_first,len(first_eligible)),
        identical_removal_partition_among_both_arrived=rate(sum(set(left[k]['moves'])==set(right[k]['moves']) for k in both),len(both)),
        mean_identical_action_prefix=prefix/len(ids), failure_type_pairs=dict(detail), by_length=strata)


def reference_summary(cases, kind):
    errors = Counter()
    first_safe = 0
    total = safe = 0
    for case in cases.values():
        live = [r for r in case['turns'] if r['before_distance' if kind=='sft' else 'remaining_shortest']>0]
        key = 'reachability' if kind=='sft' else 'action_keeps_goal_reachable'
        bad = sum(not r.get(key,False) for r in live)
        errors[bad] += 1
        total += len(live);safe += len(live)-bad
        first_safe += bool(case['turns'][0].get(key,False))
    return dict(tasks=len(cases), live_safe=rate(safe,total), first_safe=rate(first_safe,len(cases)),
                errors_per_task=dict(sorted(errors.items())), tasks_with_no_live_decision_errors=errors[0])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('config','sft-run','map-run','map-cases','out'):
        parser.add_argument('--'+name,type=Path,required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    records = {r['trajectory_id']:r for r in json.loads((args.sft_run/'data_50k/evaluation.json').read_text()) if r['split']=='test'}
    manifest = json.loads((args.map_run/'data/manifest.json').read_text())
    other = {r['trajectory_id']:r for r in manifest['records'] if r['split']=='test'}
    if len(records)!=config['tasks'] or records.keys()!=other.keys():
        raise ValueError('Physical test cohorts differ')
    if any((records[k]['start'],records[k]['goal'])!=(other[k]['start'],other[k]['goal']) for k in records):
        raise ValueError('Physical test boards differ')
    oracle = DistanceOracle(cache_limit=config['oracle_cache_limit'],seconds=config['oracle_seconds'])
    result = dict(contract=config,recorded_utc=datetime.now(timezone.utc).isoformat(),
                  conditions={},comparisons={},checks={'same_510_ids_and_boards':True,'physical_action_replay':True,'exact_first_failure':True})
    normalized, failures = {}, {}
    jobs = [('map',step,args.map_cases/f'evaluation_half_epoch/step-{step}') for step in config['map_steps']]
    jobs.extend(('sft',epoch,args.sft_run/f'continuation/evaluation/epoch_{epoch:g}/test') for epoch in config['sft_epochs'])
    for kind, point, root in jobs:
        label=f'{kind}_{point:g}'
        rollout = read_cases(root/'test_rollout' if kind=='map' else root/'rollout')
        reference = read_cases(root/'test_reference' if kind=='map' else root/'reference')
        if rollout.keys()!=records.keys() or reference.keys()!=records.keys():
            raise ValueError(f'{label}: incomplete saved cohort')
        normalized[label] = {k:normalize(c,records[k],kind) for k,c in rollout.items()}
        summary, failures[label] = summarize(normalized[label],oracle)
        summary['reference'] = reference_summary(reference,kind)
        result['conditions'][label]=summary
    for step in config['map_steps']:
        for epoch in config['sft_epochs']:
            a,b=f'map_{step}',f'sft_{epoch:g}'
            result['comparisons'][a+'/'+b]=compare(normalized[a],normalized[b],failures[a],failures[b])
    atomic_json(args.out/'summary.json',result)
    # Per-task witnesses stay outside Git with the original generations.
    atomic_json(args.out/'cases.json',{label:[dict(id=k,arrived=c['arrived'],fatal=failures[label][k],
        first_action=c['moves'][0] if c['moves'] else None,arrival_step=c['arrival_step']) for k,c in values.items()]
        for label,values in normalized.items()})
    print(json.dumps({'out':str(args.out),'checks':result['checks'],'conditions':list(result['conditions'])}))


if __name__ == '__main__':
    main()
