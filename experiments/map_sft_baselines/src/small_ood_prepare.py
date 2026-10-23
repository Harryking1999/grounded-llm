"""Pair FLAN validation starts with simple new goals; keep ID and OOD labels."""
import argparse
from collections import Counter
import json
from pathlib import Path
import shutil

import numpy as np

from grounded_llm.core.files import atomic_json
from grounded_llm.tasks.blocks import rules
from grounded_llm.tasks.blocks.protocol import action_from_key
from grounded_llm.tasks.restoration_sft import environment


def block_goal(record):
    actions = record['reference_actions']
    if len(actions) < 2 or int(record['goal']) != 0:
        raise ValueError('Expected a nontrivial original clearing route')
    state = int(record['start'])
    for action in actions[:-1]:
        state = rules.apply(state, action_from_key(action))
    if not state or rules.apply(state, action_from_key(actions[-1])) != 0:
        raise ValueError('Goal must be exactly one removable block')
    # A shorter prefix plus the final action would contradict the certified
    # shortest original route. No Q filtering or new teacher is involved.
    return dict(record, goal=str(state), reference_actions=actions[:-1],
                shortest_moves=len(actions)-1, length=len(actions)-1,
                goal_distance_from_solved=1)


def near_goals(env, distances):
    layers = {0: {0}}
    for depth in range(1, max(distances)+1):
        children = env.successors(sorted(layers[depth-1])).ravel()
        layers[depth] = set(map(int, children[children >= 0])) - set.union(*layers.values())
    return {depth: sorted(layers[depth]) for depth in distances}


def restoration_goals(task, records, source, distances, rng):
    env = environment(task, source/'oracle.npy')
    pools = near_goals(env, distances)
    if task == 'cube2' and env.depth is None:
        raise ValueError('Use the existing exact cube oracle')
    if task == 'puzzle8':
        env.prepare_goals([0, *sorted(set.union(*(set(p) for p in pools.values())))])
    assignment = rng.permutation(len(records))
    rows = []
    for order, index in enumerate(assignment):
        record = records[int(index)]
        depth = distances[order % len(distances)]
        eligible = [g for g in pools[depth] if g != int(record['start'])]
        goal = int(rng.choice(eligible))
        state, actions = int(record['start']), []
        length = int(env.distance(state, goal))
        while state != goal:
            children = env.successors([state])[0]
            legal = np.flatnonzero(children >= 0)
            remaining = env.distance(children[legal], goal)
            action = int(legal[np.flatnonzero(remaining == env.distance(state, goal)-1)[0]])
            actions.append(env.actions[action]);state = int(children[action])
        if len(actions) != length or int(env.distance(goal, 0)) != depth:
            raise ValueError('Exact goal or reference distance mismatch')
        rows.append((int(index), dict(record, goal=goal, reference_actions=actions,
            length=length, shortest_moves=length, goal_distance_from_solved=depth)))
    return [row for _, row in sorted(rows)]


def prepare(config, source, out, cube_small_ood_checkpoint=None):
    if out.exists():
        raise ValueError('Use a fresh evaluation data directory')
    cube_map = config['small_ood'].get('cube2', {}).get('map_checkpoint')
    if cube_map and (cube_small_ood_checkpoint is None or not cube_small_ood_checkpoint.is_file()):
        raise ValueError('Configured cube small OOD map requires --cube-small-ood-checkpoint')
    rng = np.random.default_rng(config['seed'])
    all_rows, audit = [], {}
    for task in config['original_tasks']:
        origin = source/task
        records = json.loads((origin/'evaluation.json').read_text())
        if len(records) != config['tasks_per_cohort']:
            raise ValueError(f'{task}: original validation cohort mismatch')
        original = [dict(r, task=task, cohort='original',
                         source_trajectory_id=r['trajectory_id']) for r in records]
        rows = list(original)
        if task in config['small_ood']:
            new = ([block_goal(r) for r in records] if task == 'blocks' else
                   restoration_goals(task, records, origin,
                       config['small_ood'][task]['goal_distances_from_solved'], rng))
            rows.extend(dict(r, task=task, cohort='small_ood',
                source_trajectory_id=r['trajectory_id'],
                trajectory_id=r['trajectory_id']+'_small_ood') for r in new)
            if task == 'cube2' and cube_map:
                for row in rows[len(original):]:
                    row['map_checkpoint'] = cube_map
                    row['map_identity'] = config['small_ood'][task]['map_identity']
            audit[task] = dict(tasks=len(new), paired_starts=True,
                unique_goals=len({r['goal'] for r in new}),
                goal_depths=dict(Counter(r['goal_distance_from_solved'] for r in new)),
                reference_lengths=dict(Counter(len(r['reference_actions']) for r in new)))
        destination = out/task
        destination.mkdir(parents=True)
        if task == 'cube2' and cube_map:
            shutil.copy2(cube_small_ood_checkpoint, destination/cube_map)
        atomic_json(destination/'evaluation.json', rows)
        atomic_json(destination/'probe_indices.json', [])
        for name in ('qmap.pt', 'oracle.npy', 'graph.npz'):
            path = origin/name
            if path.exists():
                shutil.copy2(path, destination/name)
        all_rows.extend(rows)
    if len(all_rows) != config['evaluation']['tasks']:
        raise ValueError('Combined evaluation size mismatch')
    if len({r['trajectory_id'] for r in all_rows}) != len(all_rows):
        raise ValueError('Duplicate evaluation IDs')
    atomic_json(out/'evaluation.json', all_rows)
    atomic_json(out/'probe_indices.json', [])
    atomic_json(out/'acceptance.json', dict(config=config, source=str(source),
        original_tasks=sum(r['cohort']=='original' for r in all_rows),
        small_ood_tasks=sum(r['cohort']=='small_ood' for r in all_rows), by_task=audit,
        limitation='SFT goal-condition OOD; no claim of unseen Qmap relations. '
                   'Near-solved targets do not imply short start-to-goal routes.'))


def map_reference(data, out, action_budget, source_commit):
    import torch
    from grounded_llm.core.qmap.reader import CachedQMap
    from grounded_llm.tasks.blocks.qmap import FrozenBoardMap
    from grounded_llm.tasks.blocks.sft import sorted_step
    from grounded_llm.tasks.restoration_sft import load_map, make_step
    if out.exists():
        raise ValueError('Use a fresh map reference output')
    torch.set_num_threads(2)
    out.mkdir(parents=True)
    rows = []
    with (out/'cases.jsonl').open('w') as stream:
        for task in ('blocks', 'cube2', 'puzzle8', 'path'):
            source = data/task
            env = None if task == 'blocks' else environment(task, source/'oracle.npy')
            reader = (CachedQMap(FrozenBoardMap.load(source/'qmap.pt')) if task == 'blocks'
                      else load_map(task, env, source/'qmap.pt'))
            readers = {'qmap.pt': reader}
            records = json.loads((source/'evaluation.json').read_text())
            for record in records:
                checkpoint = record.get('map_checkpoint', 'qmap.pt')
                if checkpoint not in readers:
                    readers[checkpoint] = load_map(task, env, source/checkpoint)
                reader = readers[checkpoint]
                state, goal = int(record['start']), int(record['goal'])
                actions = []
                for _ in range(action_budget):
                    if state == goal:
                        break
                    step = (sorted_step(reader, state, goal) if task == 'blocks' else
                            make_step(env, reader, state, goal))
                    if not step.candidate_actions:
                        break
                    candidate = int(np.argmin(step.candidate_map_distances))+1
                    action, state = step.execute(candidate)
                    actions.append(action)
                row = dict(trajectory_id=record['trajectory_id'], task=task,
                    cohort=record['cohort'], arrived=state==goal, actions=actions,
                    steps=len(actions), final_state=str(state))
                row['map_checkpoint'] = str(source/checkpoint)
                if 'map_identity' in record:
                    row['map_identity'] = record['map_identity']
                rows.append(row);stream.write(json.dumps(row)+'\n');stream.flush()
            print(json.dumps(dict(task=task, completed=len(records))), flush=True)
            atomic_json(out/'progress.json', dict(completed=len(rows)))
    groups = {}
    for cohort in ('original', 'small_ood'):
        groups[cohort] = {}
        for task in ('blocks', 'cube2', 'puzzle8', 'path'):
            selected = [r for r in rows if r['cohort']==cohort and r['task']==task]
            if selected:
                groups[cohort][task] = dict(tasks=len(selected),
                    arrival=sum(r['arrived'] for r in selected))
    atomic_json(out/'summary.json', dict(policy='frozen_greedy_top1',
        source_commit=source_commit, action_budget=action_budget, by_cohort=groups,
        interpretation='Empirical greedy reference, not a mathematical upper bound for LLMs.'))


def main():
    parser = argparse.ArgumentParser(__doc__)
    for name in ('config', 'source', 'out'):
        parser.add_argument('--'+name, required=True, type=Path)
    parser.add_argument('--map-reference-out', type=Path)
    parser.add_argument('--source-commit')
    parser.add_argument('--cube-small-ood-checkpoint', type=Path)
    args = parser.parse_args()
    config=json.loads(args.config.read_text())
    prepare(config, args.source, args.out, args.cube_small_ood_checkpoint)
    if args.map_reference_out:
        configs=[json.loads((args.config.parent/name).read_text())
                 for name in config['model_configs'].values()]
        budgets={c['evaluation']['action_budget'] for c in configs}
        if len(budgets)!=1 or not args.source_commit:
            raise ValueError('Map reference requires matched action budgets and source commit')
        map_reference(args.out, args.map_reference_out, budgets.pop(), args.source_commit)


if __name__ == '__main__':
    main()
