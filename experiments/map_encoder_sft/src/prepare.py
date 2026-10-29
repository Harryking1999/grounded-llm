"""Rebuild one matched five-task 50k set with the currently adopted maps."""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from itertools import chain
import json
from pathlib import Path
import shutil

import numpy as np
import torch
from transformers import AutoTokenizer

from grounded_llm.core.files import atomic_json
from grounded_llm.core.sft.transcript import encode_trajectory
from grounded_llm.tasks.state_graph_sft import StateTask, demonstration
from experiments.map_sft_baselines.src.generalization_prepare import (
    certify, query_key, block_queries, existing_queries, bridge_queries)


WORKER = None


def initialize(source, out, config, model_path, protected):
    global WORKER
    torch.set_num_threads(1)
    WORKER = (Path(source), Path(out), config,
              AutoTokenizer.from_pretrained(model_path, local_files_only=True), protected)


def map_record(row, config):
    row = dict(row, context=dict(row.get('context', {})))
    task = row['task']
    condition = config['map_conditions'][task]
    if condition.get('coordinate_frame'):
        row['context']['coordinate_frame'] = condition['coordinate_frame']
    return row


def stratum(row):
    return (row['task'], row['variant'], int(row.get('reference_cost', row.get('shortest_moves', 0))))


def physical_key(row):
    # Goal coordinates belong to inference, not the physical query identity.
    row = dict(row, context=dict(row.get('context', {})))
    row['context'].pop('coordinate_frame', None)
    return query_key(row)


def rebuild(item):
    index, row, original = item
    source, out, config, tokenizer, protected = WORKER
    row = map_record(row, config)
    name = f"{row['task']}_new_{index:07d}"
    destination = out / 'trajectories' / row['task'] / (name + '.pt')
    if destination.exists():
        return torch.load(destination, map_location='cpu', weights_only=False)['preparation']
    old_states = []
    if original:
        old = torch.load(source / 'trajectories' / row['prepared_file'], map_location='cpu', weights_only=False)['map']['steps']
        old_states = sorted({str(s.current) for s in old} | {str(c) for s in old for c in s.candidate_destinations})
    coverage = dict(task=row['task'], goal=str(row['goal']),
                    scene=json.dumps(row['context'].get('times', [])), old_states=old_states)
    rules = protected[row['task']]
    if row['task'] == 'bridge' and tuple(row['context']['times']) in rules['times']:
        return dict(accepted=False, reason='protected_times', **coverage)
    result = certify(row, out / row['task'])
    if result is None:
        return dict(accepted=False, reason='new_map_not_certified', **coverage)
    accepted, steps = result
    if row['task'] != 'bridge' and any((str(s.current), str(s.goal)) in rules['queries'] for s in steps[:-1]):
        return dict(accepted=False, reason='protected_relation', **coverage)
    # Preserve the original prompt when the start/goal query is reused.
    if row.get('initial_text'):
        accepted['initial_text'] = row['initial_text']
    accepted.update(trajectory_id=name, prepared_file=row['task'] + '/' + name + '.pt',
                    source_trajectory=row.get('trajectory_id', row.get('source_trajectory')))
    demo = demonstration(accepted, steps, with_map=True)
    encoded = encode_trajectory(demo, tokenizer, config['maximum_sequence_tokens'])
    destination.parent.mkdir(parents=True, exist_ok=True)
    states = {str(s.current) for s in steps} | {str(c) for s in steps for c in s.candidate_destinations}
    result = dict(accepted=True, row=dict(accepted, map_tokens=len(encoded.input_ids)),
                  states=sorted(states), **coverage)
    torch.save(dict(map=dict(encoded=encoded, steps=steps), preparation=result), destination)
    return result


def selection_checkpoint(selection_path, condition):
    selection_path = Path(selection_path)
    selected = json.loads(selection_path.read_text())
    for key in condition.get('selection_keys', []):
        selected = selected[key]
    path = Path(selected['checkpoint'])
    path = path if path.exists() else selection_path.parent / path
    if not path.is_file():
        raise FileNotFoundError(path)
    return path.resolve()


def prepare_layout(source, out, config, repository):
    for task, condition in config['map_conditions'].items():
        if not (source / task).exists():
            continue
        target = out / task
        target.mkdir(parents=True, exist_ok=True)
        if (source / task).exists():
            for asset in (source / task).iterdir():
                if asset.name not in ('evaluation.json', 'probe_indices.json', 'trajectories') and asset.suffix != '.pt':
                    dest = target / asset.name
                    if not dest.exists():
                        dest.symlink_to(asset.resolve(), target_is_directory=asset.is_dir())
        checkpoint = selection_checkpoint(repository / condition['selection'], condition)
        if not (target / 'qmap.pt').exists():
            shutil.copy2(checkpoint, target / 'qmap.pt')
        elif (target / 'qmap.pt').read_bytes() != checkpoint.read_bytes():
            raise ValueError(f'Existing {task} cache uses a different map')
    for asset in config.get('additional_map_assets', []):
        if not (source / asset['destination']).exists():
            continue
        checkpoint = selection_checkpoint(repository / asset['selection'], asset)
        destination = out / asset['destination']
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            shutil.copy2(checkpoint, destination)
        elif destination.read_bytes() != checkpoint.read_bytes():
            raise ValueError('Existing auxiliary map differs from its adopted condition')
    records = [map_record(row, config) for row in json.loads((source / 'evaluation.json').read_text())]
    atomic_json(out / 'evaluation.json', records)
    for task in {r['task'] for r in records}:
        atomic_json(out / task / 'evaluation.json', [r for r in records if r['task'] == task])
        atomic_json(out / task / 'probe_indices.json', [])
    atomic_json(out / 'probe_indices.json', [])
    if source.name == 'final_evaluation':
        atomic_json(out / 'ready.json', dict(tasks=len(records), smoke=False, final_only=True))


def main():
    parser = argparse.ArgumentParser(__doc__)
    for name in ('config', 'source', 'out', 'model-path'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--sources', type=Path, help='Existing FLAN source-pool paths for matched replacements')
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--smoke-per-task', type=int, default=0)
    parser.add_argument('--skip-final-evaluation', action='store_true',
                        help='Prepare training/ID first; final-only assets may be staged separately')
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    args.out.mkdir(parents=True, exist_ok=True)
    if (args.out / 'ready.json').exists():
        raise FileExistsError('Prepared cache is already complete; do not overwrite')
    prepare_layout(args.source, args.out, config, Path(__file__).resolve().parents[3])
    if not args.skip_final_evaluation and (args.source / 'final_evaluation').exists():
        prepare_layout(args.source / 'final_evaluation', args.out / 'final_evaluation',
                       config, Path(__file__).resolve().parents[3])
    source_meta = json.loads((args.source / 'dataset.json').read_text())
    rows = source_meta['rows']
    if len(rows) != config['data']['samples_per_epoch']:
        raise ValueError('Expected the complete fixed five-task source')
    if args.smoke_per_task:
        rows = [r for task in config['data']['tasks']
                for r in [x for x in rows if x['task'] == task][:args.smoke_per_task]]
    targets = Counter(map(stratum, rows))
    remaining = targets.copy()
    source_keys = {physical_key(row) for row in rows}
    seen = set(source_keys)
    selected, reasons = [], Counter()
    protected = {task: dict(queries=set(), times=set()) for task in config['data']['tasks']}
    original_pool = args.source.parent / 'pool'
    for task in protected:
        for path in [original_pool/task/'evaluation.json', original_pool/task/'small_ood.json']:
            if not path.exists():
                continue
            for row in json.loads(path.read_text()):
                protected[task]['queries'].add((str(row['start']),str(row['goal'])))
                if task == 'bridge':
                    protected[task]['times'].add(tuple(row['context']['times']))
    old_states, new_states = defaultdict(set), defaultdict(set)
    old_relations, new_relations = defaultdict(set), defaultdict(set)
    retained = Counter()
    attempts = 0

    def accept(result):
        task, scene, goal = result['task'], result['scene'], result['goal']
        old_states[task].update((scene,s) for s in result['old_states'])
        old_relations[task].update((scene,s,goal) for s in result['old_states'])
        if not result['accepted']:
            reasons[result['reason']] += 1
            return
        row = result['row']; key = stratum(row)
        if remaining[key] <= 0:
            reasons['surplus_difficulty_bin'] += 1
            return
        selected.append(row); remaining[key] -= 1
        new_states[task].update((scene,s) for s in result['states'])
        new_relations[task].update((scene,s,goal) for s in result['states'])
        if physical_key(row) in source_keys:
            retained[task] += 1

    def progress():
        state = dict(accepted=len(selected), required=len(rows), attempts=attempts,
                     remaining={'/'.join(map(str,k)):v for k,v in remaining.items() if v},
                     rejected=dict(reasons))
        atomic_json(args.out / 'progress.json', state)
        print(json.dumps(state),flush=True)

    with ProcessPoolExecutor(max_workers=args.workers, initializer=initialize,
            initargs=(str(args.source), str(args.out), config, str(args.model_path), protected)) as pool:
        for result in pool.map(rebuild, ((i,row,True) for i,row in enumerate(rows)), chunksize=4):
            attempts += 1; accept(result)
            if attempts % 500 == 0:
                progress()
        progress()
        sources = json.loads(args.sources.read_text()) if args.sources else {}
        rng = np.random.default_rng(config['data']['subset_seed'])
        banks = {}
        for task in config['data']['tasks']:
            path = original_pool/task/'dataset.json'
            bank = json.loads(path.read_text())['rows'] if path.exists() else []
            bank = [bank[int(i)] for i in rng.permutation(len(bank))]
            source = sources.get('existing',{}).get(task)
            if source and task == 'blocks':
                fallback = block_queries(Path(source),rng)
            elif source:
                fallback = existing_queries(task,Path(source),rng)
            elif task == 'bridge':
                fallback = bridge_queries(rng,20)
            else:
                fallback = iter(())
            banks[task] = chain(bank,fallback)
        index = len(rows)
        maximum = config['data']['regeneration']['maximum_candidate_queries']
        for task in config['data']['tasks']:
            while any(v for key,v in remaining.items() if key[0] == task):
                jobs=[]
                while len(jobs) < args.workers * 2:
                    try:
                        row=next(banks[task])
                    except StopIteration:
                        break
                    if not any(v for key,v in remaining.items() if key[:2] == (task,row['variant'])):
                        continue
                    # Known costs can be screened before running the new map.
                    if ('reference_cost' in row or 'shortest_moves' in row) and remaining[stratum(row)] <= 0:
                        continue
                    key=physical_key(row)
                    if key in seen:
                        continue
                    seen.add(key); jobs.append((index,row,False)); index+=1
                if not jobs:
                    raise ValueError(f'Insufficient distinct certified queries for {task}; preserve quotas and inspect progress')
                for result in pool.map(rebuild,jobs,chunksize=1):
                    attempts+=1;accept(result)
                if attempts > maximum:
                    raise ValueError('Candidate budget exhausted; do not relax certification or duplicate queries')
                progress()
    new_rows=selected
    actual=Counter(map(stratum,new_rows))
    if actual != targets or len({physical_key(r) for r in new_rows}) != len(rows):
        raise ValueError('Regenerated data must match task/variant/cost quotas without duplicate queries')
    atomic_json(args.out/'dataset.json',dict(source_meta,rows=new_rows))
    def histogram(counter):
        return {'/'.join(map(str,k)):v for k,v in sorted(counter.items())}
    coverage={task:dict(old_states=len(old_states[task]),new_states=len(new_states[task]),
        shared_states=len(old_states[task]&new_states[task]),
        old_state_goal_relations=len(old_relations[task]),new_state_goal_relations=len(new_relations[task]),
        shared_state_goal_relations=len(old_relations[task]&new_relations[task])) for task in config['data']['tasks']}
    atomic_json(args.out/'acceptance.json',dict(samples=len(new_rows),
        by_task=dict(Counter(r['task'] for r in new_rows)),
        physical_shortest_map_greedy_verified=True, unique_queries=len(new_rows),
        original_query_retained=dict(retained),candidate_queries=attempts,rejected=dict(reasons),
        original_task_variant_cost=histogram(targets),new_task_variant_cost=histogram(actual),
        coverage=coverage,source=str(args.source),evaluation_queries_unchanged=True))
    atomic_json(args.out/'ready.json',dict(samples=len(new_rows),smoke=bool(args.smoke_per_task)))


if __name__ == '__main__':
    main()
