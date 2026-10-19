"""Create a fixed 50,000-route token dataset from the saved physical task list."""

import argparse
from collections import Counter
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import random
import time

import numpy as np

from grounded_llm.tasks.blocks.protocol import (atomic_json, encode_messages, oracle_for, route_messages,
                       sample_routes)

WORK = {}


def init_worker(config, model_path):
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    from transformers import AutoTokenizer
    WORK.update(config=config, tokenizer=AutoTokenizer.from_pretrained(model_path, local_files_only=True))


def prepare_task(job):
    record, quota, encode = job
    config, tokenizer = WORK['config'], WORK['tokenizer']
    start, goal = int(record['start']), int(record['goal'])
    oracle = oracle_for(config)
    rng = random.Random(config['seed'] + record['sample_seed'])
    routes, unique_count, distance = sample_routes(start, goal, quota, rng, oracle,
                                                  config['data']['sampling_attempt_factor'])
    rows, seen, response_maximum = [], set(), 0
    for route, repeated in routes:
        messages, pairs = route_messages(start, goal, route, oracle)
        seen.update(pairs)
        item = dict(route=[list(key) for key in route], repeated=repeated)
        if encode:
            item['input_ids'], item['labels'] = encode_messages(messages, tokenizer)
            for message in messages[1::2]:
                # End token and following newline are included by the Qwen template.
                length = len(tokenizer.encode(message['content'] + '<|im_end|>\n', add_special_tokens=False))
                response_maximum = max(response_maximum, length)
        rows.append(item)
    return record, rows, unique_count, distance, seen, response_maximum


def quotas(records, config):
    ordinary = [r for r in records if int(r['start']) != int(r['goal'])]
    zero = [r for r in records if int(r['start']) == int(r['goal'])]
    spec = config['data']
    if len(ordinary) != spec['ordinary_tasks'] or len(zero) != spec['initial_goal_tasks']:
        raise ValueError(f'Source counts differ: ordinary={len(ordinary)}, initial_goal={len(zero)}')
    extra = set(random.Random(config['seed']).sample([r['trajectory_id'] for r in ordinary], spec['extra_route_tasks']))
    result = {r['trajectory_id']: spec['ordinary_base_routes'] + (r['trajectory_id'] in extra) for r in ordinary}
    result.update({r['trajectory_id']: 1 for r in zero})
    if sum(result.values()) != spec['samples_per_epoch']:
        raise ValueError('Route quotas do not sum to the declared epoch size')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('config', 'source-manifest', 'model-path', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--smoke-tasks', type=int)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding='utf-8'))
    source = json.loads(args.source_manifest.read_text(encoding='utf-8'))
    train = [r for r in source['records'] if r['split'] == 'train' and not r.get('no_solution')]
    allocation = quotas(train, config)
    if args.out.exists():
        raise FileExistsError(args.out)
    args.out.mkdir(parents=True)
    started = time.monotonic()
    if args.smoke_tasks:
        train = train[:args.smoke_tasks]
    rows, tasks, all_pairs, maximum_response = [], [], set(), 0
    offset, lengths = 0, []
    context = mp.get_context('spawn')
    with context.Pool(config['data']['workers'], initializer=init_worker,
                      initargs=(config, str(args.model_path))) as pool:
        jobs = [(r, allocation[r['trajectory_id']], True) for r in train]
        with (args.out/'input_ids.bin').open('wb') as ids_file, (args.out/'labels.bin').open('wb') as labels_file:
            for task_index, result in enumerate(pool.imap(prepare_task, jobs, chunksize=1)):
                record, prepared, unique_count, distance, pairs, response_maximum = result
                tasks.append(dict(record, quota=allocation[record['trajectory_id']], unique_routes=unique_count,
                                  certified_shortest_moves=distance))
                all_pairs.update(pairs)
                maximum_response = max(maximum_response, response_maximum)
                for item in prepared:
                    ids = np.asarray(item.pop('input_ids'), dtype='<i4')
                    labels = np.asarray(item.pop('labels'), dtype='<i4')
                    ids.tofile(ids_file)
                    labels.tofile(labels_file)
                    rows.append(dict(item, task_index=task_index, offset=offset, length=len(ids)))
                    lengths.append(len(ids))
                    offset += len(ids)
                if (task_index + 1) % 100 == 0 or task_index == 0:
                    elapsed = time.monotonic() - started
                    print(json.dumps(dict(phase='prepare_train', tasks=task_index+1, rows=len(rows),
                                          elapsed_seconds=elapsed, eta_seconds=elapsed/(task_index+1)*(len(train)-task_index-1))), flush=True)
        evaluation = []
        if not args.smoke_tasks:
            heldout = [r for r in source['records'] if r['split'] in ('validation', 'test') and not r.get('no_solution')]
            train_zero = {(int(r['start']), int(r['goal'])) for r in train if int(r['start']) == int(r['goal'])}
            initial_pairs = all_pairs | train_zero
            for result in pool.imap(prepare_task, [(r, 1, False) for r in heldout], chunksize=1):
                record, prepared, _, distance, pairs, _ = result
                evaluation.append(dict(record, reference_actions=prepared[0]['route'], certified_shortest_moves=distance,
                    sft_initial_pair_exposed=(int(record['start']), int(record['goal'])) in initial_pairs,
                    reference_nonterminal_pairs_exposed=sum(pair in all_pairs for pair in pairs)))
            if len(rows) != config['data']['samples_per_epoch']:
                raise ValueError('Incomplete training data; not publishing dataset')
    from transformers import AutoConfig
    context_limit = AutoConfig.from_pretrained(args.model_path, local_files_only=True).max_position_embeddings
    maximum_length = max(lengths)
    if maximum_length > context_limit:
        raise ValueError(f'Complete dialogue exceeds native context: {maximum_length} > {context_limit}')
    margin, rounding = config['evaluation']['generation_margin_tokens'], config['evaluation']['generation_round_to']
    generation_limit = math.ceil((maximum_response + margin) / rounding) * rounding
    audit = dict(samples=len(rows), physical_tasks=len(tasks), unique_routes=sum(t['unique_routes'] for t in tasks),
        repeated_rows=sum(r['repeated'] for r in rows), sequence_tokens_maximum=maximum_length,
        sequence_tokens_percentiles={str(q): float(np.percentile(lengths, q)) for q in (50, 90, 95, 99)},
        supervised_response_tokens_maximum=maximum_response, generation_limit=generation_limit,
        native_context_limit=context_limit, total_sequence_tokens=offset,
        evaluation_counts=dict(Counter(r['split'] + ('_initial_goal' if int(r['start']) == int(r['goal']) else '') for r in evaluation)),
        preparation_seconds=time.monotonic()-started, smoke=bool(args.smoke_tasks))
    atomic_json(args.out/'evaluation.json', evaluation)
    atomic_json(args.out/'dataset.json', dict(config=config, source_manifest=str(args.source_manifest),
        tasks=tasks, rows=rows, audit=audit))
    atomic_json(args.out/'summary.json', audit)
    atomic_json(args.out/'ready.json', dict(samples=len(rows), smoke=bool(args.smoke_tasks)))
    print(json.dumps(audit), flush=True)


if __name__ == '__main__':
    main()
