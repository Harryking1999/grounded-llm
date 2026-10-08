"""Preserve 9000 prepared trajectories and add 1000 masked failure histories."""

from argparse import ArgumentParser
from collections import Counter
import json
import multiprocessing as mp
from pathlib import Path

import numpy as np
import torch

from experiments.blocks_distance_map.src.multiboard_data import official_boards
from experiments.blocks_distance_map.src.oracle import DistanceOracle
from .blocks import FrozenBoardMap
from .blocks_data import reachable_pairs, demonstration_from_record
from .blocks_failure import failure_demonstration
from .trajectory_dataset import prepare_record, unpack_demo


WORK = {}


def physical_pairs(demo):
    return {(t.step.current, t.step.goal) for t in demo.turns}


def read_pairs(record):
    value = torch.load(WORK['old_root'] / record['prepared_file'], weights_only=True)
    return physical_pairs(unpack_demo(value['demo'], 'blocks'))


def failure_candidates(board):
    config = WORK['config']
    oracle = DistanceOracle(cache_limit=2_000_000, seconds=7200)
    rng = np.random.default_rng(config['seed'] + board[0])
    pairs = reachable_pairs(board, config['data']['sampling_rollouts'], rng, oracle, 6)
    found = []
    for start, goal, distance in pairs:
        if (start, goal) in WORK['excluded']:
            continue
        record = dict(board_row=board[0], start=str(start), goal=str(goal),
                      shortest_moves=distance, sample_seed=int(rng.integers(2**32)))
        demo = failure_demonstration(WORK['qmap'], record,
            max_actions=config['maximum_demonstration_actions'],
            reported=config['data']['reported_candidates'])
        if demo is None or physical_pairs(demo) & WORK['excluded']:
            continue
        found.append((record, demo))
        if len(found) == 3:
            break
    return found


def prepare_failure(job):
    record, demo = job
    config = WORK['config']
    if record['split'] == 'train' and 'failure_numbering_variants' in config['data']:
        config = dict(config, data=dict(config['data'],
            numbering_variants=config['data']['failure_numbering_variants']))
    return prepare_record(demo, record, config, WORK['tokenizer'],
                           WORK['root'] / (record['trajectory_id'] + '.pt'))


def main():
    parser = ArgumentParser()
    for name in ('config', 'source-manifest', 'model-path', 'out'):
        parser.add_argument('--' + name, required=True, type=Path)
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    torch.set_num_threads(1)
    config = json.loads(args.config.read_text())
    old = json.loads(args.source_manifest.read_text())
    if args.out.exists():
        raise FileExistsError(args.out)
    root = args.out / 'trajectories'
    root.mkdir(parents=True)
    old_root = args.source_manifest.parent / 'trajectories'
    train = [r for r in old['records'] if r['split'] == 'train']
    if len(train) != 9000 or any(not r['greedy_success'] for r in train):
        raise ValueError('Expected precisely 9000 successful source trajectories')
    if old['config']['data']['numbering_variants'] != config['data']['numbering_variants']:
        raise ValueError('Preserved encodings require the same numbering augmentation')
    validation = [r for r in old['records'] if r['split'] == 'validation']
    WORK.update(old_root=old_root, config=config, root=root)
    all_old_pairs = set()
    train_pairs = set()
    with mp.get_context('fork').Pool(args.workers) as pool:
        for r, pairs in zip(old['records'], pool.imap(read_pairs, old['records'], chunksize=16)):
            all_old_pairs.update(pairs)
            if r['split'] == 'train':
                train_pairs.update(pairs)
    print(json.dumps(dict(phase='audit', train=len(train), failed_train=0,
                         rejected_source=old.get('rejected_training'),
                         training_state_goal_pairs=len(train_pairs))), flush=True)
    options = config['data']
    boards, fresh = official_boards(Path(old['official_data']), options['board_seed'],
        options['training_boards'], options['validation_new_boards'] + options['test_new_boards'])
    qmap = FrozenBoardMap.load(Path(old['q_checkpoint']))
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    WORK.update(qmap=qmap, tokenizer=tokenizer)
    rng = np.random.default_rng(config['seed'])
    oracle = DistanceOracle(cache_limit=4_000_000, seconds=14400)
    selected = []
    new_demos = {}
    excluded = set(all_old_pairs)
    for group, source_boards in (
        ('same_initial_board_new_pair', boards),
        ('new_initial_board', fresh[options['validation_new_boards']:])):
        quota = {int(k): v for k, v in options['test_length_counts_per_group'].items()}
        for r in old['records']:
            if r['split'] == 'test' and r['group'] == group and quota.get(r['shortest_moves'], 0):
                selected.append(r)
                quota[r['shortest_moves']] -= 1
        for board in source_boards:
            if not any(quota.values()):
                break
            pool = reachable_pairs(board, options['sampling_rollouts'], rng, oracle, 6)
            for start, goal, distance in pool:
                if not quota.get(distance, 0) or (start, goal) in excluded:
                    continue
                record = dict(split='test', group=group, board_row=board[0], start=str(start),
                    goal=str(goal), shortest_moves=distance, sample_seed=int(rng.integers(2**32)),
                    trajectory_id=f'balanced_test_{len(new_demos):06d}')
                demo = demonstration_from_record(qmap, record, config['maximum_demonstration_actions'],
                                                options['reported_candidates'])
                # Initial tasks must not be a supervised training suffix. Greedy outcome
                # never controls acceptance. Later trajectories may legitimately merge.
                new_demos[record['trajectory_id']] = demo
                selected.append(record)
                excluded.update(physical_pairs(demo))
                excluded.add((start, goal))
                quota[distance] -= 1
                break  # Spread the new short tasks over different boards.
        if any(quota.values()):
            raise ValueError(f'Insufficient evaluation tasks for {group}: {quota}')
    records = []
    # One directory link avoids thousands of slow shared-filesystem metadata writes.
    (root / 'source').symlink_to(old_root.resolve(), target_is_directory=True)
    for record in [*train, *validation, *selected]:
        if record['trajectory_id'] in new_demos:
            record = prepare_record(new_demos[record['trajectory_id']], record, config,
                tokenizer, root / (record['trajectory_id'] + '.pt'))
        else:
            record = dict(record, prepared_file='source/' + record['prepared_file'])
        records.append(record)
    print(json.dumps(dict(phase='balanced_test', counts=dict(Counter(
        f"{r['group']}:{r['shortest_moves']}" for r in records if r['split']=='test')))), flush=True)
    negative_jobs = []
    used_negative = set()
    for split, source_boards, count in (
        ('train', boards, options['failure_trajectories']),
        ('test_no_solution', fresh[options['validation_new_boards']:], options['test_no_solution_trajectories'])):
        WORK['excluded'] = excluded | used_negative
        accepted = 0
        with mp.get_context('fork').Pool(args.workers) as pool:
            for candidates in pool.imap(failure_candidates, source_boards, chunksize=1):
                for record, demo in candidates:
                    pairs = physical_pairs(demo)
                    if pairs & used_negative:
                        continue
                    record.update(split=split, group='failed_greedy_context', no_solution=True,
                        trajectory_id=f'{split}_failure_{accepted:06d}',
                        failure_current=str(demo.turns[-1].step.current))
                    negative_jobs.append((record, demo))
                    used_negative.update(pairs)
                    accepted += 1
                    if accepted % 100 == 0:
                        print(json.dumps(dict(phase='failure_selection', split=split, accepted=accepted)), flush=True)
                    if accepted == count:
                        break
                if accepted == count:
                    break
        if accepted != count:
            raise ValueError(f'Only {accepted}/{count} exhausted failures for {split}')
    with mp.get_context('fork').Pool(args.workers) as pool:
        for i, record in enumerate(pool.imap(prepare_failure, negative_jobs, chunksize=1)):
            records.append(record)
            if (i + 1) % 100 == 0:
                print(json.dumps(dict(phase='encode_failures', completed=i+1)), flush=True)
    pairs = [(int(r['start']), int(r['goal'])) for r in records]
    if len(set(pairs)) != len(pairs):
        raise ValueError('Duplicate physical task across splits')
    audit = dict(source_training_tasks=len(train), source_failed_training_tasks=0,
        previously_rejected=old.get('rejected_training', {}),
        added_training_failures=options['failure_trajectories'],
        negative_supervision='only final no-legal-moves, non-goal answer; all previous assistant turns masked',
        failure_endpoint='no_legal_moves_and_not_at_goal',
        negative_source='regenerated failures on training boards; rejected source records were not saved',
        no_solution_test_tasks=options['test_no_solution_trajectories'],
        training_tasks=sum(r['split']=='train' for r in records),
        training_numbered_trajectories=sum(r['variants'] for r in records if r['split']=='train'),
        test_counts=dict(Counter(f"{r['group']}:{r['shortest_moves']}" for r in records if r['split']=='test')))
    manifest = dict(config=config, records=records, q_checkpoint=old['q_checkpoint'],
        official_data=old['official_data'], source_manifest=str(args.source_manifest.resolve()),
        training_board_rows=old['training_board_rows'], validation_board_rows=old['validation_board_rows'],
        test_board_rows=old['test_board_rows'], natural_target_pairs=[], audit=audit)
    (args.out / 'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    (args.out / 'audit.json').write_text(json.dumps(audit, indent=2)+'\n')
    print(json.dumps(audit), flush=True)


if __name__ == '__main__':
    main()
