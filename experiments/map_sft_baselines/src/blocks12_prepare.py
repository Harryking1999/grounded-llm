"""Prepare one shared 50k clearing cohort for map and larger text-only SFT."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing as mp
from pathlib import Path
import time
import numpy as np
import torch
from transformers import AutoTokenizer
from grounded_llm.core.files import atomic_json
from grounded_llm.core.qmap.reader import CachedQMap
from grounded_llm.core.sft.transcript import encode_trajectory
from grounded_llm.tasks.blocks.construction import construct
from grounded_llm.tasks.blocks.clear_sft import reference_route, greedy_route, select_coverage
from grounded_llm.tasks.blocks.oracle import DistanceOracle, OracleBudgetExceeded
from grounded_llm.tasks.blocks.qmap import FrozenBoardMap
from grounded_llm.tasks.blocks.sft import demonstration


def initialize(config, checkpoint, model, excluded, out):
    global CFG, QMAP, TOKENIZER, EXCLUDED, OUT
    torch.set_num_threads(1)
    CFG, EXCLUDED, OUT = config, set(excluded), Path(out)
    QMAP = FrozenBoardMap.load(Path(checkpoint))
    TOKENIZER = AutoTokenizer.from_pretrained(model, local_files_only=True)


def candidate(root):
    try:
        oracle = DistanceOracle(cache_limit=CFG['data']['oracle_cache_limit'],
                                seconds=CFG['data']['oracle_seconds_per_board'])
        return greedy_route(int(root['initial']), CachedQMap(QMAP), oracle, EXCLUDED)
    except OracleBudgetExceeded:
        return dict(unknown=True)


def prepare_row(item):
    index, record = item
    record = dict(record, trajectory_id=f'clear12_{index:05d}', prepared_file=f'{index:05d}.pt')
    path = OUT/'trajectories'/record['prepared_file']
    packed = {}
    qmap = CachedQMap(QMAP)
    for with_map, key, order in [(False, 'plain', 'reason_first'), (True, 'map', 'ranking_first')]:
        demo = demonstration(record, qmap, with_map, order)
        encoded = encode_trajectory(demo, TOKENIZER, CFG['maximum_sequence_tokens'])
        packed[key] = dict(encoded=encoded, steps=[t.step for t in demo.turns])
        record[key+'_tokens'] = len(encoded.input_ids)
    torch.save(packed, path)
    return record


def main():
    parser = argparse.ArgumentParser(__doc__)
    for name in ('config', 'q-checkpoint', 'model-path', 'evaluation', 'out'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--workers', type=int, default=24)
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out/'trajectories').mkdir(exist_ok=True)
    if (args.out/'ready.json').exists():
        raise ValueError('Accepted dataset already exists')
    evaluation, excluded = [], set()
    oracle = DistanceOracle(cache_limit=2000000, seconds=86400)
    for case in json.loads(args.evaluation.read_text())['cases']:
        start = int(case['source'])
        actions, states = reference_route(start, oracle)
        assert len(actions) == case['true_distance']
        excluded.update(states)
        evaluation.append(dict(trajectory_id=case['id'], split='test', start=str(start),
                               goal='0', reference_actions=actions, shortest_moves=len(actions)))
    assert len(evaluation) == cfg['evaluation']['tasks']
    atomic_json(args.out/'evaluation.json', evaluation)
    rng = np.random.default_rng(cfg['seed'])
    atomic_json(args.out/'probe_indices.json', sorted(map(int, rng.choice(
        len(evaluation), cfg['evaluation']['probe_tasks'], replace=False))))
    started = time.monotonic()
    initargs = (cfg, str(args.q_checkpoint), str(args.model_path), list(excluded), str(args.out))
    selected_path = args.out/'selected.json'
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context('spawn'),
                             initializer=initialize, initargs=initargs) as pool:
        if selected_path.exists():
            selected = json.loads(selected_path.read_text())
            coverage = {tuple(p) for r in selected for p in r['pairs']}
        else:
            roots = construct(cfg['data']['candidate_boards'], 12, cfg['data']['construction_seed'])
            accepted, unknown = [], 0
            for i, row in enumerate(pool.map(candidate, roots, chunksize=16), 1):
                if row and row.get('unknown'):
                    unknown += 1
                elif row and row['shortest_moves'] >= cfg['data']['minimum_shortest_moves']:
                    accepted.append(row)
                if i % 100 == 0:
                    progress=dict(phase='selecting', examined=i, candidates=len(roots), accepted=len(accepted),
                                  unknown=unknown, elapsed=time.monotonic()-started)
                    atomic_json(args.out/'progress.json', progress)
                    print(json.dumps(progress), flush=True)
            selected, coverage = select_coverage(accepted, cfg['data']['samples_per_epoch'], cfg['seed'])
            if len(selected) != cfg['data']['samples_per_epoch'] or len(coverage) < cfg['data']['coverage_target']:
                raise ValueError(f'Insufficient accepted data: {len(selected)} trajectories, {len(coverage)} pairs')
            atomic_json(selected_path, selected)
            atomic_json(args.out/'selection_stats.json', dict(candidates=len(roots), accepted=len(accepted), unknown=unknown))
        rows = []
        for i, row in enumerate(pool.map(prepare_row, enumerate(selected), chunksize=8), 1):
            rows.append(row)
            if i % 100 == 0:
                progress=dict(phase='encoding', samples=i, total=len(selected), elapsed=time.monotonic()-started)
                atomic_json(args.out/'progress.json', progress)
                print(json.dumps(progress), flush=True)
    assert not ({int(r['start']) for r in rows} & excluded)
    assert not ({p[0] for p in coverage} & excluded)
    assert len(rows) == cfg['data']['samples_per_epoch'] and len(coverage) >= cfg['data']['coverage_target']
    atomic_json(args.out/'dataset.json', dict(config=cfg, rows=rows, unique_pairs=len(coverage),
        response_order='ranking_first', smoke=False))
    atomic_json(args.out/'acceptance.json', dict(samples=len(rows), unique_state_goal_pairs=len(coverage),
        length_histogram=dict(Counter(r['shortest_moves'] for r in rows)),
        maximum_map_tokens=max(r['map_tokens'] for r in rows),
        maximum_plain_tokens=max(r['plain_tokens'] for r in rows),
        excluded_evaluation_reference_states=len(excluded), initial_terminal_samples=0,
        evaluation='fixed_clear1000_without_Q_success_filter',
        limitation='Fixed1000 was used for Qmap selection; it is not an independent unseen Qmap acceptance set.'))
    atomic_json(args.out/'ready.json', dict(samples=len(rows), unique_pairs=len(coverage), smoke=False))


if __name__ == '__main__':
    main()
