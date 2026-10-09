"""Compare frozen Q maps on the exact existing LLM test tasks, using CPU only."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import torch

from experiments.blocks_distance_map.src.oracle import DistanceOracle
from .blocks import FrozenBoardMap
from .blocks_sft import greedy_demonstration


def summarize(rows):
    n = len(rows)
    counts = {key: sum(row[key] for row in rows)
              for key in ('reached', 'first_step_reachable', 'first_step_shortest')}
    return dict(tasks=n, **counts,
                rates={key: value/n if n else None for key, value in counts.items()})


def evaluate(manifest_path, checkpoint, config_path, out, analysis_commit):
    contract = json.loads(config_path.read_text())
    assert contract['device'] == 'cpu'
    torch.set_num_threads(contract['cpu_threads'])
    manifest = json.loads(manifest_path.read_text())
    records = [r for r in manifest['records'] if r['split'] == contract['split']]
    assert len(records) == contract['expected_tasks']
    assert len({r['trajectory_id'] for r in records}) == len(records)
    assert dict(Counter(r['group'] for r in records)) == contract['expected_group_counts']
    assert dict(Counter(str(r['shortest_moves']) for r in records)) == contract['expected_shortest_length_counts']
    paths = dict(current=Path(manifest['q_checkpoint']), continued=checkpoint)
    oracle = DistanceOracle(cache_limit=2_000_000, seconds=600)
    result = dict(analysis_commit=analysis_commit, contract=contract,
                  manifest=str(manifest_path), checkpoints={}, results={}, cases=[])
    for name, path in paths.items():
        saved = torch.load(path, map_location='cpu', weights_only=True)
        identity = dict(path=str(path), source_commit=saved['source_commit'],
                        step=saved['step'], metric=saved['metric'])
        if 'continuation_step' in saved:
            identity['continuation_step'] = saved['continuation_step']
        if name == 'continued':
            assert saved['source_commit'].startswith(contract['new_map_source_commit'])
            assert saved['continuation_step'] == contract['new_map_continuation_step']
        result['checkpoints'][name] = identity
        qmap = FrozenBoardMap.load(path, device='cpu')
        rows = []
        for index, record in enumerate(records):
            start, goal = int(record['start']), int(record['goal'])
            assert oracle.distance(start, goal) == record['shortest_moves']
            demo = greedy_demonstration(qmap, start, goal,
                rng=np.random.default_rng(record['sample_seed']),
                max_actions=manifest['config']['maximum_demonstration_actions'],
                reported=manifest['config']['data']['reported_candidates'])
            if name == 'current':
                assert demo.success == record['greedy_success'], record['trajectory_id']
            remaining = oracle.distance(demo.executed_path[1], goal)
            row = dict(trajectory_id=record['trajectory_id'], group=record['group'],
                       shortest_moves=record['shortest_moves'], reached=demo.success,
                       first_step_reachable=remaining >= 0,
                       first_step_shortest=remaining == record['shortest_moves']-1,
                       executed_steps=len(demo.executed_path)-1)
            rows.append(row)
            if name == 'current':
                result['cases'].append(dict(trajectory_id=record['trajectory_id'],
                    board_row=record['board_row'], start=record['start'], goal=record['goal'],
                    group=record['group'], shortest_moves=record['shortest_moves'], outcomes={}))
            result['cases'][index]['outcomes'][name] = dict(row,
                states=list(map(str, demo.executed_path)))
        result['results'][name] = dict(all=summarize(rows),
            by_group={g:summarize([r for r in rows if r['group']==g]) for g in contract['expected_group_counts']},
            by_shortest_length={k:summarize([r for r in rows if str(r['shortest_moves'])==k]) for k in contract['expected_shortest_length_counts']})
        print(json.dumps(dict(model=name, **result['results'][name]['all'])), flush=True)
        del qmap, saved
    result['paired'] = dict(Counter(
        f"{int(c['outcomes']['current']['reached'])}->{int(c['outcomes']['continued']['reached'])}"
        for c in result['cases']))
    result['interpretation'] = ('Same fixed510 tasks, rules, action budget and greedy policy;'
        'oracle only scores, never filters successors;current map reproduces every cached outcome.'
        'The existing LLM and its prepared memories remain on the current map.'
        'Test-task novelty relative to the continued-map additional supervision has not been audited.')
    result['observed_at_utc'] = datetime.now(timezone.utc).isoformat()
    out.mkdir(parents=True, exist_ok=False)
    (out/'results.json').write_text(json.dumps(result, indent=2)+'\n')
    compact = {k:v for k,v in result.items() if k != 'cases'}
    (out/'summary.json').write_text(json.dumps(compact, indent=2)+'\n')
    return compact


def main():
    parser = argparse.ArgumentParser()
    for name in ('manifest', 'checkpoint', 'config', 'out'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--analysis-commit', required=True)
    args = parser.parse_args()
    evaluate(args.manifest, args.checkpoint, args.config, args.out, args.analysis_commit)


if __name__ == '__main__':
    main()
