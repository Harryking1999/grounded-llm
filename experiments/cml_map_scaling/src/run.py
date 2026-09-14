"""Run the bounded Step 1 contract. Each graph learns its own independent Q/V."""
import argparse
import csv
import json
import os
from pathlib import Path
import platform
import subprocess
import time

import numpy as np

from .core import (action_catalog, cosine_policy, evaluate_policy, geometry,
                   random_graph, sample_walks, shortest_distances, train_epoch)

ROOT = Path(__file__).resolve().parents[3]


def write_json(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def cases(config):
    result = []
    if config['graphs']['include_official32']:
        result.append(('official32', 32, 0, True))
    result.extend((f'random{n}_seed{seed}', n, seed, False)
                  for n in config['graphs']['sizes'] for seed in config['graphs']['seeds'])
    return result


def run_case(config, spec, output):
    name, n, seed, official = spec
    case_dir = output / name
    case_dir.mkdir()  # A rerun must get a new output directory.
    if official:
        original = json.loads((ROOT / config['sources']['graph32_config']).read_text(encoding='utf-8'))
        adj = np.zeros((n, n), dtype=bool)
        for node, neighbors in original['neighbors'].items():
            adj[int(node), neighbors] = True
    else:
        adj = random_graph(n, seed, config['graphs']['min_edges_argument'], config['graphs']['max_edges_argument'])
    distances = shortest_distances(adj)
    actions, outgoing = action_catalog(adj)
    train = config['training']
    model = config['model']
    offsets = train['seed_offsets']
    count = train['walks_at_32_nodes'] * n // 32 if train['walk_count_scales_with_nodes'] else train['walks_at_32_nodes']
    walks = sample_walks(actions, outgoing, count, train['transitions_per_walk'], seed + offsets['walks'])
    visits = np.bincount(walks[:, :, 1].ravel(), minlength=len(actions))
    init = np.random.default_rng(seed + offsets['initialization'])
    q = init.normal(0, model['q_init_std'], (n, model['state_dim'])).astype(model['dtype'])
    v = init.normal(0, model['v_init_std'], (len(actions), model['state_dim'])).astype(model['dtype'])
    np.savez_compressed(case_dir / 'inputs.npz', adjacency=adj, actions=actions,
                        walks=walks, graph_distances=distances, visits=visits, q_initial=q, v_initial=v)
    replay = np.random.default_rng(seed + offsets['replay'])
    snapshots = []
    train_seconds = 0.0
    started = time.perf_counter()
    for epoch in range(train['epochs'] + 1):
        if epoch:
            order = replay.permutation(count)
            before = time.perf_counter()
            train_epoch(q, v, walks, order, train['eta_q'], train['eta_v'], train.get('method', 'local'))
            train_seconds += time.perf_counter() - before
        if epoch in train['snapshot_epochs']:
            metrics, pair_data = geometry(q, v, actions, distances, visits)
            if not np.isfinite(q).all() or not np.isfinite(v).all():
                raise FloatingPointError('Non-finite learned map')
            snapshot = {'epoch': epoch, 'training_seconds': train_seconds, 'geometry': metrics}
            if epoch in (0, train['epochs']):
                snapshot['planning'], plans = evaluate_policy(cosine_policy(q, v, actions, outgoing), distances)
                np.savez_compressed(case_dir / f'planning_epoch{epoch}.npz', **plans)
                pi, pj, dg, ds = pair_data
                with (case_dir / f'distances_epoch{epoch}.csv').open('w', newline='', encoding='utf-8') as stream:
                    writer = csv.writer(stream)
                    writer.writerow(['node_i', 'node_j', 'graph_distance', 'latent_distance'])
                    writer.writerows(zip(pi, pj, dg, ds))
            snapshots.append(snapshot)
            with (case_dir / 'progress.jsonl').open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(snapshot, allow_nan=False) + '\n')
            print(json.dumps({'case': name, 'epoch': epoch, 'spearman': metrics['spearman'],
                              'transition_mse': metrics['transition_mse_all_actions'],
                              'train_seconds': round(train_seconds, 2)}), flush=True)
    np.savez_compressed(case_dir / 'map.npz', q=q, v=v)
    degree = adj.sum(axis=1)
    result = {
        'case': name, 'nodes': n, 'seed': seed, 'official_graph': official,
        'directed_actions': len(actions), 'degree_min': int(degree.min()),
        'degree_max': int(degree.max()), 'degree_mean': float(degree.mean()),
        'diameter': int(distances.max()), 'walks': count,
        'dataset_transitions': int(walks.shape[0] * walks.shape[1]),
        'processed_transitions': int(walks.shape[0] * walks.shape[1] * train['epochs']),
        'visited_nodes': int(len(np.unique(walks[:, :, [0, 2]]))),
        'visited_actions': int(np.count_nonzero(visits)),
        'action_coverage': float(np.mean(visits > 0)),
        'minimum_action_visits': int(visits.min()),
        'training_seconds': train_seconds, 'total_seconds': time.perf_counter() - started,
        'snapshots': snapshots,
    }
    write_json(case_dir / 'summary.json', result)
    return result


def summarize(config, output):
    results, missing = [], []
    for name, *_ in cases(config):
        path = output / name / 'summary.json'
        if path.exists():
            results.append(json.loads(path.read_text(encoding='utf-8')))
        else:
            missing.append(name)
    summary = {'complete': not missing, 'missing': missing, 'cases': results}
    write_json(output / 'summary.json', summary)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, default=ROOT / 'experiments/cml_map_scaling/configs/step1.json')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--group', choices=['official', 'scaling', 'all'], default='all')
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding='utf-8'))
    args.output.mkdir(parents=True, exist_ok=True)
    saved = args.output / 'config.json'
    if saved.exists() and json.loads(saved.read_text(encoding='utf-8')) != config:
        raise ValueError('Output directory belongs to a different configuration')
    write_json(saved, config)
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    write_json(args.output / f'runtime_{args.group}.json', {
        'source_revision': revision, 'python': platform.python_version(),
        'numpy': np.__version__, 'platform': platform.platform(),
        'execution_device': 'CPU', 'group': args.group,
        'omp_threads': os.environ.get('OMP_NUM_THREADS'),
        'openblas_threads': os.environ.get('OPENBLAS_NUM_THREADS'),
    })
    for spec in cases(config):
        if args.group == 'official' and not spec[3] or args.group == 'scaling' and spec[3]:
            continue
        run_case(config, spec, args.output)
        summarize(config, args.output)


if __name__ == '__main__':
    main()
