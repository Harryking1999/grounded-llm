"""Locate projected long edges and test whether more training changes geometry."""
import argparse
import csv
import json
from pathlib import Path
import subprocess
import time

import numpy as np

from .core import geometry, latent_distances, train_epoch
from .run import ROOT, write_json


def run(config, input_root, output):
    base = json.loads((ROOT / config['base_config']).read_text())
    train = base['training']
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / 'config.json', {'diagnostic': config, 'base': base})
    summary = {'complete': False, 'source_revision': subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(), 'numpy': np.__version__, 'device': 'CPU', 'cases': []}
    for name in config['cases']:
        with np.load(input_root / name / 'map.npz') as learned:
            q, v = learned['q'].copy(), learned['v'].copy()
        with np.load(input_root / name / 'inputs.npz') as inputs:
            actions, walks, distances = inputs['actions'], inputs['walks'], inputs['graph_distances']
            visits, adj = inputs['visits'], inputs['adjacency']
        rows = list(csv.DictReader((ROOT / config['reference_coordinates'] / f'roadmap_{name}_nodes.csv').open()))
        xy = np.array([[float(r['display_x']), float(r['display_y'])] for r in rows])
        edges = np.column_stack(np.nonzero(np.triu(adj, 1)))
        ld = np.linalg.norm(xy[edges[:, 0]] - xy[edges[:, 1]], axis=1)
        selected = np.argsort(-ld, kind='stable')[:config['selected_edges']]
        pi, pj = np.triu_indices(len(q), 1)
        initial_pairs = latent_distances(q)[pi, pj]
        initial_normalized = initial_pairs / initial_pairs.mean()
        # Restore the generator state after the original 20 shuffled epochs.
        replay = np.random.default_rng(int(name.rsplit('seed', 1)[1]) + train['seed_offsets']['replay'])
        for _ in range(config['resume_epoch']):
            replay.permutation(len(walks))
        snapshots, elapsed = [], 0.0
        for epoch in range(config['resume_epoch'], max(config['snapshot_epochs']) + 1):
            if epoch > config['resume_epoch']:
                before = time.perf_counter()
                train_epoch(q, v, walks, replay.permutation(len(walks)), train['eta_q'], train['eta_v'])
                elapsed += time.perf_counter() - before
            if epoch in config['snapshot_epochs']:
                metrics, (_, _, _, pairs) = geometry(q, v, actions, distances, visits)
                hd = np.linalg.norm(q[edges[:, 0]].astype(float) - q[edges[:, 1]].astype(float), axis=1)
                selection = []
                for rank, idx in enumerate(selected):
                    source, target = edges[idx]
                    selection.append({'label': 'ABCDE'[rank], 'source': int(source), 'target': int(target),
                                      'graph_distance': int(distances[source, target]),
                                      'high_dim_length': float(hd[idx]), 'high_dim_edge_mean_ratio': float(hd[idx] / hd.mean()),
                                      'high_dim_edge_percentile': float(np.mean(hd <= hd[idx]) * 100),
                                      'high_dim_all_pair_percentile': float(np.mean(pairs <= hd[idx]) * 100),
                                      'epoch20_2d_edge_mean_ratio': float(ld[idx] / ld.mean())})
                snapshot = {'epoch': epoch, 'spearman': metrics['spearman'],
                            'transition_mse': metrics['transition_mse_all_actions'],
                            'successor_accuracy': metrics['successor_retrieval_accuracy'],
                            'edge_mean': float(hd.mean()), 'edge_cv': float(hd.std() / hd.mean()),
                            'normalized_pair_distance_drift': float(np.sqrt(np.mean((pairs / pairs.mean() - initial_normalized)**2))),
                            'selected_edges': selection}
                snapshots.append(snapshot)
                np.savez_compressed(output / f'{name}_epoch{epoch}.npz', q=q, v=v)
                print(json.dumps({'case': name, 'epoch': epoch, 'spearman': metrics['spearman'],
                                  'mse': snapshot['transition_mse'], 'drift': snapshot['normalized_pair_distance_drift']}), flush=True)
        summary['cases'].append({'case': name, 'nodes': len(q), 'edges': len(edges), 'extra_training_seconds': elapsed,
                                 'projected_edge_cv': float(ld.std() / ld.mean()), 'snapshots': snapshots})
        write_json(output / 'summary.json', summary)
    summary['complete'] = True
    write_json(output / 'summary.json', summary)


def export(config, input_root, run_dir, output):
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from .report import save

    summary = json.loads((run_dir / 'summary.json').read_text())
    if not summary['complete']:
        raise ValueError('Incomplete diagnostic')
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / 'long_edge_summary.json', summary)
    records = [{'case': c['case'], 'epoch': s['epoch'], **e} for c in summary['cases']
               for s in c['snapshots'] for e in s['selected_edges']]
    with (output / 'long_edge_data.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0])); writer.writeheader(); writer.writerows(records)
    plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Microsoft YaHei', 'DejaVu Sans'],
                         'svg.fonttype': 'none', 'pdf.fonttype': 42, 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 9.0))
    colors = ['#bb5434', '#277e8e', '#79598e', '#b99230', '#395f95']
    for row, case in enumerate(summary['cases']):
        name = case['case']
        data = list(csv.DictReader((ROOT / config['reference_coordinates'] / f'roadmap_{name}_nodes.csv').open()))
        xy = np.array([[float(r['display_x']), float(r['display_y'])] for r in data])
        with np.load(input_root / name / 'inputs.npz') as inputs:
            edges = np.column_stack(np.nonzero(np.triu(inputs['adjacency'], 1)))
        with np.load(input_root / name / 'map.npz') as learned:
            q = learned['q'].astype(float)
        hd = np.linalg.norm(q[edges[:, 0]] - q[edges[:, 1]], axis=1)
        ld = np.linalg.norm(xy[edges[:, 0]] - xy[edges[:, 1]], axis=1)
        left, right = axes[row]
        left.add_collection(LineCollection(xy[edges], colors='#b6b6b6', linewidths=.5))
        left.scatter(xy[:, 0], xy[:, 1], s=11, facecolor='white', edgecolor='#aaaaaa', linewidth=.5, zorder=2)
        right.scatter(hd / hd.mean(), ld / ld.mean(), color='#b5bec5', s=13, alpha=.7)
        for color, edge in zip(colors, case['snapshots'][0]['selected_edges']):
            u, v = edge['source'], edge['target']
            left.plot(*xy[[u, v]].T, color=color, linewidth=1.5, zorder=3)
            midpoint = (xy[u] + xy[v]) / 2
            left.text(*midpoint, edge['label'], color=color, weight='bold', ha='center', va='center',
                      bbox={'facecolor': 'white', 'edgecolor': 'none', 'pad': 1}, zorder=4)
            x, y = edge['high_dim_edge_mean_ratio'], edge['epoch20_2d_edge_mean_ratio']
            right.scatter(x, y, color=color, s=33, zorder=3)
            right.annotate(edge['label'], (x, y), xytext=(5, 5), textcoords='offset points', color=color, weight='bold')
        left.set_aspect('equal'); left.axis('off'); left.margins(.08)
        left.set_title(f'{case["nodes"]} 节点：二维图中最长的 5 条边', loc='left')
        right.axvline(1, color='#dddddd', linestyle='--', linewidth=1)
        right.axhline(1, color='#dddddd', linestyle='--', linewidth=1)
        right.set_xlabel('1000 维边长 / 高维平均单步边长')
        right.set_ylabel('二维边长 / 二维平均单步边长')
        right.set_title('同一条边在高维与二维中的相对长度', loc='left')
        right.set_ylim(0, 8.5)
        right.set_xlim(.55, 1.3)
    fig.subplots_adjust(left=.035, right=.97, bottom=.075, top=.95, wspace=.24, hspace=.27)
    save(fig, output / 'long_edge_projection')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-root', type=Path, required=True)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--config', type=Path, default=ROOT / 'experiments/cml_map_scaling/configs/long_edge_diagnostic.json')
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if args.output:
        export(config, args.input_root, args.run_dir, args.output)
    else:
        run(config, args.input_root, args.run_dir)


if __name__ == '__main__':
    main()
