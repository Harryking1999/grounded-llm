"""Export compact per-map evidence and grouped tables for Step 1 exploration."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .run import write_json


def distance_figures(raw, output):
    """Show learned near/far structure in the original latent space.

    Quantitative grid: six graph settings, matched before-learning control.
    Export report-size PNG and editable SVG/PDF, with per-map source data.
    Normalize each map and stage by its own mean one-edge distance; this
    compares relation shape independently of dimension and overall scale.
    """
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    from .report import save

    plt.rcParams.update({'font.family': 'sans-serif',
                         'font.sans-serif': ['Microsoft YaHei', 'DejaVu Sans'],
                         'font.size': 10, 'axes.spines.top': False,
                         'axes.spines.right': False, 'svg.fonttype': 'none',
                         'pdf.fonttype': 42, 'legend.frameon': False})
    styles = {
        'local128': ('局部更新 · 128 维', '#7c9db5', ':'),
        'local1000': ('局部更新 · 1000 维', '#265e83', '-'),
        'local2048': ('局部更新 · 2048 维', '#329592', '--'),
        'gradient1000': ('完整梯度 · 1000 维', '#b87543', '-.'),
    }
    bucket_rows = []
    for case in raw['cases']:
        for snapshot in [case['snapshots'][0], case['snapshots'][-1]]:
            buckets = snapshot['geometry']['distance_buckets']
            anchor = next(b['latent_mean'] for b in buckets if b['graph_distance'] == 1)
            for b in buckets:
                bucket_rows.append({**{k: case[k] for k in ['name', 'case', 'nodes', 'seed', 'official_graph']},
                                    'epoch': snapshot['epoch'], **b,
                                    'normalized_mean': b['latent_mean'] / anchor,
                                    'normalized_within_bucket_sd': b['latent_sd'] / anchor})
    with (output / 'exploration_distance_buckets.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(bucket_rows[0]))
        writer.writeheader()
        writer.writerows(bucket_rows)

    def curve(name, official, n, initial=False):
        cases = [c for c in raw['cases'] if c['name'] == name and c['official_graph'] == official and c['nodes'] == n]
        maps = []
        for c in cases:
            b = c['snapshots'][0 if initial else -1]['geometry']['distance_buckets']
            anchor = next(x['latent_mean'] for x in b if x['graph_distance'] == 1)
            maps.append({x['graph_distance']: x['latent_mean'] / anchor for x in b})
        # Keep the number of contributing maps fixed at every plotted distance.
        distances = sorted(set.intersection(*(set(m) for m in maps)))
        values = np.array([[m[d] for d in distances] for m in maps])
        return distances, values.mean(axis=0)

    fig, axes = plt.subplots(2, 3, figsize=(11.0, 6.8), sharey=True)
    settings = [(True, 32)] + [(False, n) for n in [32, 64, 128, 256, 512]]
    ymax = max(max(curve(name, official, n)[1])
               for name in styles for official, n in settings)
    for index, (ax, (official, n)) in enumerate(zip(axes.flat, settings)):
        for name, (label, color, line) in styles.items():
            x, y = curve(name, official, n)
            ax.plot(x, y, label=label, color=color, linestyle=line, linewidth=2.1)
        x, y = curve('local1000', official, n, initial=True)
        ax.plot(x, y, color='#888888', linestyle=(0, (4, 3)), linewidth=1.4,
                label='学习前 · 1000 维')
        ax.set_title(('原始图 · 32 节点' if official else f'随机图 · {n} 节点'), loc='left', fontsize=11)
        ax.text(-.13, 1.04, 'abcdef'[index], transform=ax.transAxes, weight='bold', fontsize=12)
        ax.set_xlabel('图最短距离（步数）')
        ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))
        ax.set_ylim(.8, np.ceil(ymax * 2) / 2 + .1)
        ax.grid(axis='y', color='#eeeeee', linewidth=.7)
        if index % 3 == 0:
            ax.set_ylabel('表示距离 / 一步邻居平均距离')
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3, bbox_to_anchor=(.5, .015), fontsize=10)
    fig.subplots_adjust(left=.09, right=.98, bottom=.19, top=.94, hspace=.43, wspace=.17)
    save(fig, output / 'exploration_distance_structure')
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.3, 3.8))
    sizes = [32, 64, 128, 256, 512]
    for name, (label, color, line) in styles.items():
        means, errors = [], []
        for n in sizes:
            values = [c['snapshots'][-1]['geometry']['spearman'] for c in raw['cases']
                      if c['name'] == name and c['nodes'] == n and not c['official_graph']]
            means.append(np.mean(values))
            errors.append(np.std(values, ddof=1))
        ax.errorbar(sizes, means, yerr=errors, label=label, color=color, linestyle=line,
                    linewidth=1.7, marker='o', markersize=3.5, capsize=3)
    initial = [np.mean([c['snapshots'][0]['geometry']['spearman'] for c in raw['cases']
                       if c['name'] == 'local1000' and c['nodes'] == n and not c['official_graph']]) for n in sizes]
    ax.plot(sizes, initial, '--', color='#888888', label='学习前 · 1000 维')
    ax.set_xscale('log', base=2)
    ax.set_xticks(sizes, labels=[str(n) for n in sizes])
    ax.set_xlabel('图节点数')
    ax.set_ylabel('图距离与表示距离的秩相关')
    ax.set_ylim(-.08, 1.0)
    ax.grid(axis='y', color='#eeeeee', linewidth=.7)
    ax.legend(loc='lower left', bbox_to_anchor=(1.01, .22), fontsize=9)
    fig.subplots_adjust(left=.11, right=.67, bottom=.17, top=.95)
    save(fig, output / 'exploration_distance_correlation')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--figures', action='store_true', help='Also export distance curves with Matplotlib')
    args = parser.parse_args()
    raw = json.loads((args.run_dir / 'summary.json').read_text())
    if not raw['complete']:
        raise ValueError('Exploration is not complete')
    rows = []
    for case in raw['cases']:
        first, last = case['snapshots'][0], case['snapshots'][-1]
        g = last['geometry']
        rows.append({**{k: case[k] for k in ['name', 'case', 'nodes', 'seed',
                    'official_graph', 'state_dim', 'method', 'action_coverage', 'training_seconds']},
                     'transition_mse': g['transition_mse_all_actions'],
                     'successor_accuracy': g['successor_retrieval_accuracy'],
                     'state_centered_rms': g['state_centered_rms'],
                     'state_rms_ratio': g['state_centered_rms'] / first['geometry']['state_centered_rms'],
                     'minimum_pair_distance': g['latent_pair_min'],
                     'spearman_initial': first['geometry']['spearman'],
                     'spearman': g['spearman'], 'cosine_reach': last['planning']['reach_rate']})
    groups = []
    for name in dict.fromkeys(r['name'] for r in rows):
        for official, n in [(True, 32)] + [(False, n) for n in [32, 64, 128, 256, 512]]:
            selected = [r for r in rows if r['name'] == name and r['official_graph'] == official and r['nodes'] == n]
            group = {'name': name, 'official_graph': official, 'nodes': n, 'repeats': len(selected)}
            for field in ['transition_mse', 'successor_accuracy', 'state_centered_rms',
                          'state_rms_ratio', 'minimum_pair_distance', 'spearman', 'cosine_reach']:
                values = [r[field] for r in selected]
                group[field] = float(np.mean(values))
                group[field + '_sd'] = float(np.std(values, ddof=1)) if len(values) > 1 else None
            groups.append(group)
    args.output.mkdir(parents=True, exist_ok=True)
    write_json(args.output / 'exploration_summary.json', {
        **{k: raw[k] for k in ['complete', 'source_revision', 'python', 'numpy', 'device']},
        'cases': rows, 'groups': groups})
    with (args.output / 'exploration_data.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    if args.figures:
        distance_figures(raw, args.output)
    print(json.dumps(groups, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
