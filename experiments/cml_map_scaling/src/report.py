"""Produce compact source data and diagnostic figures from saved runs only."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    summary = json.loads((args.run_dir / 'summary.json').read_text(encoding='utf-8'))
    if not summary['complete']:
        raise ValueError(f"Incomplete run: {summary['missing']}")
    rows = []
    for case in summary['cases']:
        initial, final = case['snapshots'][0], case['snapshots'][-1]
        rows.append({
            **{k: case[k] for k in ['case', 'nodes', 'seed', 'official_graph', 'directed_actions',
                                    'degree_min', 'degree_max', 'action_coverage', 'minimum_action_visits',
                                    'dataset_transitions', 'processed_transitions', 'training_seconds', 'total_seconds']},
            'spearman_initial': initial['geometry']['spearman'],
            'spearman_final': final['geometry']['spearman'],
            'transition_mse_initial': initial['geometry']['transition_mse_all_actions'],
            'transition_mse_final': final['geometry']['transition_mse_all_actions'],
            'reach_initial': initial['planning']['reach_rate'],
            'reach_final': final['planning']['reach_rate'],
            'shortest_final': final['planning']['shortest_rate_all_pairs'],
            'first_action_progress_final': final['planning']['first_action_shortens_distance_rate'],
            'stretch_success_only': final['planning']['mean_stretch_success_only'],
            'v_norm_cv_final': final['geometry']['v_norm_cv'],
            'sqrt_fit_r2_final': final['geometry']['sqrt_fit_r2'],
            'initial_buckets': initial['geometry']['distance_buckets'],
            'final_buckets': final['geometry']['distance_buckets'],
        })
    args.output.mkdir(parents=True, exist_ok=True)
    runtime = json.loads((args.run_dir / 'runtime_scaling.json').read_text(encoding='utf-8'))
    compact = {'complete': True, 'source_revision': runtime['source_revision'],
               'python': runtime['python'], 'numpy': runtime['numpy'], 'device': runtime['execution_device'],
               'cases': rows, 'scaling': []}
    fields = ['spearman_initial', 'spearman_final', 'reach_initial', 'reach_final',
              'shortest_final', 'first_action_progress_final', 'transition_mse_final']
    for n in sorted({x['nodes'] for x in rows if not x['official_graph']}):
        selected = [x for x in rows if x['nodes'] == n and not x['official_graph']]
        aggregate = {'nodes': n, 'independent_maps': len(selected)}
        for field in fields:
            values = np.array([x[field] for x in selected])
            aggregate[field] = {'mean': float(values.mean()), 'sd_across_maps': float(values.std(ddof=1)),
                                'min': float(values.min()), 'max': float(values.max())}
        compact['scaling'].append(aggregate)
    (args.output / 'summary.json').write_text(json.dumps(compact, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    with (args.output / 'source_data.csv').open('w', newline='', encoding='utf-8') as stream:
        names = [k for k in rows[0] if not k.endswith('_buckets')]
        writer = csv.DictWriter(stream, names, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    with (args.output / 'distance_buckets.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(['case', 'phase', 'graph_distance', 'pairs', 'latent_mean', 'latent_sd'])
        for row in rows:
            for phase in ['initial', 'final']:
                for bucket in row[f'{phase}_buckets']:
                    writer.writerow([row['case'], phase, bucket['graph_distance'], bucket['pairs'],
                                     bucket['latent_mean'], bucket['latent_sd']])
    plot(summary, compact, args.output)


def plot(raw, compact, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'svg.fonttype': 'none', 'pdf.fonttype': 42,
                         'legend.frameon': False})
    colors = {'initial': '#89939E', 'learned': '#267889'}
    official = next(x for x in raw['cases'] if x['official_graph'])
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.4), layout='constrained')
    for snapshot, label, color in [(official['snapshots'][0], 'Before learning', colors['initial']),
                                    (official['snapshots'][-1], 'After learning', colors['learned'])]:
        buckets = snapshot['geometry']['distance_buckets']
        x = np.array([b['graph_distance'] for b in buckets])
        y = np.array([b['latent_mean'] for b in buckets])
        sd = np.array([b['latent_sd'] for b in buckets])
        axes[0].errorbar(x, y / y[0], yerr=sd / y[0], label=label, color=color, marker='o', capsize=2)
    axes[0].plot(x, np.sqrt(x), '--', color='#A87046', linewidth=1, label='Square-root reference')
    axes[0].set(xlabel='Shortest-path distance (edges)', ylabel='Latent distance / mean 1-hop distance',
                title='a  Original graph: distance structure')
    axes[0].legend(fontsize=8)
    snapshots = official['snapshots']
    axes[1].plot([s['epoch'] for s in snapshots], [s['geometry']['spearman'] for s in snapshots],
                 '-o', color=colors['learned'])
    axes[1].set(xlabel='Replay epoch', ylabel='Spearman correlation', ylim=(-0.1, 1),
                title='b  Geometry across learning')
    fig.suptitle('Independent Q/V learning on the fixed 32-node graph', fontsize=11)
    save(fig, output / 'geometry')
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.4), layout='constrained')
    scale = compact['scaling']
    ns = [r['nodes'] for r in scale]
    for ax, metric, ylabel, title in [(axes[0], 'spearman', 'Spearman correlation', 'a  Distance structure'),
                                      (axes[1], 'reach', 'Fraction reaching goal', 'b  Cosine action selector')]:
        for phase, label, color in [('initial', 'Before learning', colors['initial']),
                                    ('final', 'After learning', colors['learned'])]:
            key = f'{metric}_{phase}'
            mean = [r[key]['mean'] for r in scale]
            sd = [r[key]['sd_across_maps'] for r in scale]
            ax.errorbar(ns, mean, yerr=sd, color=color, label=label, marker='o', capsize=3)
        ax.set(xlabel='Nodes per independently learned map', ylabel=ylabel, title=title,
               xscale='log', ylim=(-.12 if metric == 'spearman' else 0, 1.03))
        ax.set_xticks(ns, [str(n) for n in ns])
        ax.minorticks_off()
        ax.legend(fontsize=8)
        if metric == 'reach':
            ax.annotate(f"{scale[-1]['reach_final']['mean']:.1%}",
                        (ns[-1], scale[-1]['reach_final']['mean']), xytext=(-5, -18),
                        textcoords='offset points', ha='right', color=colors['learned'])
    fig.suptitle('Fixed 1000D state space; three independent maps per size', fontsize=11)
    save(fig, output / 'scaling')
    plt.close(fig)


def save(fig, stem):
    for ext in ['png', 'svg', 'pdf']:
        fig.savefig(str(stem) + '.' + ext, dpi=180)
        if ext == 'svg':
            path = Path(str(stem) + '.svg')
            text = path.read_text(encoding='utf-8')
            path.write_text('\n'.join(line.rstrip() for line in text.splitlines()) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
