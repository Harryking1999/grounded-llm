"""GCML-style roadmaps: t-SNE of learned Q, overlaid with real graph edges.

Figure contract: show the organization of one fixed learned map at each size.
Python schematic grid; no manually moved nodes or removed graph edges.
Export independent panels and a comparison as PNG plus editable SVG/PDF,
with node coordinates, edge lists, parameters and original-space metrics.
"""
import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
import sklearn
from sklearn.manifold import TSNE, trustworthiness

from .core import latent_distances, spearman
from .report import save
from .run import ROOT, write_json


def draw(ax, data, title):
    xy, edges = data['xy'], data['edges']
    n, start, goal = len(xy), data['start'], data['goal']
    linewidth = 1.15 if n == 32 else (.65 if n == 128 else .45)
    ax.add_collection(LineCollection(xy[edges], colors='#242424',
                                     linewidths=linewidth, alpha=.88, zorder=1))
    size = 60 if n == 32 else (21 if n == 128 else 12)
    ax.scatter(xy[:, 0], xy[:, 1], s=size, facecolor='white',
               edgecolor='#999999', linewidth=.8, zorder=2)
    ax.scatter(*xy[start], s=155 if n == 32 else 105, color='black', zorder=3)
    ax.scatter(*xy[goal], s=560 if n == 32 else 390, marker='*',
               facecolor='white', edgecolor='#ed3e2e', linewidth=2.0, zorder=4)
    for node, text, dy in [(start, 'Start node', -24), (goal, 'Goal node', 22)]:
        ax.annotate(text, xy[node], xytext=(0, dy), textcoords='offset points',
                    ha='center', va='center', fontsize=10, zorder=5,
                    bbox={'facecolor': 'white', 'edgecolor': 'none', 'pad': 1.3})
    span = np.ptp(xy, axis=0)
    ax.set_xlim(xy[:, 0].min() - .13 * span.max(), xy[:, 0].max() + .13 * span.max())
    ax.set_ylim(xy[:, 1].min() - .18 * span.max(), xy[:, 1].max() + .18 * span.max())
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(title, loc='left', fontsize=13, pad=8)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-root', type=Path, required=True,
                        help='Directory containing case folders with inputs.npz and map.npz')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=ROOT / 'experiments/cml_map_scaling/configs/roadmap_visualization.json')
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding='utf-8'))
    args.output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans'],
                         'svg.fonttype': 'none', 'pdf.fonttype': 42, 'font.size': 10})
    panels, metrics = [], []
    for case in config['cases']:
        with np.load(args.input_root / case / 'map.npz') as learned:
            q = learned['q'].astype(float)
        with np.load(args.input_root / case / 'inputs.npz') as inputs:
            adjacency, distances = inputs['adjacency'], inputs['graph_distances']
        edges = np.column_stack(np.nonzero(np.triu(adjacency, 1)))
        start, goal = np.argwhere(np.triu(distances == distances.max(), 1))[0]
        model = TSNE(**config['tsne'])
        raw_xy = model.fit_transform(q)
        delta = raw_xy[goal] - raw_xy[start]
        angle = np.deg2rad(65) - np.arctan2(delta[1], delta[0])
        rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        xy = (raw_xy - raw_xy.mean(axis=0)) @ rotation.T
        xy /= np.ptp(xy, axis=0).max()
        panel = {'case': case, 'xy': xy, 'edges': edges, 'start': int(start), 'goal': int(goal)}
        panels.append(panel)
        i, j = np.triu_indices(len(q), 1)
        metrics.append({'case': case, 'nodes': len(q), 'edges': len(edges),
                        'start': int(start), 'goal': int(goal), 'endpoint_graph_distance': int(distances[start, goal]),
                        'high_dim_spearman': spearman(distances[i, j], latent_distances(q)[i, j]),
                        'projection_spearman': spearman(distances[i, j], latent_distances(xy)[i, j]),
                        'trustworthiness_k5': float(trustworthiness(q, raw_xy, n_neighbors=5)),
                        'tsne_kl_divergence': float(model.kl_divergence_)})
        with (args.output / f'roadmap_{case}_nodes.csv').open('w', newline='', encoding='utf-8') as stream:
            writer = csv.writer(stream)
            writer.writerow(['node', 'tsne_x', 'tsne_y', 'display_x', 'display_y', 'role'])
            for node, point in enumerate(xy):
                writer.writerow([node, *raw_xy[node], *point, 'start' if node == start else 'goal' if node == goal else 'node'])
        np.savetxt(args.output / f'roadmap_{case}_edges.csv', edges, fmt='%d', delimiter=',',
                   header='source,target', comments='')
        fig, ax = plt.subplots(figsize=(5.1, 5.1))
        draw(ax, panel, f'{len(q)} nodes')
        fig.subplots_adjust(left=.05, right=.95, bottom=.04, top=.92)
        save(fig, args.output / f'roadmap_{case}')
        plt.close(fig)
    fig, axes = plt.subplots(1, len(panels), figsize=(14.4, 5.1))
    for index, (ax, panel) in enumerate(zip(axes, panels)):
        draw(ax, panel, f'{"abc"[index]}   {len(panel["xy"])} nodes')
    fig.subplots_adjust(left=.025, right=.975, bottom=.03, top=.91, wspace=.12)
    save(fig, args.output / 'roadmap_comparison')
    plt.close(fig)
    write_json(args.output / 'roadmap_summary.json', {'config': config,
               'sklearn': sklearn.__version__, 'numpy': np.__version__, 'cases': metrics})
    print(json.dumps(metrics, indent=2))


if __name__ == '__main__':
    main()
