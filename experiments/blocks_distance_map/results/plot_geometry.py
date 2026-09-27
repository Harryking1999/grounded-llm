"""Project frozen Q vectors, then render projections and a board-level path example.

Initial analysis: --export-dir RUN_DIRECTORY; later renders use only the saved JSON.
Requires numpy, scipy, scikit-learn and matplotlib; never trains or updates Q.
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import ConnectionPatch, Rectangle
import numpy as np

HERE = Path(__file__).resolve().parent
BLUE, GREEN, RED, GRAY = '#27617D', '#278167', '#BC625F', '#B8C2C7'
FONT = Path('C:/Windows/Fonts/msyh.ttc')
if FONT.exists():
    font_manager.fontManager.addfont(str(FONT))
    plt.rcParams['font.family'] = font_manager.FontProperties(fname=str(FONT)).get_name()
plt.rcParams.update({'axes.unicode_minus': False, 'font.size': 11,
                     'svg.fonttype': 'none', 'pdf.fonttype': 42, 'savefig.dpi': 200})


def project(export_dir):
    from scipy.spatial.distance import cdist
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    from threadpoolctl import threadpool_limits
    import sklearn
    data = json.loads((export_dir / 'graph.json').read_text(encoding='utf-8'))
    q = np.load(export_dir / 'q_vectors.npz')['q'].astype(np.float64)
    assert q.shape == (len(data['nodes']), data['dimension'])
    assert data['metric'] == 'directed_sum'
    # For this trained readout, mean(D(s,t), D(t,s)) equals scaled L1 distance.
    symmetric = cdist(q, q, metric='cityblock') / (2 * q.shape[1] ** .5)
    pca_model = PCA(n_components=2, svd_solver='full')
    with threadpool_limits(limits=4):
        pca = pca_model.fit_transform(q)
        tsne_model = TSNE(n_components=2, metric='precomputed', method='exact',
                          init='random', perplexity=30, learning_rate='auto',
                          max_iter=1500, random_state=20260927)
        tsne = tsne_model.fit_transform(symmetric)
    # Fix arbitrary PCA signs for readability without modifying any distances.
    if pca[data['start'], 0] < pca[data['goal'], 0]:
        pca[:, 0] *= -1

    def neighbor_retention(xy):
        original = symmetric.copy()
        reduced = cdist(xy, xy)
        np.fill_diagonal(original, np.inf)
        np.fill_diagonal(reduced, np.inf)
        a, b = np.argsort(original, axis=1)[:, :10], np.argsort(reduced, axis=1)[:, :10]
        return float(np.mean([len(set(x).intersection(y)) / 10 for x, y in zip(a, b)]))

    for i, node in enumerate(data['nodes']):
        node['pca'] = pca[i].round(7).tolist()
        node['tsne'] = tsne[i].round(7).tolist()
    data['first_step_comparison'] = []
    for key in ['greedy', 'forced_bad_first_then_greedy']:
        source, target = data[key]['nodes'][:2]
        data['first_step_comparison'].append({
            'branch': key, 'target': target, 'action': data[key]['actions'][0],
            'q_forward': float(np.maximum(q[source]-q[target], 0).sum() / q.shape[1] ** .5),
            'q_reverse': float(np.maximum(q[target]-q[source], 0).sum() / q.shape[1] ** .5),
            'symmetric_q_distance': float(symmetric[source, target])})
    data['projection'] = {
        'sklearn_version': sklearn.__version__,
        'pca_input': 'raw 128D Q, centered, no feature scaling',
        'pca_variance_ratio': pca_model.explained_variance_ratio_.tolist(),
        'tsne_input': '(D_Q(s,t)+D_Q(t,s))/2; not exact puzzle distances',
        'tsne_perplexity': 30, 'tsne_seed': 20260927, 'tsne_max_iter': 1500,
        'tsne_init': 'random', 'tsne_method': 'exact', 'tsne_learning_rate': 'auto',
        'tsne_final_kl': float(tsne_model.kl_divergence_),
        'knn10_retention_against_symmetric_q': {
            'pca': neighbor_retention(pca), 'tsne': neighbor_retention(tsne)},
        'labels_used_to_fit_projection': False,
    }
    return data


def labels(data):
    names = {node: str(i) for i, node in enumerate(data['greedy']['nodes'])}
    names[data['start']], names[data['goal']] = 'S', 'G'
    for i, node in enumerate(data['forced_bad_first_then_greedy']['nodes'][1:], 1):
        names[node] = f'X{i}'
    return names


def save(fig, out, name):
    for extension in ['png', 'svg']:
        path = out / f'{name}.{extension}'
        fig.savefig(path, bbox_inches='tight', facecolor='white')
        if extension == 'svg':
            # Matplotlib leaves spaces at path line ends; retain the newline separator.
            clean = '\n'.join(line.rstrip() for line in path.read_text(encoding='utf-8').splitlines()) + '\n'
            path.write_bytes(clean.encode('utf-8'))
    plt.close(fig)


def draw_projections(data, out):
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.8), layout='constrained')
    names = labels(data)
    edges = np.array(data['edges'])
    good = np.array([node['exact_remaining'] >= 0 for node in data['nodes']])
    for ax, kind, title in zip(axes, ['pca', 'tsne'],
                               ['PCA · Q 坐标投影', 't-SNE · 对称化 Q 距离']):
        xy = np.array([node[kind] for node in data['nodes']])
        ax.add_collection(LineCollection(xy[edges[:, :2]], colors=GRAY, linewidths=.35, alpha=.27))
        ax.scatter(*xy[~good].T, marker='x', s=10, color=RED, linewidths=.65, alpha=.60, zorder=3)
        ax.scatter(*xy[good].T, s=18, color=BLUE, linewidths=.4, edgecolors='white', zorder=4)
        for path, color, style in [(data['greedy']['nodes'], GREEN, '-'),
                                    (data['forced_bad_first_then_greedy']['nodes'], RED, '--')]:
            for source, target in zip(path, path[1:]):
                ax.annotate('', xy[target], xy[source],
                            arrowprops={'arrowstyle': '-|>', 'color': color, 'lw': 1.8,
                                        'linestyle': style, 'shrinkA': 4, 'shrinkB': 4,
                                        'mutation_scale': 11}, zorder=6)
        ax.scatter(*xy[data['goal']], marker='*', s=145, color='#202D34', zorder=7)
        ax.scatter(*xy[data['start']], marker='s', s=65, facecolor='white', edgecolor='#202D34', zorder=7)
        # Dense clusters stay legible by naming only the start, goal and bad branch.
        overview = [data['start'], data['goal'], data['forced_bad_first_then_greedy']['nodes'][1]]
        for node in overview:
            ax.annotate(names[node], xy[node], xytext=(6, 7), textcoords='offset points',
                        fontsize=10, fontweight='bold', color='#202D34',
                        bbox={'boxstyle': 'round,pad=.15', 'fc': 'white', 'ec': 'none', 'alpha': .85}, zorder=8)
        ax.set_title(title, fontsize=14, pad=12)
        ax.set_aspect('equal', adjustable='datalim')
        ax.margins(.12)
        ax.axis('off')
    handles = [Line2D([], [], marker='o', color=BLUE, linestyle='none', label='可清空'),
               Line2D([], [], marker='x', color=RED, linestyle='none', label='不可清空'),
               Line2D([], [], color=GREEN, lw=2, label='实际 greedy 路径'),
               Line2D([], [], color=RED, lw=2, linestyle='--', label='先选坏动作的分支')]
    fig.legend(handles=handles, loc='outside lower center', ncol=4, frameon=False)
    save(fig, out, 'qmap_projection')


def draw_solution_zoom(data, out):
    fig, ax = plt.subplots(figsize=(7.6, 6.3), layout='constrained')
    xy = np.array([node['tsne'] for node in data['nodes']])
    remaining = np.array([node['exact_remaining'] for node in data['nodes']])
    good = remaining >= 0
    edges = np.array([edge for edge in data['edges'] if good[edge[0]] and good[edge[1]]])
    ax.add_collection(LineCollection(xy[edges[:, :2]], colors=GRAY, linewidths=.8, alpha=.55))
    points = ax.scatter(*xy[good].T, c=remaining[good], cmap='Blues', vmin=0, vmax=5,
                        edgecolors='#66838F', linewidths=.65, s=60, zorder=3)
    path = data['greedy']['nodes']
    for source, target in zip(path, path[1:]):
        ax.annotate('', xy[target], xy[source],
                    arrowprops={'arrowstyle': '-|>', 'color': GREEN, 'lw': 2,
                                'shrinkA': 5, 'shrinkB': 5, 'mutation_scale': 14}, zorder=5)
    ax.scatter(*xy[data['goal']], marker='*', s=160, color='#202D34', zorder=6)
    ax.scatter(*xy[data['start']], marker='s', s=75, facecolor='white', edgecolor='#202D34', zorder=6)
    names = labels(data)
    for node in path:
        ax.annotate(names[node], xy[node], xytext=(8, 8), textcoords='offset points',
                    fontsize=12, fontweight='bold', color='#202D34',
                    bbox={'boxstyle': 'round,pad=.1', 'fc': 'white', 'ec': 'none', 'alpha': .85}, zorder=7)
    lo, hi = xy[good].min(axis=0), xy[good].max(axis=0)
    pad = np.maximum((hi-lo) * .15, .1)
    ax.set_xlim(lo[0]-pad[0], hi[0]+pad[0])
    ax.set_ylim(lo[1]-pad[1], hi[1]+pad[1])
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('可清空区域 · t-SNE 放大', fontsize=15, pad=12)
    bar = fig.colorbar(points, ax=ax, shrink=.65, ticks=range(6))
    bar.set_label('真实剩余步数')
    save(fig, out, 'qmap_solution_projection')


def cropped_board(node, data):
    r0, c0 = data['crop_origin']
    height, width = data['crop_shape']
    mask = int(node['mask'])
    return np.array([[(mask >> (r * 10 + c)) & 1 for c in range(c0, c0+width)]
                     for r in range(r0, r0+height)])


def draw_paths(data, out):
    names = labels(data)
    success = data['greedy']['nodes']
    failed = data['forced_bad_first_then_greedy']['nodes']
    fig, axes = plt.subplots(2, len(success), figsize=(13.6, 4.8))
    fig.subplots_adjust(left=.02, right=.98, top=.84, bottom=.09, wspace=.35, hspace=.65)
    for ax in axes.flat:
        ax.axis('off')
    for row, path, color in [(0, success, GREEN), (1, failed, RED)]:
        for col, index in enumerate(path):
            if row == 1 and col == 0:
                continue
            ax = axes[row, col]
            board = cropped_board(data['nodes'][index], data)
            ax.imshow(board, cmap=ListedColormap(['#F1F4F5', BLUE]), vmin=0, vmax=1)
            for x in np.arange(-.5, board.shape[1], 1):
                ax.axvline(x, color='white', lw=1.2)
            for y in np.arange(-.5, board.shape[0], 1):
                ax.axhline(y, color='white', lw=1.2)
            ax.add_patch(Rectangle((-.5, -.5), board.shape[1], board.shape[0],
                                   fill=False, edgecolor=color, lw=1.7))
            ax.set_title(names[index], fontsize=13, color='#202D34', pad=9)
            if col:
                previous = axes[0, 0] if row == 1 and col == 1 else axes[row, col-1]
                connection = ConnectionPatch(xyA=(1.03, .5), xyB=(-.03, .5),
                                             coordsA='axes fraction', coordsB='axes fraction',
                                             axesA=previous, axesB=ax, color=color,
                                             arrowstyle='-|>', mutation_scale=12, lw=1.7,
                                             linestyle='--' if row == 1 else '-')
                fig.add_artist(connection)
    fig.suptitle('同一起点的两条转移分支', fontsize=16)
    save(fig, out, 'qmap_solution_branches')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--export-dir', type=Path)
    parser.add_argument('--data', type=Path, default=HERE / 'qmap_geometry.json')
    parser.add_argument('--out', type=Path, default=HERE)
    args = parser.parse_args()
    if args.export_dir:
        data = project(args.export_dir)
        # Keep the complete small graph and its 2D coordinates; raw Q stays in runs/.
        args.data.write_text(json.dumps(data, ensure_ascii=False, separators=(',', ':')) + '\n', encoding='utf-8')
    else:
        data = json.loads(args.data.read_text(encoding='utf-8'))
    args.out.mkdir(parents=True, exist_ok=True)
    draw_projections(data, args.out)
    draw_solution_zoom(data, args.out)
    draw_paths(data, args.out)
    print(json.dumps(data['projection'], ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
