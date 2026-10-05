"""Four compact report figures from completed, counted validation evidence."""

from argparse import ArgumentParser
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator


RESULTS = Path(__file__).resolve().parents[1] / 'results'
BLUE, ORANGE, GRAY, VERMILION = '#0072B2', '#E69F00', '#78828C', '#D55E00'


def load_data(results):
    summary = json.loads((results / 'long_trajectory_summary.json').read_text())
    turns = json.loads((results / 'blocks_rollout_turn_analysis.json').read_text())
    targets = json.loads((results / 'blocks_target_analysis.json').read_text())
    references = json.loads((results / 'learning_curve_baselines.json').read_text())
    data = dict(reference={}, rollout={}, map_agreement={}, blocks_goals={}, baselines=references,
        sources=['long_trajectory_summary.json', 'blocks_rollout_turn_analysis.json',
                 'blocks_target_analysis.json', 'learning_curve_baselines.json'])
    for task in ('path', 'blocks'):
        reading, rollout, agreement = {}, {}, {}
        for key, battery in summary['evaluation_batteries'].items():
            parts = key.split('/')
            if parts[0] != task or parts[2] not in ('validation_reference_map', 'validation_rollout_map'):
                continue
            epoch = summary['snapshot']['checkpoint_epochs'][task].get(parts[1], {}).get('epoch')
            if epoch is None or not float(epoch).is_integer() or epoch < 1:
                continue
            if battery['completion'] != 'complete':
                continue
            epoch = int(epoch)
            if parts[2] == 'validation_reference_map':
                audit = battery['reachability_audit']
                assert audit['missing_reachability_turns'] == 0
                reading[epoch] = dict(epoch=epoch, denominator=audit['current_reachable_turns'],
                    reachable=audit['keeps_goal_reachable'], shortest=audit['action_environment_shortest'])
                agreement[epoch] = dict(epoch=epoch, denominator=battery['counts']['decision_turns'],
                                        correct=battery['counts']['action_map_minimum'])
            else:
                value = battery['rollout']
                rollout[epoch] = dict(epoch=epoch, denominator=value['attempts'],
                                      reached=value['reached'], shortest=value['reached_shortest'])
        if task == 'blocks':
            for checkpoint in turns['checkpoints'].values():
                epoch = int(checkpoint['epoch'])
                ref = checkpoint['modes']['reference']['overall']
                reading[epoch] = dict(epoch=epoch, denominator=ref['solvable'], reachable=ref['kept'], shortest=ref['shortest'])
                # Exact all-decision counts are saved in the earlier summary;
                # final counts follow directly from the complete reference cases.
                correct = round(ref['map_minimum_all_decisions_rate'] * ref['decisions'])
                agreement[epoch] = dict(epoch=epoch, denominator=ref['decisions'], correct=correct)
                value = checkpoint['modes']['rollout']
                rollout[epoch] = dict(epoch=epoch, denominator=value['cases'], reached=value['reached'], shortest=value['reached_shortest'])
        expected = list(range(1, 4 if task == 'path' else 6))
        assert sorted(reading) == sorted(rollout) == sorted(agreement) == expected
        assert all(r['denominator'] == references['tasks'][task]['reference_decisions'] for r in reading.values())
        data['reference'][task] = [reading[i] for i in expected]
        data['rollout'][task] = [rollout[i] for i in expected]
        data['map_agreement'][task] = [agreement[i] for i in expected]
    for checkpoint, modes in targets['validation'].items():
        epoch = int(turns['checkpoints'][checkpoint]['epoch'])
        data['blocks_goals'][str(epoch)] = {}
        for goal in ('empty', 'nonempty'):
            value = modes['rollout'][goal]
            data['blocks_goals'][str(epoch)][goal] = dict(denominator=value['cases'],
                reached=value['reached'], shortest=value['reached_shortest'])
    return data


def style():
    plt.rcParams.update({
        'font.family': 'DejaVu Sans', 'font.size': 10.5,
        'axes.titlesize': 11, 'axes.titleweight': 'medium',
        'axes.labelsize': 10.5, 'xtick.labelsize': 9.5, 'ytick.labelsize': 9.5,
        'axes.edgecolor': '#5F6670', 'axes.linewidth': .7,
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.axisbelow': True, 'grid.color': '#E3E6EA', 'grid.linewidth': .65,
        'lines.linewidth': 2.3, 'lines.markersize': 5.5,
        'legend.frameon': False, 'legend.fontsize': 10,
        'figure.facecolor': 'white', 'axes.facecolor': 'white',
        'savefig.facecolor': 'white', 'savefig.dpi': 220,
        'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
    })


def axes_style(ax, epochs, ylabel, ylim=(0, 105), ystep=20):
    ax.set_xlim(min(epochs) - .2, max(epochs) + .2)
    ax.set_xticks(epochs)
    ax.set_ylim(*ylim)
    ax.yaxis.set_major_locator(MultipleLocator(ystep))
    ax.grid(axis='y')
    ax.set_ylabel(ylabel)
    ax.set_xlabel('Training epoch')
    ax.tick_params(length=3, width=.6)


def reference_line(ax, y, label, color, linestyle):
    return ax.axhline(100 * y, label=label, color=color, linestyle=linestyle, linewidth=1.7, zorder=2)


def curve(ax, rows, key, label='Map + LLM', color=BLUE, marker='o'):
    return ax.plot([r['epoch'] for r in rows], [100 * r[key] / r['denominator'] for r in rows],
                   color=color, marker=marker, markeredgecolor='white', markeredgewidth=.7,
                   label=label, zorder=4)[0]


def finish(fig, title, handles, output, name):
    fig.suptitle(title, fontsize=14, fontweight='medium', y=.99)
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.5, .95),
               ncol=len(handles), handlelength=2.6, columnspacing=2.2)
    single_row = len(fig.axes) == 2
    fig.subplots_adjust(left=.09, right=.98, top=.76 if single_row else .85,
                        bottom=.16 if single_row else .10, hspace=.55, wspace=.30)
    for extension in ('png', 'pdf', 'svg'):
        path = output / f'{name}.{extension}'
        fig.savefig(path, bbox_inches='tight', pad_inches=.08)
        if extension == 'svg':
            # Matplotlib puts trailing spaces in multiline SVG path attributes.
            path.write_text('\n'.join(line.rstrip() for line in path.read_text(encoding='utf-8').splitlines()) + '\n',
                            encoding='utf-8')
    plt.close(fig)


def reference_figure(data, output):
    fig, axes = plt.subplots(2, 2, figsize=(10.4, 6.6))
    handles = None
    for col, task in enumerate(('path', 'blocks')):
        rows = data['reference'][task]
        baseline = data['baselines']['tasks'][task]['reference']
        for index, (metric, title) in enumerate((('reachable', 'Action keeps goal reachable'), ('shortest', 'Action on a shortest path'))):
            ax = axes[index, col]
            greedy = reference_line(ax, baseline['greedy_' + metric], 'Map greedy', ORANGE, (0, (5, 3)))
            chance = reference_line(ax, baseline['chance_' + metric], 'Uniform chance', GRAY, (0, (1.3, 2.4)))
            model = curve(ax, rows, metric)
            ax.set_title(f'{"Pathfinding" if task == "path" else "Blocks"} · {title}', pad=10)
            axes_style(ax, [r['epoch'] for r in rows], 'Action rate (%)')
            handles = [model, greedy, chance]
    finish(fig, 'Reference: learning to choose actions from the map', handles, output, 'reference_learning_curves')


def rollout_figure(data, output):
    fig, axes = plt.subplots(2, 2, figsize=(10.4, 6.6))
    handles = None
    for col, task in enumerate(('path', 'blocks')):
        rows = data['rollout'][task]
        baseline = data['baselines']['tasks'][task]['rollout']['all']
        for index, (metric, title) in enumerate((('reached', 'Goal reached'), ('shortest', 'Goal reached on a shortest path'))):
            ax = axes[index, col]
            greedy = reference_line(ax, baseline[metric] / baseline['tasks'], 'Map greedy', ORANGE, (0, (5, 3)))
            model = curve(ax, rows, metric)
            ax.set_title(f'{"Pathfinding" if task == "path" else "Blocks"} · {title}', pad=10)
            axes_style(ax, [r['epoch'] for r in rows], 'Task rate (%)',
                       ylim=(0, 105) if index == 0 else (0, 50), ystep=20 if index == 0 else 10)
            handles = [model, greedy]
    finish(fig, 'Rollout: completing tasks with the map', handles, output, 'rollout_learning_curves')


def goal_figure(data, output):
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.1))
    epochs = sorted(map(int, data['blocks_goals']))
    for col, (metric, title) in enumerate((('reached', 'Goal reached'), ('shortest', 'Goal reached on a shortest path'))):
        ax = axes[col]
        for goal, color, marker in (('empty', BLUE, 'o'), ('nonempty', VERMILION, 's')):
            rows = [dict(data['blocks_goals'][str(epoch)][goal], epoch=epoch) for epoch in epochs]
            baseline = data['baselines']['tasks']['blocks']['rollout'][goal]
            reference_line(ax, baseline[metric] / baseline['tasks'], goal, color, (0, (5, 3)))
            curve(ax, rows, metric, goal, color, marker)
        ax.set_title(title, pad=10)
        axes_style(ax, epochs, 'Task rate (%)', ylim=(0, 80) if col == 0 else (0, 40), ystep=20 if col == 0 else 10)
    handles = [Line2D([], [], color=BLUE, marker='o', label='Empty goal'),
               Line2D([], [], color=VERMILION, marker='s', label='Nonempty goal'),
               Line2D([], [], color='#303943', linewidth=2.3, label='Map + LLM'),
               Line2D([], [], color='#303943', linestyle=(0, (5, 3)), linewidth=1.7, label='Map greedy')]
    finish(fig, 'Blocks rollout: empty vs nonempty goals', handles, output, 'blocks_goal_learning_curves')


def reading_figure(data, output):
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.1))
    handles = None
    for ax, task in zip(axes, ('path', 'blocks')):
        rows = data['map_agreement'][task]
        baseline = data['baselines']['tasks'][task]['map_agreement']
        greedy = reference_line(ax, 1., 'Map greedy', ORANGE, (0, (5, 3)))
        chance = reference_line(ax, baseline['chance'], 'Uniform chance', GRAY, (0, (1.3, 2.4)))
        model = curve(ax, rows, 'correct')
        ax.set_title('Pathfinding' if task == 'path' else 'Blocks', pad=10)
        axes_style(ax, [r['epoch'] for r in rows], 'Map-minimum action (%)')
        handles = [model, greedy, chance]
    finish(fig, 'Reference: reading map geometry', handles, output, 'map_reading_learning_curves')


def main():
    parser = ArgumentParser()
    parser.add_argument('--results', type=Path, default=RESULTS)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    output = args.out or args.results / 'figures'
    output.mkdir(parents=True, exist_ok=True)
    data = load_data(args.results)
    style()
    for make in (reference_figure, rollout_figure, goal_figure, reading_figure):
        make(data, output)
    (output / 'learning_curve_data.json').write_text(json.dumps(data, indent=2) + '\n')
    print('Saved four figures as PNG, PDF and SVG:', output)


if __name__ == '__main__':
    main()
