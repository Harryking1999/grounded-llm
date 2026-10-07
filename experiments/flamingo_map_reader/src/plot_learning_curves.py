"""Training curves and final trajectory-step comparisons from saved evidence."""

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
    references = json.loads((results / 'learning_curve_baselines.json').read_text())
    steps = json.loads((results / 'final_trajectory_step_analysis.json').read_text())
    data = dict(reference={}, rollout={}, rollout_all={}, map_agreement={}, baselines=references,
        final_steps=steps,
        sources=['long_trajectory_summary.json', 'blocks_rollout_turn_analysis.json',
                 'learning_curve_baselines.json', 'final_trajectory_step_analysis.json'])
    for task in ('path', 'blocks'):
        reading, rollout, agreement, all_rollout = {}, {}, {}, {}
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
                initial = battery['groups']['initial_goal']['rollout']
                all_rollout[epoch] = dict(epoch=epoch, denominator=value['attempts'],
                                         reached=value['reached'], shortest=value['reached_shortest'])
                rollout[epoch] = dict(epoch=epoch, denominator=value['attempts'] - initial['attempts'],
                    reached=value['reached'] - initial['reached'],
                    shortest=value['reached_shortest'] - initial['reached_shortest'])
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
                all_rollout[epoch] = dict(epoch=epoch, denominator=value['cases'],
                    reached=value['reached'], shortest=value['reached_shortest'])
                initial = value['cases'] - value['nonzero_cases']
                assert value['reached'] - value['nonzero_reached'] == initial
                rollout[epoch] = dict(epoch=epoch, denominator=value['nonzero_cases'],
                    reached=value['nonzero_reached'], shortest=value['reached_shortest'] - initial)
        expected = list(range(1, 4 if task == 'path' else 6))
        assert sorted(reading) == sorted(rollout) == sorted(agreement) == expected
        assert all(r['denominator'] == references['tasks'][task]['reference_decisions'] for r in reading.values())
        data['reference'][task] = [reading[i] for i in expected]
        data['rollout'][task] = [rollout[i] for i in expected]
        data['rollout_all'][task] = [all_rollout[i] for i in expected]
        data['map_agreement'][task] = [agreement[i] for i in expected]
        final = steps['tasks'][task]
        assert final['epoch'] == expected[-1]
        outcomes = final['nonzero_outcomes']['rollout']
        assert rollout[expected[-1]] == dict(epoch=expected[-1], denominator=outcomes['cases'],
            reached=outcomes['reached'], shortest=outcomes['shortest'])
        for mode in ('reference', 'rollout'):
            rows = final['modes'][mode]['by_step']
            assert sum(row['solvable'] for row in rows) == final['modes'][mode]['overall']['solvable']
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
        baseline = data['final_steps']['tasks'][task]['nonzero_outcomes']['greedy']
        for index, (metric, title) in enumerate((('reached', 'Goal reached'), ('shortest', 'Goal reached on a shortest path'))):
            ax = axes[index, col]
            greedy = reference_line(ax, baseline[metric] / baseline['cases'], 'Map greedy', ORANGE, (0, (5, 3)))
            model = curve(ax, rows, metric)
            ax.set_title(f'{"Pathfinding" if task == "path" else "Blocks"} · {title}', pad=10)
            axes_style(ax, [r['epoch'] for r in rows], 'Task rate (%)',
                       ylim=(0, 105) if index == 0 else (0, 50), ystep=20 if index == 0 else 10)
            handles = [model, greedy]
    finish(fig, 'Rollout: nonzero-step tasks', handles, output, 'rollout_learning_curves')


def step_axes(ax, maximum, ylabel, xlabel, start=1):
    ax.set_xlim(start - .15, maximum + .15)
    ax.set_ylim(0, 105)
    interval = 1 if maximum <= 12 else 5
    ticks = list(range(start, maximum + 1)) if interval == 1 else list(range(0, maximum + 1, interval))
    if start == 1 and interval != 1:
        ticks = [1] + [tick for tick in ticks if tick]
    if ticks[-1] != maximum:
        ticks.append(maximum)
    ax.set_xticks(ticks)
    ax.yaxis.set_major_locator(MultipleLocator(20))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(axis='y')
    ax.tick_params(length=3, width=.6)


def decision_curve(ax, rows, key, color, marker):
    rows = [row for row in rows if row['solvable']]
    x = [row['step'] for row in rows]
    y = [100 * row[key] / row['solvable'] for row in rows]
    # A singleton late decision should not look like a reliable continuation.
    ax.plot(x, [value if row['solvable'] >= 30 else float('nan') for row, value in zip(rows, y)],
            color=color, marker=marker, markeredgecolor='white', markeredgewidth=.7, zorder=4)
    low = [(row['step'], value) for row, value in zip(rows, y) if row['solvable'] < 30]
    if low:
        ax.plot([at for at, _ in low], [value for _, value in low], linestyle='none',
                marker=marker, markersize=4, color=color, markerfacecolor='white',
                markeredgewidth=1.0, alpha=.7, zorder=4)


def progress_curve(ax, rows, key, color, dashed=False):
    ax.step([row['step'] for row in rows], [100 * row[key] / row['denominator'] for row in rows],
            where='post', color=color, linestyle=(0, (5, 3)) if dashed else '-', zorder=3)


def step_handles():
    return [Line2D([], [], color=BLUE, marker='o', label='Reference'),
            Line2D([], [], color=VERMILION, marker='s', label='Rollout'),
            Line2D([], [], color=ORANGE, linestyle=(0, (5, 3)), label='Map greedy')]


def final_blocks_figure(data, output):
    final = data['final_steps']['tasks']['blocks']
    fig, axes = plt.subplots(2, 2, figsize=(10.4, 6.6))
    for ax, key, title in zip(axes[0], ('kept', 'shortest'),
                             ('Action keeps goal reachable', 'Action on a shortest path')):
        for mode, color, marker in (('reference', BLUE, 'o'), ('rollout', VERMILION, 's')):
            decision_curve(ax, final['modes'][mode]['by_step'], key, color, marker)
        maximum = max(row['step'] for mode in final['modes'].values()
                      for row in mode['by_step'] if row['solvable'])
        step_axes(ax, maximum, 'Action rate (%)', 'Decision step')
        ax.set_title(title, pad=10)
    for ax, key, title in zip(axes[1], ('reached', 'dead'),
                             ('Goal reached (cumulative)', 'Dead end (cumulative)')):
        progress_curve(ax, final['progress']['rollout'], key, VERMILION)
        progress_curve(ax, final['progress']['greedy'], key, ORANGE, dashed=True)
        step_axes(ax, final['progress']['rollout'][-1]['step'], 'Tasks (%)',
                  'Executed actions', start=0)
        ax.set_title(title, pad=10)
    finish(fig, 'Blocks · final epoch 5', step_handles(), output, 'blocks_final_trajectory_steps')


def final_path_figure(data, output):
    final = data['final_steps']['tasks']['path']
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.1))
    for mode, color, marker in (('reference', BLUE, 'o'), ('rollout', VERMILION, 's')):
        decision_curve(axes[0], final['modes'][mode]['by_step'], 'shortest', color, marker)
    maximum = max(row['step'] for mode in final['modes'].values()
                  for row in mode['by_step'] if row['solvable'])
    step_axes(axes[0], maximum, 'Action rate (%)', 'Decision step')
    axes[0].set_title('Action on a shortest path', pad=10)
    progress_curve(axes[1], final['progress']['rollout'], 'reached', VERMILION)
    progress_curve(axes[1], final['progress']['greedy'], 'reached', ORANGE, dashed=True)
    step_axes(axes[1], final['progress']['rollout'][-1]['step'], 'Tasks (%)',
              'Executed actions', start=0)
    axes[1].set_title('Goal reached (cumulative)', pad=10)
    finish(fig, 'Pathfinding · final epoch 3', step_handles(), output, 'path_final_trajectory_steps')


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
    for make in (reference_figure, rollout_figure, reading_figure, final_blocks_figure, final_path_figure):
        make(data, output)
    (output / 'learning_curve_data.json').write_text(json.dumps(data, indent=2) + '\n')
    print('Saved five figures as PNG, PDF and SVG:', output)


if __name__ == '__main__':
    main()
