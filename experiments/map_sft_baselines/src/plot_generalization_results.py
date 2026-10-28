"""Render task-specific FLAN ID curves from the saved accepted results."""
from pathlib import Path
import argparse
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager


TASK_NAMES = {'blocks': '积木', 'cube2': '二阶魔方', 'path': '寻路',
              'bridge': '过桥', 'puzzle8': '华容道（8-Puzzle）',
              'token_swapping': '棋子交换'}
COLORS = {'4': '#2563A6', '5': '#D77924', '6': '#27836C'}
STYLES = {'4': ('-', 'o'), '5': ('--', 's'), '6': (':', '^')}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    base = Path(__file__).resolve().parents[1] / 'results'
    parser.add_argument('--results', type=Path, default=base / 'flan_generalization_results.json')
    parser.add_argument('--out', type=Path, default=base / 'figures')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    data = json.loads(args.results.read_text(encoding='utf-8'))
    font = Path('C:/Windows/Fonts/msyh.ttc')
    if font.exists():
        font_manager.fontManager.addfont(str(font))
        family = font_manager.FontProperties(fname=str(font)).get_name()
    else:
        family = ['Noto Sans CJK SC', 'Microsoft YaHei', 'DejaVu Sans']
    plt.rcParams.update({'font.family': family, 'font.size': 12,
                         'axes.unicode_minus': False, 'axes.spines.top': False,
                         'axes.spines.right': False, 'axes.edgecolor': '#8B96A3',
                         'axes.labelcolor': '#334155', 'text.color': '#172B3A',
                         'xtick.color': '#475569', 'ytick.color': '#475569',
                         'svg.fonttype': 'none', 'pdf.fonttype': 42})
    for task, name in TASK_NAMES.items():
        fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.7))
        handles = []
        for n, cohort in data['cohorts'].items():
            points = [(int(step), metrics)
                      for step, metrics in cohort['task_curves'].get(task, {}).items()]
            if not points:
                continue
            points.sort()
            steps = [step / 1000 for step, _ in points]
            completion = [100 * m['arrival_and_stop'] / m['tasks'] for _, m in points]
            follow = [100 * m['map_optimal']['value'] for _, m in points]
            style, marker = STYLES[n]
            for ax, values in zip(axes, [completion, follow]):
                line, = ax.plot(steps, values, label=f'{n}类训练', color=COLORS[n],
                                linestyle=style, marker=marker, markersize=5.5,
                                linewidth=2.1, markeredgecolor='white', markeredgewidth=.6)
            handles.append(line)
        for ax, title in zip(axes, ['正确完成率（%）', '跟图率（%）']):
            ax.set_title(title, loc='left', fontsize=13, weight='bold', pad=12)
            ax.set_xlim(0, 260)
            ax.set_ylim(0, 105)
            ax.set_xticks([0, 50, 100, 150, 200, 250])
            ax.set_yticks([0, 20, 40, 60, 80, 100])
            ax.set_xlabel('训练步数（千步）', labelpad=9)
            ax.grid(axis='y', color='#E5EAF0', linewidth=.8)
            ax.set_axisbelow(True)
        fig.suptitle(name, x=.07, y=.98, ha='left', fontsize=17, weight='bold')
        fig.legend(handles, [h.get_label() for h in handles], loc='upper right',
                   bbox_to_anchor=(.97, .995), ncol=3, frameon=False, fontsize=12)
        fig.text(.07, .025, 'ID固定题集：每任务250题；按已有评测结果绘制。',
                 fontsize=10.5, color='#64748B')
        fig.subplots_adjust(left=.07, right=.975, top=.79, bottom=.19, wspace=.22)
        for ext in ['png', 'pdf']:
            fig.savefig(args.out / f'flan_id_{task}.{ext}', dpi=180, facecolor='white')
        plt.close(fig)
    print(f'Rendered {len(TASK_NAMES)} task-specific curve figures to {args.out}')


if __name__ == '__main__':
    main()
