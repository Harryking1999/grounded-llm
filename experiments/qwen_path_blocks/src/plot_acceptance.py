"""Render the acceptance report figures from its reviewed compact audit."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--audit', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    data = json.loads(args.audit.read_text(encoding='utf-8'))
    args.out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Microsoft YaHei', 'DejaVu Sans'],
                         'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False})
    models = ['4b', '8b', '32b']
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, field, denom, title in zip(axes, ['successes', 'pass8_solved_cases'], [128, 16],
                                     ['单次最短解准确率', '每题八次至少成功一次（pass@8）']):
        counts = [data['path_'+m]['metrics']['path_undirected_256'][field] for m in models]
        bars = ax.bar([m.upper() for m in models], [n/denom*100 for n in counts], color='#287b9e', width=.55)
        ax.set(ylim=(0, 100), ylabel='比例（%）', title=title)
        ax.yaxis.grid(True, alpha=.15); ax.set_axisbelow(True)
        for bar, count in zip(bars, counts):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+2,
                    f'{count/denom:.2%}\n{count}/{denom}', ha='center', fontsize=11)
    fig.suptitle('Path · 同一 256 节点双向图，16 个起终点对', fontsize=15)
    fig.tight_layout(); fig.savefig(args.out/'path_accuracy.png', dpi=160); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6), gridspec_kw={'width_ratios': [1, 1.6]})
    labels, hits, segments = [], [], []
    for m in models:
        for condition in ['blocks8', 'blocks12']:
            s=data['blocks_'+m]['metrics'][condition]; c=s['diagnostic_counts']
            labels.append(f'{m.upper()} · {condition[6:]} 块'); hits.append(s['budget_truncations'])
            segments.append([c.get('illegal_action',0), c.get('illegal_before_truncation',0),
                             c.get('stopped_early',0), c.get('truncated_unresolved',0)])
    axes[0].barh(labels, [h/128*100 for h in hits], color='#8e9ca8')
    for i,h in enumerate(hits): axes[0].text(h/128*100+1, i, f'{h}/128 ({h/128:.1%})', va='center', fontsize=9)
    axes[0].set(xlim=(0,130), xticks=[0,25,50,75,100], title='实际触及输出上限', xlabel='比例（%）')
    colors=['#c15f4c','#e3a05f','#67959a','#cbd3da']
    names=['自然结束时非法','截断前已非法','合法但未清空','截断、原因未决']
    left=[0]*6
    for j,(color,name) in enumerate(zip(colors,names)):
        vals=[row[j] for row in segments]
        axes[1].barh(labels, vals, left=left, color=color,label=name)
        for i,(v,l) in enumerate(zip(vals,left)):
            if v>=6: axes[1].text(l+v/2,i,str(v),ha='center',va='center',fontsize=10)
        left=[l+v for l,v in zip(left,vals)]
    axes[1].set(xlim=(0,128), title='错误归因：已确认非法优先', xlabel='尝试次数（每行合计 128）')
    for ax in axes: ax.invert_yaxis()
    fig.legend(*axes[1].get_legend_handles_labels(), loc='lower center', ncol=4, frameon=False)
    fig.suptitle('Blocks · 准确率全部为 0%，触顶与错误原因分开统计',fontsize=15)
    fig.tight_layout(rect=(0,.09,1,.96));fig.savefig(args.out/'blocks_failures.png',dpi=160);plt.close(fig)

    fig,axes=plt.subplots(1,2,figsize=(11,4.5))
    # Both cases use blocks8_00. A small local crop makes the exact error readable.
    grids=[['0000001100','0000111000','0001100000','0001000000'],
           ['0000001100','0000000000','0001100000','0001000000']]
    marks=[[(4,3),(4,4),(4,5)],[(3,6),(3,7),(4,6)]]
    for ax,rows,cells,title in zip(axes,grids,marks,
        ['B1：第 2 步覆盖初始空格 (4,3)','B2：第 3 步重复移除 (4,6)']):
        for rr,row in enumerate(rows,3):
            for cc in range(2,8):
                occupied=row[cc]=='1'
                ax.add_patch(Rectangle((cc-.5,rr-.5),1,1,facecolor='#547c90' if occupied else '#f0f3f5',edgecolor='white'))
                ax.text(cc,rr,'1' if occupied else '0',ha='center',va='center',color='white' if occupied else '#617080')
        for rr,cc in cells: ax.add_patch(Rectangle((cc-.46,rr-.46),.92,.92,fill=False,edgecolor='#ce4b36',linewidth=3))
        ax.set(xlim=(1.5,7.5),ylim=(6.5,2.5),xticks=range(2,8),yticks=range(3,7),xlabel='列（从 0 计）',ylabel='行（从 0 计）',title=title)
        ax.set_aspect('equal')
    fig.suptitle('同一棋盘的两类非法动作：红框是模型要移除的三个格子',fontsize=13)
    fig.tight_layout();fig.savefig(args.out/'blocks_cases.png',dpi=160);plt.close(fig)


if __name__ == '__main__':
    main()
