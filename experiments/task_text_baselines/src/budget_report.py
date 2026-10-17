"""Report 8k/16k sensitivity from saved answers, without new model calls."""
import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .budget import POLICY, output_limit
from .report import build, NAMES
from .run import write_json


def cell(row):
    if row['valid_responses'] != row['planned']:
        return f"已答 {row['valid_responses']}/{row['planned']}，解出 {row['solved']}"
    return f"{row['solved']}/{row['planned']}（{100*row['solved']/row['planned']:.1f}%）"


def main():
    parser = argparse.ArgumentParser(__doc__)
    for name in ('qwen','sol56','sol61','legacy','states'):
        parser.add_argument('--'+name,required=True,type=Path)
    parser.add_argument('--sz',type=Path)
    parser.add_argument('--out',required=True,type=Path)
    args = parser.parse_args()
    limits = [output_limit(POLICY.with_name('budget_8k.json')),output_limit()]
    groups = {'Qwen3-32B':args.qwen,'5.6-sol high':args.sol56,'6.1-sol high':args.sol61,
              '6.1 旧十形状十二块':args.legacy,'5.6 原八形状＋逐步状态':args.states}
    if args.sz:
        groups['5.6 S/Z 八形状＋逐步状态'] = args.sz
    results = {label:{limit:build(path,limit)[0] for limit in limits} for label,path in groups.items()}
    now = datetime.now(timezone(timedelta(hours=8))).strftime('%Y-%m-%d %H:%M（北京时间）')
    lines = ['# 输出预算敏感性：8k 与 16k','',f'更新：{now}。','',
             '输出 token 包含思考与最终答案；严格大于阈值即计失败，恰好等于阈值可计成功。下表重算已保存答卷，没有新增模型调用，不等于在 8k 请求上限下重新生成。','',
             '## 初始五任务：8k 内到达','',
             '| 任务 | Qwen3-32B | 5.6-sol high | 6.1-sol high |','|---|---:|---:|---:|']
    main_labels = list(groups)[:3]
    for task in results[main_labels[0]][limits[0]]['tasks']:
        lines.append('| '+NAMES[task]+' | '+' | '.join(cell(results[label][limits[0]]['tasks'][task]) for label in main_labels)+' |')
    lines += ['', '| 模型 | 8k 内到达 | 16k 内到达 |','|---|---:|---:|']
    for label in main_labels:
        values = []
        for limit in limits:
            rows = results[label][limit]['tasks'].values()
            total = {key:sum(r[key] for r in rows) for key in ('valid_responses','planned','solved')}
            values.append(cell(total))
        lines.append('| '+label+' | '+' | '.join(values)+' |')
    lines += ['', '## 积木其他条件','', '| 条件 | 8k 内到达 | 16k 内到达 |','|---|---:|---:|']
    lines.append('| 5.6 原八形状、无逐步状态 | '+' | '.join(cell(results['5.6-sol high'][limit]['tasks']['blocks']) for limit in limits)+' |')
    for label in list(groups)[3:]:
        lines.append('| '+label+' | '+' | '.join(cell(results[label][limit]['tasks']['blocks']) for limit in limits)+' |')
    lines += ['', '8k 是额外评分口径；当前新实验仍请求 16k。不同形状条件的题目独立生成，不逐题配对。旧五任务还包含不同请求上限阶段，不能声称严格等生成预算。','',
              '[8k 评分合同](../configs/budget_8k.json) · [16k 评分合同](../configs/budget_16k.json)','']
    args.out.mkdir(parents=True,exist_ok=True)
    (args.out/'budget_sensitivity.md').write_text('\n'.join(lines),encoding='utf-8',newline='\n')
    compact = {label:{str(limit):{'status':r['status'],'tasks':{task:{key:row[key] for key in
               ('valid_responses','planned','solved','budget_exceeded','budget_unknown')} for task,row in r['tasks'].items()}}
               for limit,r in values.items()} for label,values in results.items()}
    write_json(args.out/'budget_sensitivity.json',{'updated_at':now,'runs':{k:str(v) for k,v in groups.items()},'results':compact})


if __name__ == '__main__':
    main()
