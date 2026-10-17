"""Compare matched blocks plans with and without predicted states."""
import argparse
from collections import Counter
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path

from .budget import output_limit, rescore
from .report import build
from .run import write_json


def latest_records(folder):
    rows = [json.loads(p.read_text(encoding='utf-8')) for p in (Path(folder)/'records').glob('*.json')]
    return {r['case_id']:r for r in sorted(rows,key=lambda r:(r['case_id'],r['attempt']))}


def publish(baseline, run, out):
    limit = output_limit()
    original,_ = build(baseline,limit)
    current,_ = build(run,limit)
    old = {i:rescore(r,limit) for i,r in latest_records(baseline).items()}
    new = {i:rescore(r,limit) for i,r in latest_records(run).items()}
    ids = {c['id'] for c in json.loads((Path(run)/'suite.json').read_text(encoding='utf-8'))['cases']}
    paired = [i for i in ids if i in new and not new[i].get('service_error')]
    transitions = Counter((bool(old[i]['verdict'].get('solved')),bool(new[i]['verdict'].get('solved'))) for i in paired)
    within = [r for r in new.values() if r.get('budget_status') == 'within']
    evaluated = sum(r['verdict'].get('state_reports_evaluated',0) for r in within)
    valid = sum(r['verdict'].get('state_reports_valid',0) for r in within)
    correct = sum(r['verdict'].get('state_reports_correct',0) for r in within)
    now = datetime.now(timezone(timedelta(hours=8))).strftime('%Y-%m-%d %H:%M（北京时间）')
    lines = ['# 积木：逐步状态输出对照','',f'更新：{now}。','',
             '同一模型 5.6-sol high、同一批 40 道积木题、相同规则和一次完整计划；新组仅在原提示词末尾追加每步 board_after 要求，无中途反馈或工具。统一按 16,000 输出 token（思考＋答案）评分，超限即错。','',
             '| 条件 | 有效／计划 | 预算内解出 | 原执行解出 | 严格合同通过 |',
             '|---|---:|---:|---:|---:|']
    for label,result in (('无逐步状态',original),('逐步报告状态',current)):
        row = result['tasks']['blocks']
        lines.append(f"| {label} | {row['valid_responses']}/{row['planned']} | {row['solved']} | {row['execution_solved']} | {row['contract_pass']} |")
    row = current['tasks']['blocks']
    lines += ['',f"新组状态：{current['status']}；服务失败待补 {row['service_errors']} 题，超限 {row['budget_exceeded']} 题。未完成时只报告已有样本，不将未返回题当作错误。",'',
              '## 逐步状态与配对变化','',
              f'预算内答卷的合法动作前缀：完整状态正确 {correct}/{evaluated}，格式有效 {valid}/{evaluated}。缺失状态报告计错；首次非法动作及其后不评价状态。严格合同额外要求所有动作后的棋盘均正确，动作执行成功单独计分。','',
              f"已返回的 {len(paired)} 道配对题：两组均解出 {transitions[True,True]}；原组解出、新组失败 {transitions[True,False]}；原组失败、新组解出 {transitions[False,True]}；两组均失败 {transitions[False,False]}。",'',
              '## 解释边界','',
              '原无状态组请求上限混用 32k／16k，但这 40 份答卷均未超过 16k；新组全部请求 16k。题目、模型请求与规则对齐，生成请求上限仍存在差异，不能称为完全隔离的单变量实验。状态文本是模型预测，不能覆盖真实环境或用于反馈。','',
              f"新组错误分类：{json.dumps(row['failures'],ensure_ascii=False)}。模型答错、截断和超限均保留，只补服务失败。",'',
              '[正式配置](../configs/sol56_blocks40_state_report.json) · [原五任务对照](model_retest.md)','']
    out = Path(out)
    out.mkdir(parents=True,exist_ok=True)
    (out/'blocks_state_report.md').write_text('\n'.join(lines),encoding='utf-8',newline='\n')
    write_json(out/'blocks_state_report.json',{'updated_at':now,'baseline_blocks':original['tasks']['blocks'],
        'state_report':current,'within_budget_state_accuracy':{'evaluated':evaluated,'valid':valid,'correct':correct},
        'paired':{'answered':len(paired),'both_solved':transitions[True,True],
                  'lost_success':transitions[True,False],'gained_success':transitions[False,True],
                  'both_failed':transitions[False,False]}})
    return current


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--baseline',required=True,type=Path)
    parser.add_argument('--run',required=True,type=Path)
    parser.add_argument('--out',required=True,type=Path)
    args = parser.parse_args()
    result = publish(args.baseline,args.run,args.out)
    print(json.dumps({'status':result['status'],'blocks':result['tasks']['blocks']}))


if __name__ == '__main__':
    main()
