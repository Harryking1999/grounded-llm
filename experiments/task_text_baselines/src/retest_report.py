"""Summarize the two authorized retests, keeping incomplete samples explicit."""
import argparse
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path

from .protocol import ROOT
from .report import build, NAMES
from .run import write_json


def score(row):
    valid,planned,solved = row['valid_responses'],row['planned'],row['solved']
    if valid == planned:
        return f'{solved}/{planned}（{100*solved/planned:.1f}%）'
    return f'已答 {valid}/{planned}，解出 {solved}'


def publish(baseline, sol56, legacy, out):
    original,_ = build(baseline)
    older,_ = build(sol56)
    repeated,_ = build(legacy)
    historical = json.loads((ROOT/'experiments/sol_dag_blocks/results/summary.json').read_text(encoding='utf-8'))
    prior = historical['conditions']['blocks12']
    progress = json.loads((Path(legacy)/'progress.json').read_text(encoding='utf-8'))
    records = [json.loads(p.read_text(encoding='utf-8')) for p in (Path(legacy)/'records').glob('*.json')]
    latest = {r['case_id']:r for r in sorted(records,key=lambda r:(r['case_id'],r['attempt']))}
    now = datetime.now(timezone(timedelta(hours=8))).strftime('%Y-%m-%d %H:%M（北京时间）')
    lines = ['# GPT 模型与提示词复测','',f'更新：{now}。','',
             '## 初始五任务：5.6 与 6.1 同题对照','',
             '完全复用原 200 题与逐字相同的提示词，每题一次有效答案；两组均请求 high、相同输出预算。服务失败单列，正常答错不重抽。','',
             '| 任务 | 5.6-sol high | 6.1-sol high |','|---|---:|---:|']
    for task,row in original['tasks'].items():
        lines.append(f'| {NAMES[task]} | {score(older["tasks"][task])} | {score(row)} |')
    lines += ['',f"5.6 状态：{older['status']}；尚待补测的服务失败 {sum(r['service_errors'] for r in older['tasks'].values())} 题。",'',
              '## 旧十二块题：6.1 使用新提示词','',
              '保留原 16 张棋盘、原含 S/Z 十形状与空目标；采用当前规则文字和完整计划 JSON，不要求逐步 board_after。每题一次，最短移除数未求解。','',
              f"**当前结果：{score(repeated['tasks']['blocks'])}。** 状态：{progress['status']}。",'',
              f"历史 5.6 medium 为 {prior['successes']}/{prior['planned_samples']}（{100*prior['sample_success_rate']:.1f}%），每题八次。这次同时改变模型、提示词和思考设置，不能把差值归因于单一因素。",'',
              '| 原题 | 历史 5.6 解出／8 | 本轮 6.1 |','|---|---:|---|']
    suite = json.loads((Path(legacy)/'suite.json').read_text(encoding='utf-8'))
    labels = {'illegal_action':'非法动作','unsolved':'未完成','parse_error':'解析失败','truncated':'输出截断'}
    for case in suite['cases']:
        row = latest.get(case['id'])
        result = ('待返回' if row is None else '服务失败待补' if row.get('service_error') else
                  '成功' if row['verdict'].get('solved') else labels.get(row['verdict'].get('failure'),'失败'))
        lines.append(f"| {case['id']} | {prior['per_case'][case['id']]['successes']}/8 | {result} |")
    repeated_row = repeated['tasks']['blocks']
    if repeated_row['valid_responses'] == repeated_row['planned']:
        former_failures = [case['id'] for case in suite['cases'] if prior['per_case'][case['id']]['successes'] == 0]
        recovered = sum(bool(latest[i]['verdict'].get('solved')) for i in former_failures)
        lines += ['', f"历史 0/8 的 {len(former_failures)} 道题，本轮解出 {recovered}/{len(former_failures)}。严格 JSON 与终态声明通过 {repeated_row['contract_pass']}/{repeated_row['planned']}；执行成功单独计分。"]
    lines += ['', '## 接口与解释边界','']
    for name,result in (('5.6 同题组',older),('6.1 旧题组',repeated)):
        metadata = result['response_metadata_coverage']
        lines.append(f"- {name}：有效答卷 {metadata['valid_answers']}，回显 high {metadata['effort_counts'].get('high',0)}，未回传强度 {metadata['effort_counts'].get('not_returned',0)}；超过请求输出预算 {result['exceeded_output_limit']} 份，服务失败尝试 {result['service_error_attempts']} 次。")
    lines += ['- 第三方回显不独立证明后台模型或强度；实际超限另报，不声称严格等预算。','',
              '[五任务配置](../configs/sol56_high_same40.json) · [旧题复测配置](../configs/sol61_legacy_blocks12_new_prompt.json) · [返回主报告](report.md)','']
    out = Path(out)
    out.mkdir(parents=True,exist_ok=True)
    (out/'model_retest.md').write_text('\n'.join(lines),encoding='utf-8',newline='\n')
    write_json(out/'model_retest.json',{'updated_at':now,'original_sol61':original,'sol56':older,'legacy_sol61':repeated})
    return older,repeated


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--baseline',required=True,type=Path)
    parser.add_argument('--sol56',required=True,type=Path)
    parser.add_argument('--legacy',required=True,type=Path)
    parser.add_argument('--out',required=True,type=Path)
    args = parser.parse_args()
    older,repeated = publish(args.baseline,args.sol56,args.legacy,args.out)
    print(json.dumps({'sol56':older['status'],'legacy_sol61':repeated['status']}))


if __name__ == '__main__':
    main()
