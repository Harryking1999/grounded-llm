"""Compare matched blocks plans with and without predicted states."""
import argparse
from collections import Counter
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path

from .budget import output_limit, rescore, POLICY
from .report import build
from .run import write_json


def latest_records(folder):
    rows = [json.loads(p.read_text(encoding='utf-8')) for p in (Path(folder)/'records').glob('*.json')]
    return {r['case_id']:r for r in sorted(rows,key=lambda r:(r['case_id'],r['attempt']))}


def publish(baseline, run, out):
    limit = output_limit()
    original,_ = build(baseline,limit)
    current,_ = build(run,limit)
    secondary_limit = output_limit(POLICY.with_name('budget_8k.json'))
    original8,_ = build(baseline,secondary_limit)
    current8,_ = build(run,secondary_limit)
    old = {i:rescore(r,limit) for i,r in latest_records(baseline).items()}
    new = {i:rescore(r,limit) for i,r in latest_records(run).items()}
    cases = json.loads((Path(run)/'suite.json').read_text(encoding='utf-8'))['cases']
    prior = {c['id']:c for c in json.loads((Path(baseline)/'suite.json').read_text(encoding='utf-8'))['cases']}
    ids = {c['id'] for c in cases}
    same_questions = all(c['id'] in prior and all(c.get(k) == prior[c['id']].get(k)
                         for k in ('initial','goal','shape_options')) for c in cases)
    paired = [i for i in ids if same_questions and i in new and not new[i].get('service_error')]
    transitions = Counter((bool(old[i]['verdict'].get('solved')),bool(new[i]['verdict'].get('solved'))) for i in paired)
    within = [r for r in new.values() if r.get('budget_status') == 'within']
    evaluated = sum(r['verdict'].get('state_reports_evaluated',0) for r in within)
    valid = sum(r['verdict'].get('state_reports_valid',0) for r in within)
    correct = sum(r['verdict'].get('state_reports_correct',0) for r in within)
    now = datetime.now(timezone(timedelta(hours=8))).strftime('%Y-%m-%d %H:%M（北京时间）')
    lines = ['# 积木：逐步状态输出对照' if same_questions else '# 积木：双格替换为 S/Z','',f'更新：{now}。','',
             ('同一模型 5.6-sol high、同一批 40 道积木题、相同规则和一次完整计划；新组仅在原提示词末尾追加每步 board_after 要求。' if same_questions else
              '5.6-sol high，10×10、八形状、12 次构造放置。将 4、5 号双格替换为固定朝向 S、Z，其他六形状不变，生成新的 40 道可解题；两组均逐步输出 board_after。'),
             '无中途反馈或工具，请求上限为 16k。分别按 16,000／8,000 输出 token（思考＋答案）重算，超过相应阈值即错。','',
             '| 条件 | 有效／计划 | 16k 内解出 | 8k 内解出 |',
             '|---|---:|---:|---:|']
    labels = ('无逐步状态','逐步报告状态') if same_questions else ('原八形状＋逐步状态','S/Z 八形状＋逐步状态')
    for label,result,result8 in ((labels[0],original,original8),(labels[1],current,current8)):
        row = result['tasks']['blocks']
        lines.append(f"| {label} | {row['valid_responses']}/{row['planned']} | {row['solved']} | {result8['tasks']['blocks']['solved']} |")
    row = current['tasks']['blocks']
    lines += ['',f"新组状态：{current['status']}；原执行解出 {row['execution_solved']}，16k 严格合同通过 {row['contract_pass']}；服务失败待补 {row['service_errors']} 题，超出 16k {row['budget_exceeded']} 题。未完成时只报告已有样本，不将未返回题当作错误。",'',
              '## 逐步状态与配对变化','',
              f'预算内答卷的合法动作前缀：完整状态正确 {correct}/{evaluated}，格式有效 {valid}/{evaluated}。缺失状态报告计错；首次非法动作及其后不评价状态。严格合同额外要求所有动作后的棋盘均正确，动作执行成功单独计分。','',
              (f"已返回的 {len(paired)} 道配对题：两组均解出 {transitions[True,True]}；原组解出、新组失败 {transitions[True,False]}；原组失败、新组解出 {transitions[False,True]}；两组均失败 {transitions[False,False]}。" if same_questions else
               '新形状条件重新生成题目，与原题不逐题配对。12 是构造放置次数，参考解保证可解，不表示精确最短长度。'),'',
              '## 解释边界','',
              ('原无状态组请求上限混用 32k／16k，但这 40 份答卷均未超过 16k；新组全部请求 16k。生成请求上限仍存在差异，不能称为完全隔离的单变量实验。' if same_questions else
               '两组均请求 16k，但形状目录和棋盘分布同时变化，表格描述两批题的结果，不作同题因果归因。S/Z 只含表中固定朝向，不额外开放旋转。'),
              '8k 数字是历史答卷的评分敏感性分析，不是在 8k 请求上限下重新生成。状态文本是模型预测，不能覆盖真实环境或用于反馈。','',
              f"新组错误分类：{json.dumps(row['failures'],ensure_ascii=False)}。模型答错、截断和超限均保留，只补服务失败。",'',
              ('[正式配置](../configs/sol56_blocks40_state_report.json)' if same_questions else '[正式配置](../configs/sol56_blocks40_sz_state_report.json)')+' · [原五任务对照](model_retest.md)','']
    out = Path(out)
    out.mkdir(parents=True,exist_ok=True)
    stem = 'blocks_state_report' if same_questions else 'blocks_sz_state_report'
    (out/f'{stem}.md').write_text('\n'.join(lines),encoding='utf-8',newline='\n')
    write_json(out/f'{stem}.json',{'updated_at':now,'same_questions':same_questions,'baseline_blocks':original['tasks']['blocks'],
        'secondary_output_limit':secondary_limit,'baseline_blocks_8k':original8['tasks']['blocks'],
        'new_blocks_8k':current8['tasks']['blocks'],
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
