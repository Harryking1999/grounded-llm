"""Publish a compact baseline/current-queue snapshot from saved answers only."""
import argparse
from datetime import datetime, timezone, timedelta
import json
from pathlib import Path

from .report import build, NAMES
from .run import write_json


def conditions(folder):
    suite = json.loads((folder/'suite.json').read_text(encoding='utf-8'))
    records = [json.loads(path.read_text(encoding='utf-8')) for path in (folder/'records').glob('*.json')]
    latest = {row['case_id']:row for row in sorted(records,key=lambda r:(r['case_id'],r['attempt']))}
    groups = {}
    for case in suite['cases']:
        key = case['task'],case.get('phase','screen'),case['stratum']
        row = groups.setdefault(key,{'planned':0,'valid':0,'solved':0,'service_errors':0})
        row['planned'] += 1
        record = latest.get(case['id'])
        if record:
            row['service_errors'] += bool(record.get('service_error'))
            row['valid'] += not bool(record.get('service_error'))
            row['solved'] += bool(record['verdict'].get('solved'))
    failures = [row for row in latest.values() if not row.get('service_error') and not row['verdict'].get('solved')]
    return groups,failures


def label(task,stratum):
    if stratum.startswith('SZ10_board'):
        size = stratum.split('_')[1][5:]
        return f'{size}×{size}，{stratum.rsplit("_",1)[1]} 块，含 S/Z'
    if stratum.startswith('SZ10_connected_'):
        return f'10×10，{stratum.rsplit("_",1)[1]} 块，含 S/Z'
    if task == 'hanoi':
        return f'{int(stratum.rsplit("_",1)[1]).bit_length()} 盘'
    if stratum.startswith('scramble_'):
        size = {'cube3':'三阶','puzzle15':'4×4','puzzle24':'5×5'}[task]
        return f'{size}，打乱 {stratum.rsplit("_",1)[1]} 步'
    if stratum.startswith('nodes_'):
        return f'{stratum.split("_")[1]} 节点'+('树图' if '_tree_' in stratum else '')
    return stratum


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--baseline',required=True,type=Path)
    parser.add_argument('--current',required=True,type=Path)
    parser.add_argument('--qwen',required=True,type=Path)
    parser.add_argument('--history',nargs='*',type=Path,default=[])
    parser.add_argument('--out',required=True,type=Path)
    args = parser.parse_args()
    baseline,_ = build(args.baseline)
    current,_ = build(args.current)
    qwen,_ = build(args.qwen)
    suite = json.loads((args.current/'suite.json').read_text(encoding='utf-8'))
    history = {path.name:build(path)[0] for path in args.history}
    now = datetime.now(timezone(timedelta(hours=8))).strftime('%Y-%m-%d %H:%M（北京时间）')
    lines = ['# 五任务纯文字评测：结果与当前安排','',f'更新时间：{now}。', '',
             '## 当前队列：每档至少 20 题','',
             '按用户最新范围：积木 15×15、20 块、含 S/Z；三阶魔方扩测；华容道试 5×5 后停止加尺寸；11 盘汉诺塔扩样；寻路停止加难。', '',
             f"首批四组共 80 题已冻结并回放参考解，复用 {suite.get('reused_valid_answers',0)} 份完全同题的有效答卷，答错的也保留。有效答卷不足 20 时仅报告进度，服务失败不当作答错。",'',
             '| 任务 | 条件 | 阶段 | 有效／本组题数 | 解出 | 待补服务失败 |',
             '|---|---|---|---:|---:|---:|']
    groups,failures = conditions(args.current)
    for (task,phase,stratum),row in groups.items():
        lines.append(f"| {NAMES[task]} | {label(task,stratum)} | {'独立确认' if phase=='confirm' else '初筛'} | {row['valid']}/{row['planned']} | {row['solved']} | {row['service_errors']} |")
    if failures:
        lines += ['', '**已返回题的具体失败**', '']
        for row in failures:
            verdict = row['verdict']
            index = verdict.get('first_illegal_index')
            detail = (f'第 {index+1} 步非法' if index is not None else
                      '将已验证可解题误判为无解' if verdict.get('reported_status') == 'no_solution' else
                      {'unsolved':'未到目标','parse_error':'答案无法解析','truncated':'输出截断'}.get(verdict.get('failure'),verdict.get('failure')))
            lines.append(f"- {NAMES[row['task']]} `{row['case_id']}`：{detail}。")
    lines += ['',
              '5×5 题目均可解，Manhattan 下界至少 60，因此任何解至少需要 60 步；256 是构造步数上界。积木 20 是构造放置数，允许找到其他合法分解。参考解和距离下界均不提供给模型。', '',
              '每档完成 20 题后再判断；成功不高于一半时另取同条件 20 道独立新题确认。保留所有合法返回，执行错误、未完成、格式和截断分别统计。', '',
              f"当前加难答卷中 {current['exceeded_output_limit']} 份超出请求 token 上限；解出数按真实回放计，不能视为严格预算内成功。", '',
              '## 原始各 40 题：GPT-6.1-sol high','',
              '| 任务 | 有效答卷 | 解出 | 非法动作 | 最优解 |','|---|---:|---:|---:|---:|']
    for task,row in baseline['tasks'].items():
        lines.append(f"| {NAMES[task]} | {row['valid_responses']}/40 | {row['solved']} | {row['failures'].get('illegal_action',0)} | {row['optimal']} |")
    lines += ['', '共 195/200 解出，原题服务失败已补完。积木八形状、12 次放置、10×10；寻路 256 节点、距离 3–10；汉诺塔八盘、距离 1–255；二阶魔方距离 1–14；3×3 华容道距离 4–30。非最短成功解仍计解出。', '',
              '积木两题移除到了空格；汉诺塔两题违反盘子大小规则；魔方一题未复原。基础设施失败未计入这些模型错误。', '',
              f"原题 {baseline['exceeded_output_limit']} 份答卷超出请求预算。接口未返回模型名与 reasoning 配置，名称及 high 仅指请求值；客户端请求不提供工具、无中途反馈，后台身份与强度不能独立核验。", '',
              '## 同题 Qwen3-32B：最后回收的快照','',
              f"028 的 GPU 0–3、完整 BF16、thinking 模式；最近回收时间 {qwen['updated_at']}。此前跳板 SSH 超时，当前远端进度尚未核实。以下是部分题，不能当作完整成功率比较。", '',
              '| 任务 | 有效答卷 | 解出 |','|---|---:|---:|']
    for task,row in qwen['tasks'].items():
        lines.append(f"| {NAMES[task]} | {row['valid_responses']}/40 | {row['solved']} |")
    lines += ['', '## 历史探索与入口','',
              '此前五题一档的探索、28 块积木、4×4 压力题和寻路加难均已停止派新题，旧答卷保留。它们不替代当前每档 20 题；复用答卷也不当作新增独立试验重复计算。', '',
              '[当前配置](../configs/difficulty_20.json) · [执行与复用说明](../README.md) · [原题配置](../configs/sol_high_40.json) · [32B 配置](../configs/qwen32b_same40.json)', '',
              '原始题目、答卷、服务失败与来源均在 Git 外的 runs/；当前运行 runs/sol_difficulty_20/。', '']
    args.out.mkdir(parents=True,exist_ok=True)
    (args.out/'report.md').write_text('\n'.join(lines),encoding='utf-8')
    write_json(args.out/'summary.json',{'updated_at':now,'baseline':baseline,'current':current,
                                      'current_failures':[{'case_id':r['case_id'],'task':r['task'],'verdict':r['verdict']} for r in failures],
                                      'reused_valid_answers':suite.get('reused_valid_answers',0),
                                      'qwen_last_collected':qwen,'retired_runs':history})
    print(json.dumps({'out':str(args.out),'current_valid':sum(r['valid_responses'] for r in current['tasks'].values())}))


if __name__ == '__main__':
    main()
