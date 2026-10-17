"""Publish original-condition results with retired difficulty runs as supplements."""
import argparse
from datetime import datetime, timezone, timedelta
import json
from pathlib import Path

from .report import build, NAMES
from .run import write_json
from .budget import output_limit


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


def render_report(baseline,qwen):
    if (baseline['tasks'].keys() != qwen['tasks'].keys() or
            any(row['valid_responses'] != row['planned']
                for result in (baseline,qwen) for row in result['tasks'].values()) or
            any(row['planned'] != baseline['tasks'][task]['planned']
                for task,row in qwen['tasks'].items())):
        raise ValueError('The main report requires complete, matching baseline groups')
    total = sum(row['planned'] for row in qwen['tasks'].values())
    qwen_solved = sum(row['solved'] for row in qwen['tasks'].values())
    gpt_solved = sum(row['solved'] for row in baseline['tasks'].values())
    settings = {'blocks':'10×10，8 种形状，12 次放置',
                'path':'256 节点三度图，最短 3–10 步',
                'hanoi':'8 盘，最短 1–255 步',
                'cube2':'2×2，最短 1–14 个四分之一转',
                'puzzle8':'3×3，最短 4–30 步'}
    lines = ['# 五任务基线评测','',
             '**采用最初配置。最低难度要求改为 32B 不能可靠完成；GPT 作为强基线参照，不再要求稳定难住 GPT。**','',
             '## 同题主结果（16k 评分）','',
             '统一上限为 16,000 输出 token，包含思考与最终答案；超限即错。以下从已保存答卷重算，不是重新在 16k 请求上限下生成。','',
             '| 任务 | Qwen3-32B 成功 | GPT 成功 | GPT 最优解 |',
             '|---|---:|---:|---:|']
    for task,row in qwen['tasks'].items():
        gpt = baseline['tasks'][task]
        count = row['planned']
        def cell(value):
            return f'{value}/{count}（{100*value/count:.1f}%）'
        lines.append(f"| {NAMES[task]} | {cell(row['solved'])} | {cell(gpt['solved'])} | {cell(gpt['optimal'])} |")
    lines += ['',f"**预算内成功：32B {qwen_solved}/{total}（{100*qwen_solved/total:.1f}%）；GPT {gpt_solved}/{total}（{100*gpt_solved/total:.1f}%）。** 两组各任务 40 题，均已完成并回收。",'',
              f"原执行成功分别为 32B {sum(r['execution_solved'] for r in qwen['tasks'].values())}/{total}、GPT {sum(r['execution_solved'] for r in baseline['tasks'].values())}/{total}；该历史口径保留，当前主指标采用 16k。",'',
              '## 任务设置','', '| 任务 | 采用的初始条件 |','|---|---|']
    for task in qwen['tasks']:
        lines.append(f'| {NAMES[task]} | {settings[task]} |')
    lines += ['',
              '## 当前判断','',
              '- 初始条件已满足本轮筛选要求；32B 表现不足，寻路保留作较简单的对照。',
              '- 同时比较预算内成功率与最优解率，避免把无限延长思考或计划的结果计作预算内成功。',
              '- 这些结果用于确定任务条件；地图是否有额外收益，仍由匹配的地图／无地图训练与评测验证。','',
              '## 评测口径','',
              '两组使用相同题目、规则与提示词，一次输出完整计划，无中途反馈，客户端不提供工具。成功按环境回放到达目标判定；非最短的成功解照计成功，最优解率以全部 40 题为分母。服务失败已补测，正常错误答案不重抽。积木 12 指构造放置次数，不要求还原原来的 12 块。','',
              'Qwen3-32B 使用完整 BF16 权重并开启 thinking；GPT 请求模型为 gpt-6.1-sol，reasoning effort 为 high。两者思考模式不等同。', '']
    lines.append(f"超出 16k 的答卷：32B {qwen['budget_policy']['exceeded']} 份、GPT {baseline['budget_policy']['exceeded']} 份，均计失败；用量缺失不计预算内成功。格式问题仍单列，成功要求实际完成且未超限。")
    metadata = baseline['response_metadata_coverage']
    lines += ['',f"GPT 的 {metadata['effort_counts'].get('high',0)}/200 份答卷回显模型名及 high，其余未回传强度；{baseline['exceeded_output_limit']} 份答卷超过请求 token 上限。因此报告为第三方接口下的实测结果，不声称后台配置独立核验或严格等预算。",'',
              '## 配置与补充','',
              '[初始题集配置](../configs/sol_high_40.json) · [32B 推理配置](../configs/qwen32b_same40.json) · [复现说明](../README.md)', '',
              '新增 [5.6 同题五任务与 6.1 旧十二块题复测](model_retest.md)单独呈现，主表保留上述完整结果。','',
              '此前加难探索已停止派新题，已有答卷保留为[补充结果](difficulty_exploration.md)，不混入主表、不再要求补满原加难计划。原始题目、答案和服务记录保存在 Git 外的 runs/。','']
    return '\n'.join(lines)


def render_supplement(folder,result):
    progress = json.loads((folder/'progress.json').read_text(encoding='utf-8'))
    groups,_ = conditions(folder)
    inflight = len(progress.get('inflight',[]))
    status = (f'已停止派新题，尚有 {inflight} 条已发出请求等待收尾。' if inflight else
              '已停止，未完成题不再补测。')
    lines = ['# 加难探索：补充结果','',
             '用户已决定以初始配置下 32B 表现不足为最低要求，本轮加难不再作为当前目标。以下为已返回样本，包含复用答卷；未完成组不当作完整成功率，不与主结果合并。','',
             status,'','| 任务 | 探索条件 | 阶段 | 有效／原计划 | 解出 | 服务失败未补 |',
             '|---|---|---|---:|---:|---:|']
    for (task,phase,stratum),row in groups.items():
        lines.append(f"| {NAMES[task]} | {label(task,stratum)} | {'确认' if phase=='confirm' else '初筛'} | {row['valid']}/{row['planned']} | {row['solved']} | {row['service_errors']} |")
    lines += ['',f"其中 {result['exceeded_output_limit']} 份答卷超过请求输出预算。",'',
              '15×15 积木使用含 S/Z 十形状、20 次放置；5×5 华容道的 Manhattan 下界至少 60，参考解可回放。构造／打乱长度不代表精确最短距离。更早的五题初筛、28 块积木和长路径探索也已停用，原始结果保留，不再扩展。','',
              '[返回主报告](report.md)','']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--baseline',required=True,type=Path)
    parser.add_argument('--qwen',required=True,type=Path)
    parser.add_argument('--supplementary','--current',dest='supplementary',type=Path)
    parser.add_argument('--history',nargs='*',type=Path,default=[])
    parser.add_argument('--out',required=True,type=Path)
    args = parser.parse_args()
    limit = output_limit()
    baseline,_ = build(args.baseline,limit)
    qwen,_ = build(args.qwen,limit)
    history = {path.name:build(path)[0] for path in args.history}
    supplemental = build(args.supplementary)[0] if args.supplementary else None
    args.out.mkdir(parents=True,exist_ok=True)
    (args.out/'report.md').write_text(render_report(baseline,qwen),encoding='utf-8',newline='\n')
    if supplemental:
        (args.out/'difficulty_exploration.md').write_text(render_supplement(args.supplementary,supplemental),encoding='utf-8',newline='\n')
    write_json(args.out/'summary.json',{'updated_at':datetime.now(timezone(timedelta(hours=8))).isoformat(),
        'decision':'Retain initial conditions; Qwen3-32B unreliability is the minimum difficulty criterion; GPT is a strong reference, not a required failure target.',
        'main_config':'experiments/task_text_baselines/configs/sol_high_40.json',
        'scoring_config':'experiments/task_text_baselines/configs/budget_16k.json',
        'baseline':baseline,'qwen':qwen,'supplementary_difficulty':supplemental,'retired_runs':history})
    print(json.dumps({'out':str(args.out),'qwen_status':qwen['status'],'supplementary_status':supplemental['status'] if supplemental else None}))


if __name__ == '__main__':
    main()
