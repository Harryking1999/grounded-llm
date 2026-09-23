"""Compare completed model batches; never turn partial batches into final accuracy."""
import argparse
from collections import Counter
import json
from pathlib import Path


def visible_output_diagnostics(records, condition):
    """Summarize saved model-visible behavior without reproducing chain of thought."""
    selected = [r for r in records if r.get('condition') == condition]
    failure_counts = Counter(
        r.get('verdict', {}).get('failure_type') or 'success' for r in selected)
    parse_modes = Counter(
        r.get('verdict', {}).get('content_parse_mode') or 'none' for r in selected)
    return {
        'samples': len(selected),
        'visible_thinking_text': sum(bool(r.get('reasoning_text')) for r in selected),
        'natural_completions': sum(r.get('response_status') == 'completed' for r in selected),
        'reached_goal_but_non_shortest': sum(
            r.get('verdict', {}).get('failure_type') == 'non_shortest'
            and r.get('verdict', {}).get('execution_pass') for r in selected),
        'illegal_routes': failure_counts['illegal_move'],
        'parse_errors': failure_counts['parse_error'],
        'budget_truncations': failure_counts['budget_truncated'],
        'content_parse_modes': dict(parse_modes),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--runs', nargs='+', required=True)
    p.add_argument('--baseline-summary',
                   help='Optional non-Qwen comparison summary on the same conditions.')
    p.add_argument('--conditions', nargs='+',
                   help='Conditions to include; defaults to all conditions in the first Qwen summary.')
    p.add_argument('--out', required=True)
    p.add_argument('--title', default='Qwen thinking 结果')
    p.add_argument('--visible-output-diagnostics', action='store_true',
                   help='Summarize saved output behavior without exposing reasoning text.')
    args = p.parse_args()
    models = {}
    suites = []
    run_paths = {}
    records_by_model = {}
    for directory in args.runs:
        root = Path(directory)
        summary = json.loads((root / 'summary.json').read_text())
        run = json.loads((root / 'run.json').read_text())
        suite = json.loads((root / 'suite.json').read_text())
        suites.append(suite)
        expected = {(c['id'], n) for c in suite['cases'] for n in range(1, c['replicates'] + 1)}
        if run['status'] != 'completed' or len(run['cases']) != len(expected):
            raise ValueError(f'Batch is incomplete: {directory}; expected {len(expected)} slots')
        if run.get('service_errors'):
            raise ValueError(f'Batch has service errors: {directory}')
        slots = [(r['case_id'], r['replicate']) for r in run['cases']]
        if len(set(slots)) != len(slots) or set(slots) != expected:
            raise ValueError('Batch does not cover the fixed sample slots exactly')
        model = run['api_config']['model']
        models[model] = summary
        run_paths[model] = str(root)
        records_by_model[model] = run['cases']
    if len(models) != 3 or any(s['cases'] != suites[0]['cases'] for s in suites):
        raise ValueError('Need all three models on identical cases')
    baseline = (json.loads(Path(args.baseline_summary).read_text())
                if args.baseline_summary else None)
    conditions = args.conditions or list(next(iter(models.values()))['conditions'])
    for condition in conditions:
        if baseline and condition not in baseline['conditions']:
            raise ValueError(f'Baseline has no {condition} condition')
        if any(condition not in summary['conditions'] for summary in models.values()):
            raise ValueError(f'A Qwen batch has no {condition} condition')
    comparisons = ({'gpt-5.6-sol (gateway)': baseline, **models}
                   if baseline else models)
    lines = [f'# {args.title}', '',
        f'固定输入、英文完整计划提示词和裁判，每题八次；本报告包含 {", ".join(conditions)}。'
        '三种 Qwen 均开启 thinking，截断计入准确率分母；Qwen 主指标按内容判分，JSON 合同仅保留为诊断。', '']
    if baseline:
        lines[-1] = 'GPT 是已有网关返回的 Sol 结果。'
        lines.append('')
    lines += [
        '| 模型 | 条件 | 成功 / 尝试 | 准确率 | pass@8 | 截断 | 非法完整答案 |',
        '| --- | --- | ---: | ---: | ---: | ---: | ---: |']
    compact = {}
    for model, summary in comparisons.items():
        compact[model] = {}
        for condition in conditions:
            s = summary['conditions'][condition]
            lines.append(f"| {model} | {condition} | {s['successes']}/{s['final_samples']} | "
                         f"{s['sample_success_rate']:.1%} | {s['pass8_solved_cases']}/{s['pass8_eligible_cases']} | "
                         f"{s['budget_truncations']} | {s['illegal_answers']} |")
            fields = ('final_samples', 'successes', 'sample_success_rate', 'pass_at_8',
                      'budget_truncations', 'illegal_answers', 'failure_counts', 'zero_of_8_cases',
                      'contract_successes', 'execution_successes', 'strict_json_responses',
                      'output_tokens', 'state_reports', 'per_case')
            compact[model][condition] = {k: s[k] for k in fields if k in s}
    lines += ['', '## 失败与状态报告', '']
    for model, summary in models.items():
        lines += [f'### {model}', '', f'原始运行：`{run_paths[model]}`。', '']
        for condition in conditions:
            s = summary['conditions'][condition]
            lines += [f"- {condition}：失败分类 `{json.dumps(s['failure_counts'], ensure_ascii=False)}`；"
                      f"0/8 实例 `{s['zero_of_8_cases']}`；严格 JSON 合同（仅诊断）"
                      f"{s['contract_successes']}/{s['final_samples']}。"]
            if 'state_reports' in s:
                state = s['state_reports']
                lines += [f"  合法前缀逐步棋盘完全正确 {state['correct']}/{state['evaluated_legal_prefix_steps']}。"]
        lines.append('')
    if args.visible_output_diagnostics:
        lines += ['## 可见输出行为诊断', '',
                  '以下是保存的模型输出及裁判结果的有限统计，不复现推理文本，也不能直接证明内部机制。', '']
        for model, records in records_by_model.items():
            lines += [f'### {model}', '']
            for condition in conditions:
                d = visible_output_diagnostics(records, condition)
                lines += [
                    f"- {condition}：保存 reasoning 文本 {d['visible_thinking_text']}/{d['samples']}；"
                    f"自然完成 {d['natural_completions']}/{d['samples']}；"
                    f"到达目标但非最短 {d['reached_goal_but_non_shortest']}；"
                    f"非法路线 {d['illegal_routes']}；解析失败 {d['parse_errors']}；"
                    f"预算截断 {d['budget_truncations']}；"
                    f"最终内容提取方式 `{json.dumps(d['content_parse_modes'], ensure_ascii=False)}`。"
                ]
            lines.append('')
    lines += ['## 解释边界', '',
        '本轮测量原模型在固定完整计划任务上的可靠性，不检验状态增强模块的因果增益。'
        '任务失败需区分格式、非法动作、状态报告与预算截断；不能仅凭准确率归因于缺乏认知地图。', '',
        'Qwen 同代官方权重、采样参数和输出上限匹配。样本每题重复八次，不能当作 128 个独立布局；'
        'pass@8 是固定八次采样的经验覆盖率。', '',
        '状态报告准确率仅覆盖可执行的合法前缀，非法动作之后与截断输出没有纳入该指标。'
        '保存的 thinking 文本可供后续诊断，但不是内部机制的直接证据。', '']
    if baseline:
        lines.insert(-4, 'GPT medium 与 Qwen thinking 不等价，GPT 网关也没有执行可核验的 token 上限，'
                     '所以跨系列差值不是严格计算量匹配的结果。')
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'comparison.json').write_text(json.dumps(compact, ensure_ascii=False, indent=2) + '\n')
    (out / 'report.md').write_text('\n'.join(lines), encoding='utf-8')


if __name__ == '__main__':
    main()
