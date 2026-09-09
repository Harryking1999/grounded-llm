"""Compare completed model batches; never turn partial batches into final accuracy."""
import argparse
from collections import Counter
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--runs', nargs='+', required=True)
    p.add_argument('--baseline-summary', required=True)
    p.add_argument('--out', required=True)
    args = p.parse_args()
    models = {}
    suites = []
    run_paths = {}
    for directory in args.runs:
        root = Path(directory)
        summary = json.loads((root / 'summary.json').read_text())
        run = json.loads((root / 'run.json').read_text())
        if run['status'] != 'completed' or len(run['cases']) != 384:
            raise ValueError(f'Batch is incomplete: {directory}')
        suite = json.loads((root / 'suite.json').read_text())
        suites.append(suite)
        expected = {(c['id'], n) for c in suite['cases'] for n in range(1, c['replicates'] + 1)}
        slots = [(r['case_id'], r['replicate']) for r in run['cases']]
        if len(set(slots)) != len(slots) or set(slots) != expected:
            raise ValueError('Batch does not cover the fixed sample slots exactly')
        model = run['api_config']['model']
        models[model] = summary
        run_paths[model] = str(root)
    if len(models) != 3 or any(s['cases'] != suites[0]['cases'] for s in suites):
        raise ValueError('Need all three models on identical cases')
    baseline = json.loads(Path(args.baseline_summary).read_text())
    comparisons = {'gpt-5.6-sol (gateway)': baseline, **models}
    lines = ['# Qwen thinking 匹配基线结果', '',
        '固定 Sol 的 48 个输入、英文完整计划提示词和裁判，每题八次。三种 Qwen 均开启 thinking；'
        '截断计入准确率分母。GPT 是已有网关返回的 Sol 结果。', '',
        '| 模型 | 条件 | 成功 / 尝试 | 准确率 | pass@8 | 截断 | 非法完整答案 |',
        '| --- | --- | ---: | ---: | ---: | ---: | ---: |']
    compact = {}
    for model, summary in comparisons.items():
        compact[model] = {}
        for condition, s in summary['conditions'].items():
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
        for condition, s in summary['conditions'].items():
            lines += [f"- {condition}：失败分类 `{json.dumps(s['failure_counts'], ensure_ascii=False)}`；"
                      f"0/8 实例 `{s['zero_of_8_cases']}`；完整合同通过 {s['contract_successes']}/{s['final_samples']}。"]
            if 'state_reports' in s:
                state = s['state_reports']
                lines += [f"  合法前缀逐步棋盘完全正确 {state['correct']}/{state['evaluated_legal_prefix_steps']}。"]
        lines.append('')
    lines += ['## 解释边界', '',
        '本轮测量原模型在固定完整计划任务上的可靠性，不检验状态增强模块的因果增益。'
        '任务失败需区分格式、非法动作、状态报告与预算截断；不能仅凭准确率归因于缺乏认知地图。', '',
        'Qwen 同代官方权重、采样参数和输出上限匹配；GPT medium 与 Qwen thinking 不等价，'
        'GPT 网关也没有执行可核验的 token 上限，所以跨系列差值不是严格计算量匹配的结果。'
        '样本每题重复八次，不能当作 128 个独立布局；pass@8 是固定八次采样的经验覆盖率。', '',
        '状态报告准确率仅覆盖可执行的合法前缀，非法动作之后与截断输出没有纳入该指标。'
        '保存的 thinking 文本可供后续诊断，但不是内部机制的直接证据。', '']
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'comparison.json').write_text(json.dumps(compact, ensure_ascii=False, indent=2) + '\n')
    (out / 'report.md').write_text('\n'.join(lines), encoding='utf-8')


if __name__ == '__main__':
    main()
