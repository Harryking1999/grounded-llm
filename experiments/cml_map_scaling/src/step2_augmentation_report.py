"""Merge matched augmentation shards and score explicit semantic decisions."""
import argparse
from collections import Counter
import json
from pathlib import Path
from datetime import datetime

import numpy as np

from .core import shortest_distances
from .run import ROOT, write_json
from .step2_augmentation import score_decision, summarize_actions
from .step2_data import examples


def build(run, base, adjudications=None):
    base_config = json.loads((base / 'config.json').read_text(encoding='utf-8'))
    split = json.loads((base / 'split.json').read_text(encoding='utf-8'))
    graph = json.loads((ROOT / base_config['assets']['graph_reference']).read_text(encoding='utf-8'))
    adjacency = np.zeros((base_config['assets']['node_count'],) * 2, dtype=bool)
    for node, neighbors in graph['neighbors'].items():
        adjacency[int(node), neighbors] = True
    distances = shortest_distances(adjacency)
    manual = {(r['condition'], r['u'], r['g']): r for r in (adjudications or [])}
    rows, runtimes = [], []
    config = None
    for shard in sorted(p for p in run.glob('shard_*') if p.is_dir()):
        runtime = json.loads((shard / 'runtime.json').read_text(encoding='utf-8'))
        assert runtime.get('completed_utc'), 'Wait for both shards to finish'
        runtimes.append(runtime)
        shard_config = json.loads((shard / 'config.json').read_text(encoding='utf-8'))
        if config is not None:
            assert config == shard_config
        config = shard_config
        for line in (shard / 'predictions.jsonl').read_text(encoding='utf-8').splitlines():
            row = json.loads(line)
            key = (row['condition'], row['u'], row['g'])
            original_decision = row['decision']
            if key in manual:
                review = manual[key]
                assert review['evidence'] and review['rationale']
                decision = dict(node=review['node'], source='semantic_review',
                                status='identified' if review['node'] is not None else review['status'])
                row['original_decision'] = original_decision
                row['adjudication'] = review
            else:
                decision = None
            row.update(score_decision(row, row['raw_output'], adjacency, distances, decision))
            rows.append(row)
    assert len(runtimes) == runtimes[0]['num_shards']
    expected_items = examples(split[config['split']], config['tasks'])
    expected = {(c['name'], r['u'], r['g']) for c in config['conditions'] for r in expected_items}
    assert len(rows) == len(expected) and {(r['condition'],r['u'],r['g']) for r in rows} == expected
    assert set(manual).issubset(expected)
    unresolved = [r for r in rows if r['decision']['status']=='needs_review']
    write_json(run / 'semantic_review_pending.json', [{k:r[k] for k in ('condition','u','g','raw_output','hit_token_limit','decision')} for r in unresolved])
    summary = {c['name']:summarize_actions([r for r in rows if r['condition']==c['name']]) for c in config['conditions']}
    reference = config.get('reference_condition', 'text_baseline')
    assert reference in summary
    keyed = {(r['condition'],r['u'],r['g']):r for r in rows}
    paired, failures, review_counts, decision_statuses = {}, {}, {}, {}
    for condition in config['conditions']:
        name = condition['name']
        selected = [r for r in rows if r['condition']==name]
        review_counts[name] = sum(r['decision']['source']=='semantic_review' for r in selected)
        decision_statuses[name] = dict(Counter(r['decision']['status'] for r in selected))
        failures[name] = dict(Counter('success' if r['correct'] else
            'no_unique_decision' if r['decision']['node'] is None else
            'out_of_range' if not r['node_id_valid'] else
            'stays_at_current' if r['parsed']==r['u'] else
            'nonadjacent_goal_jump' if not r['action_legal'] and r['parsed']==r['g'] else
            'other_nonadjacent' if not r['action_legal'] else 'legal_not_closer' for r in selected))
        if name != reference:
            comparison = Counter()
            for r in selected:
                baseline = keyed[reference,r['u'],r['g']]
                comparison['both_success' if r['correct'] and baseline['correct'] else
                           'augmentation_only' if r['correct'] else
                           'baseline_only' if baseline['correct'] else 'both_failure'] += 1
            paired[name] = {k:comparison[k] for k in ('both_success','augmentation_only','baseline_only','both_failure')}
            paired[name]['success_difference_percentage_points'] = 100 * (summary[name]['one_step_success']['rate'] - summary[reference]['one_step_success']['rate'])
    result = dict(conditions=summary, matched_comparison=paired, failure_categories=failures,
                  reference_condition=reference,
                  manual_review_counts=review_counts, decision_statuses=decision_statuses,
                  source_commit=runtimes[0]['source_commit'], base_source_commit=config['base_source_commit'],
                  evaluated_records=len(rows), unresolved_semantic_reviews=len(unresolved),
                  completed_utc=max(r['completed_utc'] for r in runtimes),
                  started_utc=min(r['started_utc'] for r in runtimes),
                  interpretation=config['interpretation'], training=False, roadmap_scope=config['roadmap_scope'])
    result['wall_seconds'] = (datetime.fromisoformat(result['completed_utc'].replace('Z','+00:00')) -
                              datetime.fromisoformat(result['started_utc'].replace('Z','+00:00'))).total_seconds()
    write_json(run/'summary.json', result)
    with (run/'scored_predictions.jsonl').open('w',encoding='utf-8') as stream:
        for r in rows:
            stream.write(json.dumps(r,ensure_ascii=False)+'\n')
    return result


def write_report(run, result):
    def f(metric):
        return f"{metric['successes']}/{metric['total']}"
    lines = ['# Step 2：完整文字输入上的一步动作比较', '',
             result['interpretation'], '',
             '## 匹配测试结果', '',
             '| 条件 | 明确选出动作 | 合法动作 | 一步成功 | 触及输出预算 | 平均生成 token |',
             '| --- | --- | --- | --- | --- | --- |']
    for name, metrics in result['conditions'].items():
        lines.append('| '+' | '.join([name]+[f(metrics[m]) for m in ('decision_identified','action_legal','one_step_success','token_limit')]+[f"{metrics['mean_generated_tokens']:.1f}"])+' |')
    reference = result.get('reference_condition', 'text_baseline')
    lines += ['', '一步成功指选出的节点与当前节点相邻，且到目标的最短距离严格减少；所有正确后继均被接受。全部 100 个有序起终点对进入分母。明确的自然语言决定和裸编号按同一动作计分；输出格式不单独决定成败。', '',
              f'## 与 {reference} 逐题配对', '',
              '| 对照组 | 两组均成功 | 仅对照组成功 | 仅参考组成功 | 两组均失败 | 成功率差（百分点） |',
              '| --- | --- | --- | --- | --- | --- |']
    for name, values in result['matched_comparison'].items():
        lines.append('| '+' | '.join([name]+[str(values[k]) for k in ('both_success','augmentation_only','baseline_only','both_failure')]+[f"{values['success_difference_percentage_points']:+.1f}"])+' |')
    lines += ['', '## 未成功原因', '',
              '| 条件 | 未选定唯一动作 | 非法动作 | 合法但未接近目标 |',
              '| --- | --- | --- | --- |']
    for name, values in result['failure_categories'].items():
        illegal = sum(values.get(k,0) for k in ('out_of_range','stays_at_current','nonadjacent_goal_jump','other_nonadjacent'))
        lines.append(f"| {name} | {values.get('no_unique_decision',0)} | {illegal} | {values.get('legal_not_closer',0)} |")
    lines += ['', '未选定动作与非法动作分别统计；触及输出预算是另一个标签，不覆盖已表达动作的正确或错误。']
    lines += ['', '## 按图距离', '', '| 图距离 | '+' | '.join(result['conditions'])+' |',
              '| --- | '+' | '.join('---' for _ in result['conditions'])+' |']
    distances = result['conditions'][reference]['success_by_graph_distance']
    for d in distances:
        lines.append('| '+' | '.join([d]+[f(result['conditions'][c]['success_by_graph_distance'][d]) for c in result['conditions']])+' |')
    lines += ['', '## 评测口径与来源', '',
              '- 具体输入、checkpoint、模板与生成参数以本次各 shard 的 `config.json` 和 `runtime.json` 为准。',
              '- 输出允许简短说明，提示在结尾明确所选节点。先根据答案文本提取决定，再使用图裁判判分；未识别或多重决定必须语义审阅，不能挑选其中正确的候选。预算耗尽单独记录，有明确决定的触顶输出仍可判分。',
              f"- 总计 {result['evaluated_records']} 条记录，尚待语义审阅 {result['unresolved_semantic_reviews']} 条。逐题原文保存在各 shard 的 `predictions.jsonl`，合并后的评分保存在 `scored_predictions.jsonl`。",
              f"- 已提供语义审阅共 {sum(result['manual_review_counts'].values())} 条；原文依据和审阅记录随运行保存。若尚有待审阅输出，当前结果只作暂定摘要。",
              f"- 推理源码 `{result['source_commit'][:7]}`；基础训练源码 `{result['base_source_commit'][:7]}`。墙钟耗时 {result['wall_seconds']/60:.1f} 分钟。",
              '- 固定图和已查看过的测试节点组合只支持探索性比较；输出完成度与已选动作的方向正确性需分别解释。', '']
    (run/'report.md').write_text('\n'.join(lines),encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True)
    parser.add_argument('--base-run', required=True)
    parser.add_argument('--adjudications')
    args = parser.parse_args()
    adjudications = json.loads(Path(args.adjudications).read_text(encoding='utf-8')) if args.adjudications else None
    result = build(Path(args.run),Path(args.base_run),adjudications)
    write_report(Path(args.run),result)
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
