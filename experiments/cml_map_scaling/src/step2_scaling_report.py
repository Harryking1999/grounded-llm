"""Merge scaling results and retain semantic-review cases before conclusions."""
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np

from .core import shortest_distances
from .run import write_json
from .step2_augmentation import score_decision, summarize_actions
from .step2_data import examples


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def build(root,adjudications):
    contract=read(root/'scaling_config.json')
    manual={(r['case'],r['condition'],r['u'],r['g']):r for r in adjudications}
    all_rows,pending,results=[],[],{}
    expected_keys=set()
    for case in contract['assets']['cases']:
        folder=root/case['case']
        graph=read(folder/'graph.json')
        adj=np.zeros((case['node_count'],)*2,dtype=bool)
        for node,neighbors in graph['neighbors'].items():
            adj[int(node),neighbors]=True
        distances=shortest_distances(adj)
        rows=[]
        interfaces={}
        for interface in contract['training']['conditions']:
            evaluation=folder/('eval_'+interface)
            if not (evaluation/'completed.json').exists():
                raise RuntimeError(f'Wait for {case["case"]} {interface} evaluation to finish')
            interfaces[interface]=dict(training=read(folder/interface/'training_summary.json'),
                                        reports=read(evaluation/'summary.json')['reports'],
                                        geometry=read(evaluation/'summary.json')['geometry'])
            for line in (evaluation/'predictions.jsonl').read_text(encoding='utf-8').splitlines():
                row=json.loads(line)
                row['case']=case['case']
                key=(case['case'],row['condition'],row['u'],row['g'])
                decision=None
                if key in manual:
                    review=manual[key]
                    assert review['evidence'] in row['raw_output'] and review['rationale']
                    decision=dict(node=review['node'],source='semantic_review',
                                  status='identified' if review['node'] is not None else review['status'])
                    row['original_decision']=row['decision']
                    row['adjudication']=review
                row.update(score_decision(row,row['raw_output'],adj,distances,decision))
                rows.append(row)
                if row['decision']['status']=='needs_review':
                    pending.append({k:row[k] for k in ('case','condition','u','g','raw_output','hit_token_limit','decision')})
        items=examples(read(folder/'split.json')['test'],('action',))
        expected={(case['case'],condition,item['u'],item['g'])
                  for condition in contract['evaluation']['conditions'] for item in items}
        observed={(r['case'],r['condition'],r['u'],r['g']) for r in rows}
        assert len(rows)==len(expected) and expected==observed
        expected_keys.update(expected)
        conditions={name:summarize_actions([r for r in rows if r['condition']==name])
                    for name in contract['evaluation']['conditions']}
        baseline={(r['u'],r['g']):r for r in rows if r['condition']=='text_baseline'}
        paired,failures={},{}
        for name in conditions:
            selected=[r for r in rows if r['condition']==name]
            failures[name]=dict(Counter('success' if r['correct'] else 'no_unique_decision' if r['decision']['node'] is None
                                        else 'illegal_action' if not r['action_legal'] else 'legal_not_closer' for r in selected))
            if name!='text_baseline':
                counts=Counter('both_success' if r['correct'] and baseline[r['u'],r['g']]['correct'] else
                               'augmentation_only' if r['correct'] else 'baseline_only' if baseline[r['u'],r['g']]['correct'] else 'both_failure'
                               for r in selected)
                paired[name]={k:counts[k] for k in ('both_success','augmentation_only','baseline_only','both_failure')}
                paired[name]['difference_percentage_points']=100*(conditions[name]['one_step_success']['rate']-conditions['text_baseline']['one_step_success']['rate'])
        results[case['case']]=dict(node_count=case['node_count'],conditions=conditions,matched_comparison=paired,
                                   failure_categories=failures,interfaces=interfaces,runtime=read(folder/'runtime.json'))
        all_rows.extend(rows)
    assert set(manual).issubset(expected_keys)
    summary=dict(cases=results,records=len(all_rows),semantic_review_pending=len(pending),
                 adjudicated_records=len(manual),interpretation=contract['interpretation'])
    write_json(root/'summary.json',summary)
    write_json(root/'semantic_review_pending.json',pending)
    with (root/'scored_predictions.jsonl').open('w',encoding='utf-8') as stream:
        for row in all_rows:
            stream.write(json.dumps(row,ensure_ascii=False)+'\n')
    lines=['# Step 2 扩规模：完整文字＋roadmap','','训练仅使用状态报告；动作比较保留完整文字图与当前／目标 ID。',
           '',f'记录 {len(all_rows)} 条，待语义审阅 {len(pending)} 条；待审阅未清零时，以下仅为暂定分数。','',
           '| 节点 | 条件 | 报告两项均正确 | 一步成功 | 合法动作 | 触顶 |',
           '| --- | --- | --- | --- | --- | --- |']
    def count(metric):
        return f"{metric['successes']}/{metric['total']}"
    for result in results.values():
        for name,metrics in result['conditions'].items():
            adapter=name.removeprefix('text_plus_')+'_report'
            report=count(result['interfaces'][adapter]['reports']['both_reports_correct_rate']) if name!='text_baseline' else '—'
            lines.append('| '+' | '.join([str(result['node_count']),name,report]+[count(metrics[m]) for m in ('one_step_success','action_legal','token_limit')])+' |')
    lines+=['','每规模的配对收益、失败类型、分距离结果、训练选模和几何统计见 summary.json。',
            '每个规模仅使用一张已有 seed 0 地图；训练样本按节点数增长，不能把规模间差异解释为纯节点数效应。',
            '冻结文字前缀缓存通过 FP32 损失／梯度等价性检查；BF16 实现存在数值舍入差异，不声称与 32 节点旧实现逐位等价。','']
    (root/'report.md').write_text('\n'.join(lines),encoding='utf-8')
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--adjudications',type=Path)
    args=parser.parse_args()
    summary=build(args.run,read(args.adjudications) if args.adjudications else [])
    print(json.dumps(dict(records=summary['records'],pending=summary['semantic_review_pending'])))


if __name__=='__main__':
    main()
