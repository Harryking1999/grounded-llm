"""Separate empty and nonempty goals without mixing in zero-move tasks."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

from experiments.flamingo_map_reader.src.analyze_rollout_turns import analyze


def target_kind(record):
    if int(record['start']) == int(record['goal']):
        return 'initial_goal'
    return 'empty' if int(record['goal']) == 0 else 'nonempty'


def main():
    from experiments.flamingo_map_reader.src.collect_evidence import complete_rollout_reachability, load
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    manifest=json.loads((args.run_root/'blocks/data/manifest.json').read_text())
    records={r['trajectory_id']:r for r in manifest['records']}
    result=dict(run=args.run_root.name,data={},validation={})
    train_goals={int(r['goal']) for r in records.values() if r['split']=='train'}
    for split in ['train','validation','test']:
        groups=defaultdict(list)
        for row in records.values():
            if row['split']==split:
                groups[target_kind(row)].append(row)
        result['data'][split]={kind:dict(
            tasks=len(rows),unique_goals=len({int(r['goal']) for r in rows}),
            goal_cells=dict(sorted(Counter(int(r['goal']).bit_count() for r in rows).items())),
            unique_starts=len({r['start'] for r in rows}),
            goals_seen_in_training=sum(int(r['goal']) in train_goals for r in rows),
            groups=dict(Counter(r['group'] for r in rows)),
            greedy_reached=sum(r['greedy_success'] for r in rows),
            shortest_moves=dict(Counter(r['shortest_moves'] for r in rows)))
            for kind,rows in groups.items()}
    expected={r['trajectory_id'] for r in records.values() if r['split']=='validation'}
    for checkpoint in ['checkpoint-147000','checkpoint-196000','final']:
        modes={}
        for mode in ['rollout','reference']:
            cases=load(args.run_root/'evaluation/blocks'/checkpoint/('validation_'+mode+'_map'))
            assert {(c['trajectory_id'],c['variant']) for c in cases}=={(key,0) for key in expected}
            assert len(cases)==len(expected)
            complete_rollout_reachability(cases,manifest)
            groups=defaultdict(list)
            for case in cases:
                record=records[case['trajectory_id']]
                kind=target_kind(record)
                groups[kind].append(case)
                groups[kind+'/'+case['group']].append(case)
            summaries={}
            for kind,members in groups.items():
                full=analyze(members)
                fields=['cases','nonzero_cases','reached','nonzero_reached','reached_shortest','overall',
                        'first_irreversible_dead_end_by_turn']
                if kind in ('empty','nonempty'):
                    fields.append('by_turn')
                summaries[kind]={key:full[key] for key in fields}
            modes[mode]=summaries
        result['validation'][checkpoint]=modes
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(args.out)


if __name__=='__main__':
    main()
