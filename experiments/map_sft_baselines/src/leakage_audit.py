"""Audit physical SFT exposure separately from intended shared-board evaluation."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path


def train_path(row):
    pairs=[(int(s),int(g)) for s,g in row['pairs']]
    start,goal=int(row['start']),int(row['goal'])
    if any(g!=goal for _,g in pairs) or (pairs and pairs[0][0]!=start):
        raise ValueError('Training decision metadata is inconsistent')
    if len(pairs)!=len(row['route_ids']):
        raise ValueError('Training route and state counts differ')
    return tuple(s for s,_ in pairs)+(goal,)


def reference_path(row):
    import grounded_llm.tasks.blocks.rules as rules
    from grounded_llm.tasks.blocks.protocol import action_from_key
    states=[int(row['start'])]
    for key in row['reference_actions']:
        states.append(rules.apply(states[-1],action_from_key(key)))
    if states[-1]!=int(row['goal']):
        raise ValueError('Reference does not reach its goal')
    return tuple(states)


def audit_exposure(rows,references):
    import grounded_llm.tasks.blocks.rules as rules
    initial={(int(r['start']),int(r['goal'])) for r in rows}
    pairs={(int(s),int(g)) for r in rows for s,g in r['pairs']}
    terminal={(int(r['goal']),int(r['goal'])) for r in rows}
    supervised=pairs|terminal
    states={s for s,_ in pairs}|{int(r['goal']) for r in rows}
    paths={(int(r['goal']),train_path(r)) for r in rows}
    boards={r['board_row'] for r in rows}
    by_group=defaultdict(list)
    for record in references:
        path=reference_path(record);goal=int(record['goal'])
        decision_pairs=[(s,goal) for s in path[:-1]]
        exposed=[pair for pair in decision_pairs if pair in supervised]
        # An unchosen successor may have appeared in the map/ranking input even
        # when it was never a teacher-forced current state. Check without building
        # every training candidate slot in memory.
        candidate_exposed=[]
        for state,g in decision_pairs:
            if any(not (tile & state) and (state|tile,g) in pairs
                   for tile,_ in rules.PLACEMENTS):
                candidate_exposed.append((state,g))
        by_group[record.get('group','unspecified')].append(dict(
            trajectory_id=record['trajectory_id'],
            source_board_seen=record['board_row'] in boards,
            initial_pair_in_training_initial=(path[0],goal) in initial,
            initial_pair_in_training_supervision=(path[0],goal) in supervised,
            reference_decisions=len(decision_pairs),supervised_pair_matches=len(exposed),
            candidate_slot_pair_matches=len(candidate_exposed),
            exact_physical_route_seen=(goal,path) in paths,
            current_state_seen=sum(s in states for s in path[:-1]),
            terminal_pair_seen=(goal,goal) in terminal,
            overlapping_supervised_pairs=[[str(s),str(g)] for s,g in exposed[:5]]))
    groups={}
    for group,cases in by_group.items():
        groups[group]=dict(tasks=len(cases),
            **{key:sum(int(r[key]) for r in cases) for key in (
                'source_board_seen','initial_pair_in_training_initial',
                'initial_pair_in_training_supervision','reference_decisions',
                'supervised_pair_matches','candidate_slot_pair_matches',
                'exact_physical_route_seen','current_state_seen','terminal_pair_seen')},
            tasks_with_supervised_pair_match=sum(r['supervised_pair_matches']>0 for r in cases),
            tasks_with_candidate_slot_match=sum(r['candidate_slot_pair_matches']>0 for r in cases))
    return dict(training=dict(trajectories=len(rows),boards=len(boards),
                    unique_initial_pairs=len(initial),unique_nonterminal_pairs=len(pairs),
                    unique_terminal_pairs=len(terminal)),groups=groups,
                cases=[row for cases in by_group.values() for row in cases],
                interpretation=dict(
                    same_board='Shared source boards are intentional for same_initial_board_new_pair.',
                    state_only='State-only and terminal equality do not establish answer leakage.',
                    candidate_slots='Candidate exposure includes map inputs/ranking supervision, not a current-state action label.',
                    qmap='This audit covers SFT. Frozen-Q training exposure requires a separate data-provenance check.',
                    selection='The fixed cohort has been used to choose checkpoints and methods; it is not an untouched final test.'))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--training-data',required=True,type=Path)
    parser.add_argument('--evaluation-data',required=True,type=Path)
    parser.add_argument('--out',required=True,type=Path)
    parser.add_argument('--cache-samples',type=int,default=64)
    args=parser.parse_args()
    meta=json.loads((args.training_data/'dataset.json').read_text())
    refs=json.loads((args.evaluation_data/'evaluation.json').read_text())
    report=audit_exposure(meta['rows'],refs)
    if args.cache_samples:
        import random
        import torch
        chosen=random.Random(20261019).sample(meta['rows'],min(args.cache_samples,len(meta['rows'])))
        actions=0
        for row in chosen:
            item=torch.load(args.training_data/'trajectories'/row['prepared_file'],map_location='cpu',weights_only=False)['map']
            expected=train_path(row)
            if len(item['steps'])!=len(expected):raise ValueError('Cached route length differs from metadata')
            for i,step in enumerate(item['steps']):
                if step.current!=expected[i] or step.goal!=int(row['goal']):
                    raise ValueError('Cached map states differ from metadata')
                if i<len(row['route_ids']):
                    if step.execute(row['route_ids'][i])[1]!=expected[i+1]:
                        raise ValueError('Cached action differs from metadata')
                    actions+=1
        report['cache_replay']=dict(trajectories=len(chosen),actions=actions,passed=True)
    from grounded_llm.tasks.blocks.sft import atomic_json
    args.out.mkdir(parents=True,exist_ok=True)
    atomic_json(args.out/'exposure.json',report)
    compact={k:v for k,v in report.items() if k!='cases'}
    print(json.dumps(compact),flush=True)


if __name__=='__main__':main()
