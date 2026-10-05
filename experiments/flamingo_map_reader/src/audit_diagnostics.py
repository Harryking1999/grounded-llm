"""Read-only acceptance of saved final-map diagnostics; never generate answers."""

from argparse import ArgumentParser
from collections import Counter, defaultdict
from datetime import datetime, timezone, timedelta
import json
from pathlib import Path
import re

import numpy as np
import torch

from .counterfactual import reverse_candidate_q
from .diagnostic_eval import safe_choice, candidate_bin
from .evaluate_blocks import parse_control
from .trajectory_dataset import load_record
from .trajectory_eval import TaskEnvironment
from .trajectory_metrics import score_turn


def completed_cases(group):
    shards = [p for p in sorted(group.glob('*/summary.json'))
              if re.fullmatch(r'\d{5}_\d{5}', p.parent.name)]
    cases = []
    for path in shards:
        value = json.loads(path.read_text())
        cases.extend((json.loads(line), value) for line in
                     (path.parent / 'cases.jsonl').read_text().splitlines() if line.strip())
    return cases, len(shards)


def reference_metrics(rows):
    n = len(rows)
    counts = dict(decisions=n,
        normal_reachable=sum(r['normal']['action_keeps_goal_reachable'] for r in rows),
        reverse_reachable=sum(r['reverse']['action_keeps_goal_reachable'] for r in rows),
        normal_shortest=sum(r['normal']['action_environment_shortest'] for r in rows),
        reverse_shortest=sum(r['reverse']['action_environment_shortest'] for r in rows),
        normal_map_minimum=sum(r['normal']['action_map_minimum'] for r in rows),
        reverse_original_map_minimum=sum(r['reverse']['action_map_minimum'] for r in rows),
        reverse_presented_map_minimum=sum(r['reverse_presented_map_minimum'] for r in rows),
        changed_physical_action=sum(r['changed_physical_action'] for r in rows),
        normal_legal=sum(r['normal']['legal_action'] for r in rows),
        reverse_legal=sum(r['reverse']['legal_action'] for r in rows),
        lost_reachability=sum(r['normal']['action_keeps_goal_reachable'] and not r['reverse']['action_keeps_goal_reachable'] for r in rows),
        gained_reachability=sum(not r['normal']['action_keeps_goal_reachable'] and r['reverse']['action_keeps_goal_reachable'] for r in rows))
    counts['rates'] = {k: v / n if n else None for k, v in counts.items() if k != 'decisions'}
    return counts


def rollout_metrics(cases):
    return dict(cases=len(cases), reached=sum(c['reached_goal'] for c in cases),
        reached_shortest=sum(c['reached_goal'] and c['moves'] == c['shortest_moves'] for c in cases),
        success=sum(c['success'] for c in cases))


def audit(run, output):
    torch.set_num_threads(1)
    protocol = json.loads((output / 'protocol.json').read_text())
    evidence = dict(snapshot_beijing=datetime.now(timezone(timedelta(hours=8))).isoformat(),
                    source_commit=json.loads((output / 'worker_0_launch.json').read_text())['source_commit'],
                    protocol=protocol, reference={}, rollout={}, audit_checks=Counter())
    checks = evidence['audit_checks']
    for worker in (0, 1):
        assert json.loads((output / f'protocol_worker_{worker}.json').read_text()) == protocol
        events = [json.loads(line) for line in (output / f'worker_{worker}_queue.log').read_text().splitlines()
                  if line.startswith('{')]
        barrier = next(i for i, e in enumerate(events) if e.get('event') == 'stage_complete' and e['stage'] == 1)
        assert all(e.get('job', {}).get('kind') != 'safe_prefix_rollout' for e in events[:barrier])
        checks['stage_barriers_verified'] += 1
    manifests = {}
    for task in ('path', 'blocks'):
        manifest_path = run / task / 'data/manifest.json'
        manifest = manifests[task] = json.loads(manifest_path.read_text())
        records = [r for r in manifest['records'] if r['split'] == protocol['split']]
        by_id = {r['trajectory_id']: r for r in records}
        baseline, _ = completed_cases(run / 'evaluation' / task / 'final/validation_reference_map')
        original = {c['trajectory_id']: c for c, _ in baseline}
        assert len(original) == len(records)
        pairs, shards = completed_cases(output / task / 'q_reverse_reference')
        assert len(pairs) == len(records) and {c['trajectory_id'] for c, _ in pairs} == set(by_id)
        rows, strata = [], defaultdict(list)
        expected_decisions = 0
        for case_index, (case, summary) in enumerate(pairs, 1):
            record = by_id[case['trajectory_id']]
            assert case['mode'] == 'reference' and case['kind'] == 'q_reverse_reference' and case['variant'] == 0
            contract = summary['contract']
            assert contract['ready']['epoch'] == protocol['final_epochs'][task]
            assert contract['protocol'] == protocol and contract['kind'] == 'q_reverse_reference'
            demo = load_record(manifest_path.parent / 'trajectories', record, manifest['config'], 0)
            saved = original[case['trajectory_id']]
            assert len(demo.turns) == len(saved['turns'])
            selected = [i for i, r in enumerate(saved['turns']) if not r['done'] and r['remaining_shortest'] >= 0]
            assert [r['turn'] for r in case['turns']] == selected
            assert case['excluded'] == dict(terminal=sum(r['done'] for r in saved['turns']),
                already_dead=sum(not r['done'] and r['remaining_shortest'] < 0 for r in saved['turns']))
            expected_decisions += len(selected)
            for row in case['turns']:
                index = row['turn']
                step = demo.turns[index].step
                altered, sources = reverse_candidate_q(step)
                assert row['source_candidate_ids'] == [i + 1 for i in sources]
                assert list(step.map_minimal_candidates) == row['original_minimal_ids']
                assert list(altered.map_minimal_candidates) == row['reversed_minimal_ids']
                torch.testing.assert_close(altered.map_batch.vectors[:, :2], step.map_batch.vectors[:, :2])
                assert row['current'] == str(step.current) and row['goal'] == str(step.goal)
                assert row['candidates'] == len(step.candidate_actions)
                assert row['normal']['answer'] == saved['turns'][index]['answer']
                for field in ('action_keeps_goal_reachable', 'action_environment_shortest', 'chosen_id', 'legal_action'):
                    assert row['normal'].get(field) == saved['turns'][index].get(field), (task, case['trajectory_id'], index, field)
                for condition in ('normal', 'reverse'):
                    scored = score_turn(step, row[condition]['answer'], reported=manifest['config']['data'].get('reported_candidates'))
                    for field in ('chosen_id', 'legal_action', 'action_map_minimum', 'exact_ranking'):
                        assert row[condition].get(field) == scored.get(field)
                    chosen = scored.get('chosen_id')
                    expected_action = step.candidate_actions[chosen - 1] if scored['legal_action'] else None
                    assert row[condition]['chosen_action'] == expected_action
                scored = score_turn(altered, row['reverse']['answer'], reported=manifest['config']['data'].get('reported_candidates'))
                assert row['reverse_presented_map_minimum'] == scored['action_map_minimum']
                assert row['changed_physical_action'] == (row['normal']['chosen_action'] != row['reverse']['chosen_action'])
                rows.append(row)
                for key in ('goal/' + case['goal_type'], 'turn/' + str(index + 1),
                            'candidates/' + candidate_bin(row['candidates']),
                            'distance_changed/' + str(row['distance_order_changed'])):
                    strata[key].append(row)
                checks['reference_turns_verified'] += 1
            if case_index % 100 == 0:
                print(json.dumps(dict(audited_reference=task, cases=case_index)), flush=True)
        assert len(rows) == expected_decisions
        published = json.loads((output / task / 'q_reverse_reference/summary.json').read_text())['strata']['all']
        counts = reference_metrics(rows)
        assert published['pairs'] == len(rows)
        assert published['normal']['action_keeps_goal_reachable'] == counts['normal_reachable']
        assert published['reverse']['action_keeps_goal_reachable'] == counts['reverse_reachable']
        evidence['reference'][task] = dict(accepted=True, complete=True, cases=len(pairs), shards=shards,
            overall=counts, strata={k: reference_metrics(v) for k, v in strata.items()})
    manifest = manifests['blocks']
    records = {r['trajectory_id']: r for r in manifest['records'] if r['split'] == protocol['split']}
    baseline_pairs, _ = completed_cases(run / 'evaluation/blocks/final/validation_rollout_map')
    baselines = {c['trajectory_id']: c for c, _ in baseline_pairs}
    assert set(baselines) == set(records)
    # Reuse existing saved paths; never execute model generation during acceptance.
    environment = TaskEnvironment(manifest['config'], manifest)
    for helper in protocol['stages'][1]['helper_steps']:
        cases, shards = completed_cases(output / 'blocks' / f'safe_prefix_{helper}_rollout')
        assert len({c['trajectory_id'] for c, _ in cases}) == len(cases)
        verified_cases, assisted_rounds, kept_before_handoff = [], 0, 0
        for case_index, (case, summary) in enumerate(cases, 1):
            record = records[case['trajectory_id']]
            assert case['variant'] == 0 and case['mode'] == 'rollout' and case['helper_steps'] == helper
            assert summary['contract']['ready']['epoch'] == 5 and summary['contract']['protocol'] == protocol
            path = [int(record['start'])]
            goal = int(record['goal'])
            rng = np.random.default_rng(record['sample_seed'] + 104729)
            actual = [int(n) for n in case['actual_path']]
            actions = []
            for index, row in enumerate(case['turns']):
                step = environment.step(path, goal, rng)
                assert path == actual[:index + 1]
                assert row['current'] == str(path[-1]) and row['goal'] == str(goal)
                assert row['assisted'] == (index < helper and not step.done)
                remaining = environment.remaining(step.current, goal, path)
                assert row['remaining_shortest'] == remaining
                if row['assisted']:
                    chosen = safe_choice(environment, step, path)
                    assert row['executed_candidate_id'] == chosen
                    assert row['answer'] == f'<action>{chosen}</action>'
                    assert environment.chosen_keeps_reachable(step, chosen, path, remaining)
                    assisted_rounds += 1
                else:
                    assert row['answer'] == row['model_answer']
                try:
                    chosen = parse_control(row['answer'])
                except ValueError:
                    chosen = None
                if 'action_keeps_goal_reachable' in row:
                    assert row['action_keeps_goal_reachable'] == environment.chosen_keeps_reachable(step, chosen, path, remaining)
                    assert row['action_environment_shortest'] == environment.chosen_is_shortest(step, chosen, path, remaining)
                if len(actual) > index + 1:
                    action, destination = environment.execute(step, chosen)
                    assert action == case['actual_actions'][index] and destination == actual[index + 1]
                    actions.append(action)
                    path.append(destination)
                    if index == helper - 1:
                        assert environment.remaining(destination, goal, path) >= 0
                        kept_before_handoff += 1
                checks['rollout_turns_verified'] += 1
            assert actual == path and case['moves'] == len(actions) == len(case['actual_actions'])
            assert case['reached_goal'] == (goal in path)
            verified_cases.append(case)
            if case_index % 100 == 0:
                print(json.dumps(dict(audited_helper=helper, cases=case_index)), flush=True)
        same = [baselines[c['trajectory_id']] for c in verified_cases]
        strata = {}
        for goal_type in ('empty', 'nonempty', 'initial_goal'):
            selected = [c for c in verified_cases if c['goal_type'] == goal_type]
            strata[goal_type] = dict(assisted=rollout_metrics(selected),
                matched_baseline=rollout_metrics([baselines[c['trajectory_id']] for c in selected]))
        evidence['rollout'][str(helper)] = dict(complete=len(cases) == len(records),
            accepted_completed_shards=bool(cases), completed_cases=len(cases), expected_cases=len(records), completed_shards=shards,
            assisted_rounds_verified=assisted_rounds, solvable_handoffs_verified=kept_before_handoff,
            assisted=rollout_metrics(verified_cases), matched_baseline=rollout_metrics(same), strata=strata)
    evidence['audit_checks'] = dict(checks)
    evidence['acceptance'] = 'reference_complete_rollout_partial' if not all(v['complete'] for v in evidence['rollout'].values()) else 'all_complete'
    evidence['worker_snapshots'] = {}
    for path in output.glob('worker_*_status.json'):
        status = json.loads(path.read_text())
        evidence['worker_snapshots'][path.stem] = dict(stage=status['stage'],
            running=len(status['running']), pending=status.get('pending'), completed=status.get('completed'))
    failures = list(output.glob('worker_*_failed.json'))
    assert not failures, failures
    return evidence


def main():
    parser = ArgumentParser()
    for name in ('run-root', 'diagnostics-root', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    value = audit(args.run_root, args.diagnostics_root)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(value, indent=2) + '\n')
    print(json.dumps(dict(acceptance=value['acceptance'], checks=value['audit_checks'])))


if __name__ == '__main__':
    main()
