"""Final validation decisions and physical progress, from saved cases only."""

from argparse import ArgumentParser
from collections import Counter
import json
from pathlib import Path

import torch

from .analyze_rollout_turns import analyze
from .collect_evidence import complete_rollout_reachability, load
from .trajectory_dataset import load_record
from .trajectory_eval import TaskEnvironment


def cumulative(events, denominator, horizon):
    return [dict(step=step, denominator=denominator,
                 reached=sum(count for at, count in events['reached'].items() if at <= step),
                 dead=sum(count for at, count in events['dead'].items() if at <= step))
            for step in range(horizon + 1)]


def collect(run):
    result = dict(run=run.name, split='validation', variant=0, tasks={}, definitions=dict(
        actions='One-based decision step; only solvable nonterminal states; invalid choices count as errors.',
        progress='Executed actions, starting at zero; fixed cohort excludes start=goal.',
        reference='LLM answers on the saved map-greedy history; its answers are not executed.',
        greedy='Actual saved map-greedy trajectory, not the LLM reference answers.',
        dead='First executed action that makes the goal unreachable; invalid outputs are not physical dead ends.'))
    for task in ('path', 'blocks'):
        data_dir = run / task / 'data'
        manifest = json.loads((data_dir / 'manifest.json').read_text())
        records = [r for r in manifest['records'] if r['split'] == 'validation']
        expected = {(r['trajectory_id'], 0) for r in records}
        nonzero = [r for r in records if int(r['start']) != int(r['goal'])]
        ready = json.loads((run / task / 'training/models/final/evaluation_ready.json').read_text())
        value = dict(epoch=ready['epoch'], cases=len(records), nonzero_cases=len(nonzero), modes={})
        cases_by_mode = {}
        for mode in ('reference', 'rollout'):
            directory = run / 'evaluation' / task / 'final' / f'validation_{mode}_map'
            cases = load(directory)
            identities = [(c['trajectory_id'], c['variant']) for c in cases]
            assert len(identities) == len(expected) and set(identities) == expected
            assert all(c['mode'] == mode and not c['no_map'] for c in cases)
            restored = complete_rollout_reachability(cases, manifest)
            counted = analyze(cases)
            value['modes'][mode] = dict(overall=counted['overall'],
                by_step=[dict(step=int(step), **row) for step, row in counted['by_turn'].items()],
                replayed_missing_turns=restored,
                source_shards=[p.parent.name for p in sorted(directory.glob('*/summary.json'))])
            cases_by_mode[mode] = {c['trajectory_id']: c for c in cases}
        environment = TaskEnvironment(manifest['config'], manifest)
        events = {mode: dict(reached=Counter(), dead=Counter()) for mode in ('rollout', 'greedy')}
        horizon = 0
        shortest = Counter()
        for record in nonzero:
            key = record['trajectory_id']
            case = cases_by_mode['rollout'][key]
            assert len(case['actual_path']) == case['moves'] + 1
            horizon = max(horizon, case['moves'])
            if case['reached_goal']:
                assert case['actual_path'].index(str(record['goal'])) == case['moves']
                events['rollout']['reached'][case['moves']] += 1
                shortest['rollout'] += case['moves'] == record['shortest_moves']
            for row in case['turns']:
                if (row['turn'] < case['moves'] and row['remaining_shortest'] >= 1
                        and row.get('legal_action') and row.get('chosen_action') is not None
                        and not row['action_keeps_goal_reachable']):
                    assert not case['reached_goal']
                    events['rollout']['dead'][row['turn'] + 1] += 1
                    break
            demo = load_record(data_dir / 'trajectories', record, manifest['config'], 0)
            reference = cases_by_mode['reference'][key]
            assert len(reference['turns']) == len(demo.turns)
            assert all(str(turn.step.current) == row['current']
                       for turn, row in zip(demo.turns, reference['turns']))
            moves = len(demo.executed_path) - 1
            horizon = max(horizon, moves)
            if demo.success:
                events['greedy']['reached'][moves] += 1
                shortest['greedy'] += moves == record['shortest_moves']
            dead_at = next((row['turn'] for row in reference['turns']
                            if row['remaining_shortest'] < 0), None)
            if dead_at is None and environment.remaining(demo.executed_path[-1],
                    int(record['goal']), demo.executed_path) < 0:
                dead_at = moves
            if dead_at is not None:
                assert not demo.success and 1 <= dead_at <= moves
                events['greedy']['dead'][dead_at] += 1
        value['progress'] = {mode: cumulative(event, len(nonzero), horizon)
                             for mode, event in events.items()}
        value['nonzero_outcomes'] = {}
        for mode, event in events.items():
            reached, dead = sum(event['reached'].values()), sum(event['dead'].values())
            value['nonzero_outcomes'][mode] = dict(cases=len(nonzero), reached=reached,
                shortest=shortest[mode], dead=dead, ended_unreached_without_dead_end=len(nonzero)-reached-dead)
            assert reached + dead <= len(nonzero)
        assert value['nonzero_outcomes']['greedy']['reached'] == sum(r['greedy_success'] for r in nonzero)
        result['tasks'][task] = value
        print(json.dumps(dict(task=task, epoch=value['epoch'], outcomes=value['nonzero_outcomes'])), flush=True)
    return result


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    result = collect(args.run_root)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
