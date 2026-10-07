"""Matched figure references from existing validation states; no LLM inference."""

from argparse import ArgumentParser
from collections import defaultdict
import json
from pathlib import Path

import torch

from .trajectory_dataset import load_record
from .trajectory_eval import TaskEnvironment


def collect(run):
    result = dict(run=run.name, split='validation', variant=0, tasks={},
        definitions=dict(reference='Solvable nonterminal gold-history states; same states as final reference.',
            random='Mean of per-state fractions, with uniform sampling among all legal candidates.',
            map_greedy='Existing physical demonstration action on each reference state; retains its fixed tie choice.',
            map_agreement='All nonterminal reference states, including already-dead states, matching the report.',
            rollout='Saved complete map-greedy physical episodes; never a claimed planning upper bound.',
            random_rollout='Not available; never extrapolate one-step chance into episode success.'))
    for task in ('path', 'blocks'):
        path = run / task / 'data/manifest.json'
        manifest = json.loads(path.read_text())
        records = [r for r in manifest['records'] if r['split'] == 'validation']
        baseline = {}
        for summary in (run / 'evaluation' / task / 'final/validation_reference_map').glob('*/summary.json'):
            for line in (summary.parent / 'cases.jsonl').read_text().splitlines():
                case = json.loads(line)
                assert case['variant'] == 0 and case['trajectory_id'] not in baseline
                baseline[case['trajectory_id']] = case
        assert len(baseline) == len(records)
        environment = TaskEnvironment(manifest['config'], manifest)
        live, map_chance, greedy_episodes = [], [], defaultdict(list)
        for index, record in enumerate(records, 1):
            demo = load_record(path.parent / 'trajectories', record, manifest['config'], 0)
            saved = baseline[record['trajectory_id']]
            assert len(saved['turns']) == len(demo.turns)
            for turn, row in zip(demo.turns, saved['turns']):
                step = turn.step
                count = len(step.candidate_actions)
                if step.done:
                    continue
                assert count > 0
                map_chance.append(len(step.map_minimal_candidates) / count)
                remaining = row['remaining_shortest']
                if remaining < 1:
                    continue
                distances = [environment.remaining(destination, step.goal, [*turn.executed_path, destination])
                             for destination in step.candidate_destinations]
                kept = sum(d >= 0 for d in distances)
                shortest = sum(d == remaining - 1 for d in distances)
                assert kept == row['reachable_candidates']
                chosen = turn.chosen_id
                assert chosen in step.map_minimal_candidates
                live.append(dict(chance_reachable=kept / count, chance_shortest=shortest / count,
                                 greedy_reachable=distances[chosen - 1] >= 0,
                                 greedy_shortest=distances[chosen - 1] == remaining - 1))
            reached = bool(demo.success)
            moves = len(demo.executed_path) - 1
            goal_type = ('initial_goal' if int(record['start']) == int(record['goal']) else
                         'empty' if int(record['goal']) == 0 else 'nonempty') if task == 'blocks' else 'path'
            value = dict(reached=reached, shortest=reached and moves == record['shortest_moves'])
            greedy_episodes['all'].append(value)
            greedy_episodes[goal_type].append(value)
            if index % 200 == 0:
                print(json.dumps(dict(task=task, processed=index)), flush=True)
        result['tasks'][task] = dict(reference_decisions=len(live),
            reference={k: sum(r[k] for r in live) / len(live) for k in live[0]},
            map_agreement=dict(decisions=len(map_chance), chance=sum(map_chance) / len(map_chance), greedy=1.),
            rollout={goal: dict(tasks=len(rows), reached=sum(r['reached'] for r in rows),
                               shortest=sum(r['shortest'] for r in rows)) for goal, rows in greedy_episodes.items()})
        assert sum(v['reached'] for v in greedy_episodes['all']) == sum(r['greedy_success'] for r in records)
    return result


def main():
    parser = ArgumentParser()
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    value = collect(args.run_root)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(value, indent=2) + '\n')
    print(json.dumps(value))


if __name__ == '__main__':
    main()
