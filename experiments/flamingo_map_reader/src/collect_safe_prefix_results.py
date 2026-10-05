"""Separate early assistance from subsequent model completion in saved rollouts."""

from argparse import ArgumentParser
from collections import Counter
import json
from pathlib import Path

from .diagnostic_queue import read_cases


def metrics(cases):
    count = len(cases)
    reached = sum(c['reached_goal'] for c in cases)
    shortest = sum(c['reached_goal'] and c['moves'] == c['shortest_moves'] for c in cases)
    return dict(cases=count, reached=reached, reached_shortest=shortest,
                reached_rate=reached / count if count else None,
                reached_shortest_rate=shortest / count if count else None)


def collect(run, output):
    manifest = json.loads((run / 'blocks/data/manifest.json').read_text())
    records = {r['trajectory_id']: r for r in manifest['records'] if r['split'] == 'validation'}
    nonzero = {key for key, r in records.items() if int(r['start']) != int(r['goal'])}
    longer = {key for key in nonzero if records[key]['shortest_moves'] > 3}
    groups = [('baseline', 0, run / 'evaluation/blocks/final/validation_rollout_map')]
    groups += [(f'prefix_{k}', k, output / 'blocks' / f'safe_prefix_{k}_rollout') for k in (1, 2, 3)]
    result = dict(run=run.name, split='validation', variant=0,
        definitions=dict(nonzero='Excludes start=goal.',
            fixed_longer_cohort='Same tasks with environment shortest distance greater than 3 in every condition.',
            completed_by_prefix='Goal is reached within the first k executed actions, before the model resumes.',
            model_handoff='Nonzero tasks not completed by the assisted prefix; reach rates are conditional on this cohort.'),
        shortest_distance_histogram=dict(sorted(Counter(records[key]['shortest_moves'] for key in nonzero).items())),
        conditions={})
    for name, helper, group in groups:
        cases = read_cases(group)
        by_id = {c['trajectory_id']: c for c in cases}
        assert len(cases) == len(by_id) and set(by_id) == set(records)
        assert all(c['variant'] == 0 for c in cases)
        selected = [by_id[key] for key in sorted(nonzero)]
        value = dict(helper_steps=helper, all=metrics(cases), nonzero=metrics(selected),
            fixed_shortest_gt3=metrics([by_id[key] for key in sorted(longer)]),
            goal_types={goal: metrics([by_id[key] for key in sorted(nonzero)
                if ('empty' if int(records[key]['goal']) == 0 else 'nonempty') == goal])
                for goal in ('empty', 'nonempty')})
        if helper:
            completed = [c for c in selected if int(records[c['trajectory_id']]['goal'])
                in [int(state) for state in c['actual_path'][:helper + 1]]]
            completed_ids = {c['trajectory_id'] for c in completed}
            assert completed_ids == {key for key in nonzero if records[key]['shortest_moves'] <= helper}
            handed = [c for c in selected if c['trajectory_id'] not in completed_ids]
            value['completed_by_prefix'] = metrics(completed)
            value['model_handoff'] = metrics(handed)
            assert value['nonzero']['reached'] == len(completed) + value['model_handoff']['reached']
        result['conditions'][name] = value
    return result


def main():
    parser = ArgumentParser()
    for name in ('run-root', 'diagnostics-root', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    result = collect(args.run_root, args.diagnostics_root)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
