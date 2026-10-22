"""Read saved graphs and predictions; locate first policy errors without retraining."""
import argparse
from collections import Counter, defaultdict
import gzip
import json
from pathlib import Path
import numpy as np
from grounded_llm.core.files import atomic_json


def diagnose(run):
    run = Path(run)
    config = json.loads((run / 'config.json').read_text())
    with np.load(run / 'data.npz') as archive:
        data = dict(archive)
    with np.load(run / 'predictions.npz') as archive:
        predicted = archive['predicted']
    with gzip.open(run / 'scenes.jsonl.gz', 'rt') as stream:
        scenes = [json.loads(line) for line in stream]
    starts = data['starts'][data['split'][data['starts']] == 2]
    events, cycles, errors = Counter(), Counter(), []
    for start in starts:
        current, seen, first = int(start), {}, None
        for step in range(config['evaluation']['horizon']):
            if data['terminal'][current]:
                break
            if current in seen:
                cycles[step - seen[current]] += 1
                break
            seen[current] = step
            actions = np.flatnonzero(data['children'][current] >= 0)
            if not len(actions):
                break
            children = data['children'][current, actions]
            action = int(actions[(data['costs'][current, actions] + predicted[children]).argmin()])
            target = int(data['children'][current, action])
            remaining = data['costs'][current, action] + data['distances'][target]
            if first is None and remaining > data['distances'][current]:
                first = {'step': step, 'state': current, 'action': action, 'next': target,
                         'oracle_remaining_before': float(data['distances'][current]),
                         'chosen_remaining': float(remaining) if np.isfinite(remaining) else 'unreachable',
                         'state_seen_in_training': bool(data['train_feature_seen'][current])}
                events[str(step)] += 1
            current = target
        if first is not None:
            first.update(start=int(start), eventually_solved=bool(data['terminal'][current]))
            errors.append(first)
    audit = {'first_wrong_step_histogram': dict(events), 'cycle_length_histogram': dict(cycles),
             'first_error_examples': errors[:8]}
    if config['task'] == 'bridge':
        failed = {row['start'] for row in errors if not row['eventually_solved']}
        audit['failed_scenarios'] = [s['metadata'] for s in scenes if s['offset'] in failed]
    if config['task'] == 'game24':
        training, testing, features = set(), set(), {}
        aliases, depths = 0, defaultdict(list)
        for scene in scenes:
            for local, state in enumerate(scene['metadata']['exact_rational_states']):
                key, row = tuple(state), scene['offset'] + local
                feature = data['features'][row].tobytes()
                if feature in features and features[feature] != key:
                    aliases += 1
                features[feature] = key
                if scene['split'] == 'train': training.add(key)
                if scene['split'] == 'test': testing.add(key)
                if scene['split'] == 'test' and np.isfinite(data['distances'][row]) and not data['terminal'][row]:
                    actions = np.flatnonzero(data['children'][row] >= 0)
                    children = data['children'][row, actions]
                    chosen = children[(data['costs'][row, actions] + predicted[children]).argmin()]
                    depths[len(key)].append(bool(np.isfinite(data['distances'][chosen])))
        audit.update(unique_train_states=len(training), unique_test_states=len(testing),
                     unique_exact_state_overlap=len(training & testing), feature_alias_rows=aliases,
                     solvable_action_by_remaining_count={str(k): {'correct': sum(v), 'count': len(v)}
                                                        for k, v in depths.items()})
    return audit


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    atomic_json(args.out, {task: diagnose(args.root / task) for task in ('bridge', 'sokoban', 'game24')})
