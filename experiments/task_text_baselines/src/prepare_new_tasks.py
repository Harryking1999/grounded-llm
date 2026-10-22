"""Freeze a screening subset of the existing three-task Qmap test cases."""
import argparse
from collections import Counter
import gzip
import json
from pathlib import Path
import subprocess
import numpy as np
from grounded_llm.tasks.text import ROOT, prompt, judge
from grounded_llm.core.files import atomic_json


def reference(data, start, catalogue):
    state, actions, cost = int(start), [], 0
    while not data['terminal'][state]:
        legal = np.flatnonzero(data['children'][state] >= 0)
        children = data['children'][state, legal]
        index = int(legal[np.argmin(data['costs'][state, legal] + data['distances'][children])])
        actions.append(catalogue[index])
        cost += float(data['costs'][state, index])
        state = int(data['children'][state, index])
    return {'length': len(actions), 'cost': cost, 'actions': actions}


def cases_from_run(config):
    rng = np.random.default_rng(config['seed'])
    cases = []
    for task, options in config['tasks'].items():
        run = ROOT / config['source_runs'] / task
        with np.load(run / 'data.npz') as archive:
            data = dict(archive)
        with gzip.open(run / 'scenes.jsonl.gz', 'rt') as stream:
            scenes = [json.loads(line) for line in stream if json.loads(line)['split'] == 'test']
        selected = []
        for scene in scenes:
            sid = scene['scene']
            pool = data['starts'][data['goal_ids'][data['starts']] == sid]
            if task == 'sokoban':
                # Equal layout quotas and coverage across each layout's existing test depths.
                order = pool[rng.permutation(len(pool))]
                order = order[np.argsort(data['distances'][order], kind='stable')]
                pool = order[np.linspace(0, len(order) - 1, options['per_scene'], dtype=int)]
            selected.extend((int(start), scene) for start in pool)
        if options.get('count') is not None:
            chosen = sorted(rng.choice(len(selected), options['count'], replace=False).tolist())
            selected = [selected[i] for i in chosen]
        group = []
        for number, (start, scene) in enumerate(selected):
            meta = scene['metadata']
            case = {'id': f'{task}_{number:03d}', 'task': task,
                    'source_state_row': start, 'source_scene': scene['scene'],
                    'action_limit': options['action_limit']}
            if task == 'bridge':
                case.update(times=meta['times'], initial=0, goal='all_right', stratum='six_people')
                catalogue = meta['actions']
            elif task == 'sokoban':
                height, width = meta['shape']
                channels = data['features'][start].reshape(4, height, width)
                lines = []
                for r in range(height):
                    row = ''
                    for c in range(width):
                        wall, goal, box, player = channels[:, r, c].astype(bool)
                        row += '#' if wall else '+' if player and goal else '@' if player else '*' if box and goal else '$' if box else '.' if goal else ' '
                    lines.append(row)
                case.update(initial=lines, goal='all_boxes_on_targets', stratum=f'distance_{int(data["distances"][start]):02d}')
                catalogue = ['U', 'D', 'L', 'R']
            else:
                from grounded_llm.tasks.game24.environment import Game24
                case.update(initial=meta['numbers'], goal=[24], stratum='four_numbers')
                catalogue = [list(a) for a in Game24.actions]
            case['reference'] = reference(data, start, catalogue)
            case['prompt'] = prompt(case)
            verdict = judge(case, json.dumps({'actions': case['reference']['actions'], 'final_status': 'solved'}))
            if not verdict['contract_pass'] or not verdict['optimal']:
                raise ValueError((case['id'], verdict))
            group.append(case)
        print(json.dumps({'task': task, 'cases': len(group), 'strata': dict(Counter(c['stratum'] for c in group))}), flush=True)
        cases.extend(group)
    if len(cases) != config['expected_cases']:
        raise ValueError('Incorrect screening count')
    return cases


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding='utf-8'))
    args.out.mkdir(parents=True, exist_ok=True)
    if (args.out / 'suite.json').exists():
        raise FileExistsError('Frozen suite already exists')
    cases = cases_from_run(config)
    atomic_json(args.out / 'suite.json', {'config': config, 'cases': cases,
                'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()})


if __name__ == '__main__': main()
