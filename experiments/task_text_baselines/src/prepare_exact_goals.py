"""Freeze the specified-goal diagnostic using existing starts and exact teachers."""
import argparse
from collections import deque
from copy import deepcopy
import json
from pathlib import Path
import random
import subprocess
import time

from grounded_llm.core.files import atomic_json
from grounded_llm.tasks.text import ROOT, judge, prompt
from grounded_llm.tasks.rush_hour.environment import RushHour
from grounded_llm.tasks.sokoban.environment import Sokoban
from grounded_llm.tasks.sokoban import text as sokoban_text
from grounded_llm.tasks.token_swapping.environment import TokenSwapping


def exact_plan(env, start, goal, options):
    """Unit-cost BFS over task-owned transitions; no search is exposed to the LLM."""
    parents, queue = {start: None}, deque([start])
    deadline = time.monotonic() + options['solver_timeout_seconds']
    while queue:
        state = queue.popleft()
        if state == goal:
            actions = []
            while parents[state] is not None:
                state, action = parents[state]
                actions.append(action)
            return list(reversed(actions)), len(parents)
        for action, child, cost in env.transitions(state):
            if cost != 1:
                raise ValueError('Specified-goal diagnostic requires unit-cost actions')
            if child not in parents:
                if len(parents) >= options['max_search_states']:
                    raise ValueError('Exact goal preparation exceeded the state bound')
                parents[child] = state, env.actions[action] if isinstance(env, Sokoban) else action
                queue.append(child)
        if time.monotonic() > deadline:
            raise ValueError('Exact goal preparation exceeded the time bound')
    raise ValueError('Specified complete goal is unreachable')


def sokoban_board(env, state):
    player, boxes = state
    cells = ['#' if cell not in env.floor else '.' if cell in env.goals else ' '
             for cell in range(env.height * env.width)]
    for cell in boxes:
        cells[cell] = '*' if cell in env.goals else '$'
    cells[player] = '+' if player in env.goals else '@'
    return [''.join(cells[row*env.width:(row+1)*env.width]) for row in range(env.height)]


def prepare(config):
    rng, cases = random.Random(config['seed']), []
    for task, options in config['tasks'].items():
        source = json.loads((ROOT/options['source_run']/'suite.json').read_text(encoding='utf-8'))
        for original in source['cases']:
            if original['task'] != task:
                continue
            if task == 'sokoban' and original['id'] not in options['case_ids']:
                continue
            if task == 'rush_hour' and original['source']['derivation'] != 'original':
                continue
            if task == 'token_swapping' and original['source']['architecture'] != options['architecture']:
                continue
            case = deepcopy(original)
            case.pop('prompt', None)
            case['id'] = f'exact_{original["id"]}'
            case['matched_source'] = dict(run=options['source_run'], case_id=original['id'])
            if task == 'token_swapping':
                env = TokenSwapping(len(case['initial']), case['edges'])
                labels = rng.sample(range(env.n), env.n)
                if labels == list(env.goal):
                    labels[0], labels[1] = labels[1], labels[0]
                case['initial'] = [labels[token] for token in original['initial']]
                case['goal'] = labels
                if env.relative_permutation(case['initial'], case['goal']) != tuple(original['initial']):
                    raise ValueError('Token relabelling did not preserve the exact source problem')
                case['preparation'] = dict(method='token-label automorphism of verified exact source',
                                           source_preparation=original.get('preparation'))
            else:
                env = Sokoban(case['initial']) if task == 'sokoban' else RushHour(case['initial'])
                goal = env.start
                for action in original['reference']['actions']:
                    goal = (sokoban_text.step(original, goal, action)[0] if task == 'sokoban'
                            else env.apply(goal, action)[0])
                case['goal_board'] = sokoban_board(env, goal) if task == 'sokoban' else ''.join(env.render(goal))
                case['goal'] = 'specified_complete_board'
                actions, visited = exact_plan(env, env.start, goal, options)
                case['reference'] = dict(actions=actions, length=len(actions), cost=len(actions))
                case['preparation'] = dict(method='unit-cost BFS to specified complete state', visited=visited,
                                           target_origin='endpoint of frozen legal source witness')
                if task == 'rush_hour' and len(actions) != original['reference']['cost']:
                    raise ValueError('Rush exact target lost the source minimum-cost certificate')
            case['stratum'] = 'specified_goal_' + original['stratum']
            case['prompt'] = prompt(case)
            verdict = judge(case, json.dumps(dict(actions=case['reference']['actions'], final_status='solved')))
            if not verdict['contract_pass'] or not verdict['optimal']:
                raise ValueError((case['id'], verdict))
            cases.append(case)
            print(json.dumps(dict(case=case['id'], optimum=case['reference']['cost'],
                                  visited=case['preparation'].get('visited'))), flush=True)
    if len(cases) != config['expected_cases']:
        raise ValueError(('Unexpected exact-goal case count', len(cases)))
    return cases


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    if (args.out/'suite.json').exists():
        raise FileExistsError('Specified-goal suite already frozen')
    config = json.loads(args.config.read_text(encoding='utf-8'))
    cases = prepare(config)
    atomic_json(args.out/'suite.json', dict(config=config, cases=cases,
        source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()))


if __name__ == '__main__':
    main()
