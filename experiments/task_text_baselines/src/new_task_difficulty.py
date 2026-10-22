"""Task-local harder settings with reference witnesses, for first-error screening."""
import csv
from fractions import Fraction
import io
import json
from pathlib import Path
import random
import urllib.request
from grounded_llm.tasks.text import ROOT, prompt, judge
from grounded_llm.tasks.bridge.solve import optimal_plan
from grounded_llm.tasks.game24.environment import Game24
from grounded_llm.tasks.game24.solve import solve as solve24
from grounded_llm.tasks.sokoban.environment import Sokoban
from grounded_llm.tasks.sokoban.solve import solve as solve_boxes


def download(url, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_bytes(urllib.request.urlopen(url, timeout=60).read())
    return path.read_text(encoding='utf-8')


def boxoban_boards(text):
    boards, current, identity = [], [], None
    for line in text.splitlines() + ['; end']:
        if line.startswith(';'):
            if current:
                boards.append((identity, current))
            identity, current = line[1:].strip(), []
        elif line and line.startswith('#'):
            current.append(line)
    return boards


def constructed_numbers(limit, rng):
    """Construct an integer fourth input by inverting a random rational expression."""
    if rng.randrange(2) == 0:
        # a/(b-c/d)=24, with a=24q+r, b=q+u, c=k(24u-r), d=24k.
        # This branch avoids vanishing integer-acceptance rates at large ranges.
        a = rng.randint(1, limit)
        q, r = divmod(a, 24)
        if r == 0: return None
        u = rng.randint(1, min(10, max(1, limit//24)))
        factor = 24*u-r
        maximum_k = min(limit//24, limit//factor)
        if maximum_k < 1 or q+u > limit: return None
        k = rng.randint(1, maximum_k)
        return tuple(sorted([a, q+u, k*factor, 24*k]))
    numbers = [rng.randint(1, limit) for _ in range(3)]
    a, b, c, d = Fraction(1), Fraction(0), Fraction(0), Fraction(1)
    for value in numbers:
        operation = rng.randrange(6)
        if operation == 0: a, b = a + value*c, b + value*d
        elif operation == 1: a, b = a - value*c, b - value*d
        elif operation == 2: a, b = value*c - a, value*d - b
        elif operation == 3: a, b = value*a, value*b
        elif operation == 4: c, d = value*c, value*d
        else: a, b, c, d = value*c, value*d, a, b
    if a*d == b*c or a == 24*c:
        return None
    fourth = (24*d - b)/(a - 24*c)
    if fourth.denominator != 1 or not 1 <= fourth <= limit:
        return None
    return tuple(sorted(numbers + [int(fourth)]))


def prepare_level(task, level, options, config):
    rng = random.Random(config['seed'] + list(config['tasks']).index(task)*100000 + level*1000)
    count = config['cases_per_level']
    excluded = json.loads((ROOT / config['baseline_run'] / 'suite.json').read_text(encoding='utf-8'))['cases']
    seen = {tuple(sorted(c['initial'])) for c in excluded if c['task'] == 'game24'} if task == 'game24' else set()
    cases, attempts = [], 0
    if task == 'sokoban':
        value = options['boxoban_sets'][level]
        relative = options['files'][value]
        text = download(f'https://raw.githubusercontent.com/google-deepmind/boxoban-levels/{options["source_commit"]}/{relative}',
                        ROOT / config['source_cache'] / relative.replace('/', '_'))
        boards = boxoban_boards(text)
        rng.shuffle(boards)
    elif task == 'game24':
        value = options['number_limits'][level]
        if value == 13:
            pool = []
            for split in ('test', 'train'):
                text = download(f'https://raw.githubusercontent.com/microsoft/Everything-of-Thoughts-XoT/{options["source_commit"]}/xot_mcts/game24/data/{split}.csv',
                                ROOT / config['source_cache'] / f'xot_{split}.csv')
                pool.extend(tuple(sorted(map(int, row['Puzzles'].split()))) for row in csv.DictReader(io.StringIO(text)))
            pool = list(dict.fromkeys(pool))
            rng.shuffle(pool)
    else:
        value = options['people_counts'][level]
    while len(cases) < count:
        attempts += 1
        if attempts > options.get('max_generation_attempts', 20000):
            raise ValueError(f'Unable to construct {task} level {value}')
        case = {'task': task, 'level': level, 'level_value': value,
                'id': f'{task}_level{level}_{len(cases):02d}'}
        if task == 'bridge':
            times = rng.sample(range(1, options['max_time'] + 1), value)
            identity = tuple(times)
            if identity in seen: continue
            seen.add(identity)
            actions, cost = optimal_plan(times)
            case.update(times=times, initial=0, goal='all_right', action_limit=max(128, 2*value+32),
                        stratum=f'people_{value}', reference={'actions': actions, 'length': len(actions), 'cost': cost})
        elif task == 'sokoban':
            if not boards: raise ValueError('Boxoban source exhausted')
            source_id, board = boards.pop()
            actions, stats = solve_boxes(Sokoban(board), options['solver_max_expanded'], options['solver_timeout_seconds'])
            if actions is None or len(actions) > options['action_limit']:
                continue
            case.update(initial=board, goal='all_boxes_on_targets', action_limit=options['action_limit'],
                        stratum=f'boxoban_{value}', source={'file': relative, 'puzzle_id': source_id, 'commit': options['source_commit']},
                        preparation={'solver': stats, 'candidates_examined': attempts},
                        reference={'actions': actions, 'length': None, 'cost': None, 'upper_bound': len(actions)})
        else:
            if value == 13:
                if not pool: raise ValueError('Fraction-required XoT puzzle pool exhausted')
                numbers = pool.pop()
            else:
                numbers = constructed_numbers(value, rng)
                if numbers is None or max(numbers) <= options['number_limits'][level-1]: continue
            if numbers in seen: continue
            state = Game24.canonical(numbers)
            if solve24(state, True) is not None: continue
            actions = solve24(state)
            if actions is None: continue
            seen.add(numbers)
            case.update(initial=list(numbers), goal=[24], action_limit=3,
                        stratum=f'four_numbers_fraction_required_max_{value}',
                        preparation={'integer_intermediate_solution': False, 'candidates_examined': attempts,
                                     'source': 'XoT CSV' if value == 13 else 'constructive rational expression; not uniform'},
                        reference={'actions': [list(a) for a in actions], 'length': 3, 'cost': 3})
        case['prompt'] = prompt(case)
        verdict = judge(case, json.dumps({'actions': case['reference']['actions'], 'final_status': 'solved'}))
        if not verdict['contract_pass']:
            raise ValueError((case['id'], verdict))
        cases.append(case)
    return cases
