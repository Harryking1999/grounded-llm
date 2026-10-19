"""Freeze fresh clearing and exact-distance nonempty-goal blocks questions."""
from collections import Counter
from contextlib import redirect_stdout
import io
import json
import random

from experiments.gcml_counterexamples.src import blocks
from .prepare import make_blocks
from .protocol import ROOT, judge, prompt


def isolated_cells(mask):
    result = []
    for cell in blocks.cells(mask):
        row, col = divmod(cell, blocks.SIZE)
        neighbors = [r * blocks.SIZE + c for r, c in
                     ((row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1))
                     if 0 <= r < blocks.SIZE and 0 <= c < blocks.SIZE]
        if not any(mask & (1 << neighbor) for neighbor in neighbors):
            result.append(cell)
    return result


def certify(source, goal, kind):
    """Removals commute: exact covering of the difference is exact goal distance."""
    if goal & source != goal:
        raise ValueError('Removal targets must be subsets of their sources')
    distance, indices = blocks.solver()(source ^ goal)
    if distance is None:
        raise ValueError('Unreachable target')
    case = {'task': 'blocks', 'initial': source, 'goal': goal,
            'goal_kind': kind, 'stratum': f'{kind}_distance_{distance}',
            'generation_placements': 12,
            'goal_isolated_cells': isolated_cells(goal),
            'reference': {'length': distance,
                          'actions': [blocks.PLACEMENTS[i][1] for i in indices]}}
    case['prompt'] = prompt(case)
    if not judge(case, json.dumps({'actions': case['reference']['actions'],
                                  'final_status': 'solved'}))['contract_pass']:
        raise ValueError('Reference failed exact-goal replay')
    return case


def make_cases(config, source):
    options = source['config']['tasks']['blocks']
    if options.get('shape_rows') or options['placements'] != 12:
        raise ValueError('This condition uses original eight shapes and twelve placements')
    rng, seen, cases = random.Random(config['seed']), set(), []
    for path in [config['source_suite'], *config.get('exclude_suites', [])]:
        for case in json.loads((ROOT / path).read_text(encoding='utf-8'))['cases']:
            if case['task'] == 'blocks':
                seen.add(blocks.normalized_key(int(case['initial'])))
    counts = Counter()
    isolated_counts = Counter()
    generation = config['generation']
    for _ in range(generation['max_candidate_roots']):
        # Suppress make_blocks' per-root progress; only accepted cases are logged.
        with redirect_stdout(io.StringIO()):
            root = make_blocks(options, 1, rng)[0]
        initial = root['initial']
        key = blocks.normalized_key(initial)
        if key in seen:
            continue
        seen.add(key)
        chosen = None
        if counts['clear'] < generation['clear_cases']:
            chosen = certify(initial, 0, 'clear')
        else:
            goals = set()
            # Both the clearing witness and unfiltered random legal paths supply
            # descendants. No model score or model answer selects questions.
            paths = [root['reference']['actions']]
            for _ in range(generation['random_paths_per_root']):
                state, actions = initial, []
                while True:
                    legal = blocks.legal_indices(state)
                    if not legal:
                        break
                    action = blocks.PLACEMENTS[rng.choice(legal)][1]
                    actions.append(action)
                    state = blocks.apply(state, action)
                paths.append(actions)
            for actions in paths:
                state = initial
                for step, action in enumerate(actions, 1):
                    state = blocks.apply(state, action)
                    if state and step >= generation['nonempty_min_distance']:
                        goals.add(state)
            candidates = list(sorted(goals))
            rng.shuffle(candidates)
            for goal in candidates:
                has_isolated = bool(isolated_cells(goal))
                kind = 'nonempty_isolated' if has_isolated else 'nonempty_retained'
                if counts[kind] >= generation[kind + '_cases']:
                    continue
                if not has_isolated and not blocks.locally_supported(goal):
                    continue
                distance, _ = blocks.solver()(initial ^ goal)
                if not generation['nonempty_min_distance'] <= distance <= generation['nonempty_max_distance']:
                    continue
                if has_isolated and isolated_counts[distance] >= generation['isolated_cases_per_distance']:
                    continue
                chosen = certify(initial, goal, kind)
                if has_isolated:
                    isolated_counts[distance] += 1
                break
        if chosen:
            chosen['id'] = f'blocks_mix_{len(cases):03d}'
            counts[chosen['goal_kind']] += 1
            cases.append(chosen)
            print(json.dumps({'prepared': len(cases), 'kind': chosen['goal_kind'],
                              'exact_distance': chosen['reference']['length']}), flush=True)
        if len(cases) == config['expected_cases']:
            break
    expected = {name: generation[name + '_cases']
                for name in ('clear', 'nonempty_isolated', 'nonempty_retained')}
    if dict(counts) != expected:
        raise ValueError(f'Could not meet frozen goal quotas: {dict(counts)} != {expected}')
    return cases
