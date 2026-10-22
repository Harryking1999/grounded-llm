"""Freeze public-source Rush Hour and token-swapping screening conditions."""
import argparse
import json
from pathlib import Path
import random
import re
import subprocess
import urllib.request
from grounded_llm.core.files import atomic_json
from grounded_llm.tasks.text import ROOT, prompt, judge
from grounded_llm.tasks.rush_hour.environment import RushHour
from grounded_llm.tasks.token_swapping.environment import TokenSwapping


ALPHABET = '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'


def fetch(url, path):
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        data = urllib.request.urlopen(url, timeout=60).read()
        path.write_bytes(data)
    return path.read_text(encoding='utf-8')


def decode_public_tokens(text):
    """Follow tket DecodedProblemData, including inverse-permutation convention."""
    for name, block in re.findall(r'solutions\["([^"]+)"\]\s*=\s*\{(.*?)\};', text, re.S):
        entries = [''.join(re.findall(r'"([^"]*)"', part)) for part in block.split(',') if part.strip()]
        n = len(entries[0].split(':'))+1
        edges = [[a, ALPHABET.index(c)] for a, tail in enumerate(entries[0].split(':')) for c in tail]
        for index, encoded in enumerate(entries[1:], 1):
            swaps, separator, tokens = encoded.partition('_')
            if (set(map(ALPHABET.index, tokens)) if separator else set(map(ALPHABET.index, swaps))) != set(range(n)):
                continue
            actions = [[ALPHABET.index(swaps[i]), ALPHABET.index(swaps[i+1])] for i in range(0, len(swaps), 2)]
            placement = list(range(n))
            for a, b in actions:
                placement[a], placement[b] = placement[b], placement[a]
            initial = [placement.index(v) for v in range(n)]
            yield name, index, edges, initial, actions


def prepare(config):
    rng = random.Random(config['seed'])
    cache = ROOT/config['source_cache']
    cases = []
    opt = config['tasks']['rush_hour']
    source = fetch(opt['source_url'], cache/'rush1000.txt')
    pool = []
    for line_number, line in enumerate(source.splitlines(), 1):
        minimum, board, size = line.split()
        if board.count('x') == opt['wall_count'] and int(size) <= opt['max_component_states']:
            pool.append((line_number, int(minimum), board, int(size)))
    deepest = max(pool, key=lambda item: item[1])
    pool.remove(deepest)
    selected = [deepest]+rng.sample(pool, opt['scenes']-1)
    for scene, (source_row, minimum, board, size) in enumerate(selected):
        env = RushHour(board)
        actions, count = env.solve(opt['max_component_states'])
        if actions is None or len(actions) != minimum or count != size:
            raise ValueError(('Rush Hour source label mismatch', source_row))
        distances = sorted({2, minimum//4, minimum//2, 3*minimum//4, minimum})
        if len(distances) != opt['starts_per_scene']:
            raise ValueError('Source puzzle too shallow for distinct screening depths')
        for distance in distances:
            state = env.start
            for action in actions[:minimum-distance]:
                state, _ = env.apply(state, action)
            cases.append(dict(id=f'rush_hour_{len(cases):03d}', task='rush_hour',
                initial=''.join(env.render(state)), goal='A_rightmost_in_board',
                stratum=f'distance_{distance:02d}', action_limit=opt['action_limit'],
                source=dict(url=opt['source_url'], row=source_row, original_board=board,
                    original_minimum_moves=minimum, component_states=count, scene=scene,
                    derivation='original' if distance == minimum else 'shortest-plan intermediate state'),
                reference=dict(actions=actions[minimum-distance:], length=distance, cost=distance)))
        print(json.dumps({'rush_scene':scene,'source_row':source_row,'distance':minimum,'states':count}),flush=True)
    opt = config['tasks']['token_swapping']
    url = f'https://raw.githubusercontent.com/Quantinuum/tket/{opt["source_commit"]}/{opt["source_file"]}'
    source = fetch(url, cache/'FixedCompleteSolutions.cpp')
    public = list(decode_public_tokens(source))
    for architecture in opt['architectures']:
        pool = [row for row in public if row[0] == architecture]
        rng.shuffle(pool)
        seen, accepted, rejected, duplicates = set(), 0, 0, 0
        for _, index, edges, initial, upstream in pool:
            if tuple(initial) in seen or initial == list(range(len(initial))):
                duplicates += 1
                continue
            seen.add(tuple(initial))
            env = TokenSwapping(len(initial), edges)
            current = tuple(initial)
            for action in upstream:
                current, _ = env.apply(current, action)
            if current != env.goal:
                raise ValueError('Upstream full-permutation decoding mismatch')
            try:
                actions, stats = env.solve(initial, opt['max_search_states'], opt['solver_timeout_seconds'])
            except ValueError:
                rejected += 1
                continue
            if actions is None:
                raise ValueError('Connected upstream instance unexpectedly unsolvable')
            cases.append(dict(id=f'token_swapping_{sum(c["task"] == "token_swapping" for c in cases):03d}',
                task='token_swapping', initial=initial, goal=list(env.goal), edges=edges,
                stratum=architecture, action_limit=opt['action_limit'],
                source=dict(url=url, commit=opt['source_commit'], architecture=architecture,
                    entry_index=index, upstream_cost=len(upstream), upstream_actions=upstream),
                preparation=dict(exact_search=stats, earlier_bound_rejections=rejected, earlier_duplicates=duplicates),
                reference=dict(actions=actions, length=len(actions), cost=len(actions))))
            accepted += 1
            if accepted == opt['per_architecture']:
                break
        if accepted != opt['per_architecture']:
            raise ValueError(('Insufficient exact public cases', architecture, accepted, rejected))
        print(json.dumps({'token_architecture':architecture,'accepted':accepted,'bound_rejections':rejected}),flush=True)
    if len(cases) != config['expected_cases']:
        raise ValueError('Wrong frozen screening size')
    for case in cases:
        case['prompt'] = prompt(case)
        verdict = judge(case, json.dumps({'actions':case['reference']['actions'],'final_status':'solved'}))
        if not verdict['contract_pass'] or not verdict['optimal']:
            raise ValueError((case['id'], verdict))
    return cases


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    if (args.out/'suite.json').exists():
        raise FileExistsError('Frozen suite already exists')
    config = json.loads(args.config.read_text(encoding='utf-8'))
    cases = prepare(config)
    atomic_json(args.out/'suite.json', dict(config=config, cases=cases,
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()))


if __name__ == '__main__':
    main()
