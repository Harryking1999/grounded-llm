"""Pinned Boxoban hard layouts; exhaustive physical graphs and single-state goals."""
from itertools import combinations, permutations
from math import comb
import json
from pathlib import Path
import urllib.request
import numpy as np
from grounded_llm.core.files import atomic_json
from grounded_llm.core.qmap.physical_data import PhysicalCorpus, PhysicalScene, save_scene,group_seed,group_split
from grounded_llm.core.qmap.exact_graph import UnitGraphOracle
from .environment import Sokoban


def decode(rows):
    return [(int(row[0]), tuple(map(int, row[1:]))) for row in rows]


def baseline(source, goal):
    # Boxes are indistinguishable. Matching ignores walls and player positioning.
    source, goal = np.asarray(source), np.asarray(goal)
    best = np.full(len(source), np.inf)
    for order in permutations(range(4)):
        boxes = goal[:, 1:][:, order]
        score = (np.abs(source[:, 1:].astype(int)//10-boxes.astype(int)//10)
                 + np.abs(source[:, 1:].astype(int)%10-boxes.astype(int)%10)).sum(axis=1)
        best = np.minimum(best, score)
    player = np.abs(source[:, 0].astype(int)//10-goal[:, 0].astype(int)//10)+np.abs(source[:, 0].astype(int)%10-goal[:, 0].astype(int)%10)
    return np.maximum(best, player)


def wall_key(board):
    walls = np.asarray([[c == '#' for c in row] for row in board], dtype=np.uint8)
    return min(np.rot90(a, k).tobytes() for a in (walls, walls[:, ::-1]) for k in range(4)).hex()


def boards(text):
    current, identity = [], None
    for line in text.splitlines()+['; end']:
        if line.startswith(';'):
            if current: yield identity, current
            identity, current = line[1:].strip(), []
        elif line.startswith('#'):
            current.append(line)


def build(config, out):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    spec, rng = config['data'], np.random.default_rng(config['seed'])
    pool=[]
    for relative in spec.get('source_files',['hard/000.txt']):
        cache=out/relative.replace('/','_')
        source=Path(spec['source_cache'])
        source=source/relative.replace('/','_') if source.is_dir() else source
        if not cache.exists():
            url=f'https://raw.githubusercontent.com/google-deepmind/boxoban-levels/{spec["source_commit"]}/{relative}'
            cache.write_bytes(source.read_bytes() if source.exists() else urllib.request.urlopen(url,timeout=60).read())
        pool.extend((relative,identity,board) for identity,board in boards(cache.read_text(encoding='utf-8')))
    rng.shuffle(pool)
    known = set(spec['diagnostic_goals'])
    known_keys = {wall_key(b) for relative,identity,b in pool if relative=='hard/000.txt' and identity in known}
    used, records, rejected = set(), [], []
    selected = {name: 0 for name in spec['scenes']}
    for relative,identity,board in pool:
        env = Sokoban(board); key = wall_key(board)
        goal_rng=np.random.default_rng(group_seed(relative+':'+identity,config['seed'],'goals'))
        f = len(env.floor)
        count = comb(f, 4)*(f-4)
        if relative=='hard/000.txt' and identity in known:
            split = 'diagnostic'
        else:
            if key in known_keys or key in used:
                rejected.append(dict(id=identity, reason='equivalent_wall_group')); continue
            if count > spec['max_states']:
                rejected.append(dict(id=identity, reason='unknown_resource', physical_states=count)); continue
            split=group_split(key,config['seed'],spec['split_fractions'])
            if selected[split]>=spec['scenes'][split]: continue
        if key in used: continue
        floor = sorted(env.floor)
        states = [(p, boxes) for boxes in combinations(floor, 4) for p in floor if p not in boxes]
        index = {state: i for i, state in enumerate(states)}
        children = np.full((len(states), 4), -1, dtype=np.int32)
        for i, state in enumerate(states):
            for action, child, cost in env.transitions(state):
                if cost != 1: raise ValueError('Physical move cost changed')
                children[i, action] = index[child]
        players = [p for p in floor if p not in env.goals]
        goal_rng.shuffle(players)
        prescribed = spec['diagnostic_goals'].get(identity, []) if relative=='hard/000.txt' else []
        players = prescribed+[p for p in players if p not in prescribed]
        oracle=UnitGraphOracle(children)
        goals,labels=[],[]
        for player in players:
            goal=index[(player,tuple(sorted(env.goals)))]
            values=oracle.to_goal(goal)
            if split!='diagnostic' and np.isfinite(values).sum()<spec['minimum_reachable_per_goal']:
                continue
            goals.append(goal); labels.append(values)
            if spec['goals_per_scene']!='all' and len(goals)==spec['goals_per_scene']: break
        expected=spec['goals_per_scene'] if spec['goals_per_scene']!='all' else len(players)
        if len(goals)<expected:
            rejected.append(dict(id=identity,reason='insufficient_exact_reachable_goal_coverage',goals=len(goals)))
            continue
        sid = len(records)
        metadata = dict(task='sokoban', split=split, wall_group=key,
                        source=dict(commit=spec['source_commit'], file=relative, puzzle_id=identity),
                        initial_board=board, original_state=index[env.start],
                        training_goals=len(goals)-spec['heldout_goals'], symmetric=False)
        if split=='diagnostic': metadata['diagnostic_goal']=goals[0]
        packed = np.asarray([[p, *boxes] for p, boxes in states], dtype=np.uint8)
        record = save_scene(out/f'scene_{sid:03d}', packed, children, goals, metadata,
                            distances=np.stack(labels),reverse=np.stack([oracle.from_goal(g) for g in goals]))
        used.add(key); records.append(dict(path=f'scene_{sid:03d}', **record))
        if split != 'diagnostic': selected[split] += 1
        print(json.dumps(dict(task='sokoban', scene=sid, split=split, states=len(states),
                              finite_max=record['finite_max'])), flush=True)
        del states, index, children, packed
    if selected != spec['scenes']:
        raise RuntimeError(('Insufficient bounded hard layouts', selected))
    atomic_json(out/'corpus.json', dict(task='sokoban', scenes=records, rejections=rejected,
                                      source_layouts=len(pool), seed=config['seed'], selection=selected))
    return load(out)


def load(out):
    out = Path(out)
    manifest = json.loads((out/'corpus.json').read_text(encoding='utf-8'))
    return PhysicalCorpus([PhysicalScene(out/scene['path'], Sokoban(scene['initial_board']), decode, baseline)
                           for scene in manifest['scenes']])
