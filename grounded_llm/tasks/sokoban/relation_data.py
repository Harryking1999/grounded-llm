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


def physical_graph(env):
    """Same exhaustive state order as combinations x player, without Python rows."""
    if len(env.goals)!=4: raise ValueError('This corpus requires four boxes')
    floor=np.asarray(sorted(env.floor),dtype=np.int16); f=len(floor)
    boxes=np.asarray(list(combinations(range(f),4)),dtype=np.int16)
    players=np.broadcast_to(np.arange(f),(len(boxes),f))
    player=players[~(players[:,:,None]==boxes[:,None,:]).any(axis=2)]
    box=np.repeat(boxes,f-4,axis=0)
    packed=np.column_stack([floor[player],floor[box]]).astype(np.uint8)
    weights=np.asarray([f**3,f**2,f,1],dtype=np.int64)
    combinations_index=np.full(f**4,-1,dtype=np.int32)
    combinations_index[boxes@weights]=np.arange(len(boxes))
    cell_index=np.full(env.width*env.height,-1,dtype=np.int16); cell_index[floor]=np.arange(f)
    adjacent=np.full((f,4),-1,dtype=np.int16)
    for i,cell in enumerate(floor):
        for a,delta in enumerate(env.deltas):
            dest=env.adjacent(int(cell),delta)
            if dest in env.floor: adjacent[i,a]=cell_index[dest]
    def state_id(state):
        p,b=state; p=cell_index[p]; b=cell_index[list(b)]
        return int(combinations_index[b@weights]*(f-4)+p-(b<p).sum())
    children=np.full((len(packed),4),-1,dtype=np.int32)
    for a in range(4):
        dest=adjacent[player,a]; hit=box==dest[:,None]; pushed=hit.any(axis=1)
        beyond=adjacent[np.maximum(dest,0),a]
        valid=(dest>=0) & (~pushed | ((beyond>=0) & ~(box==beyond[:,None]).any(axis=1)))
        next_boxes=box[valid].copy()
        replace=hit[valid] & pushed[valid,None]
        next_boxes[replace]=np.broadcast_to(beyond[valid,None],next_boxes.shape)[replace]
        next_boxes.sort(axis=1)
        combo=combinations_index[next_boxes@weights]
        if (combo<0).any(): raise ValueError('Invalid successor box combination')
        children[valid,a]=combo*(f-4)+dest[valid]-(next_boxes<dest[valid,None]).sum(axis=1)
    return packed,children,state_id


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
    selection=spec.get('layout_selection')
    if selection:
        chosen=json.loads(Path(selection).read_text(encoding='utf-8'))
        if chosen['source_commit']!=spec['source_commit']: raise ValueError('Layout source mismatch')
        wanted={tuple(row) for split,rows in chosen['layouts'].items()
                for row in rows[:spec['scenes'][split]]}
        pool=[row for row in pool if (row[0],row[1]) in wanted or
              (row[0]=='hard/000.txt' and row[1] in spec['diagnostic_goals'])]
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
        sid=len(records); scene_path=out/f'scene_{sid:03d}'
        if (scene_path/'scene.json').exists():
            record=json.loads((scene_path/'scene.json').read_text())
            if (record['source']!=dict(commit=spec['source_commit'],file=relative,puzzle_id=identity)
                    or record['split']!=split or record['initial_board']!=board):
                raise ValueError('Existing graph belongs to another layout selection')
            used.add(key); records.append(dict(path=f'scene_{sid:03d}',**record))
            if split!='diagnostic': selected[split]+=1
            continue
        floor=sorted(env.floor)
        packed,children,state_id=physical_graph(env)
        players = [p for p in floor if p not in env.goals]
        goal_rng.shuffle(players)
        prescribed = spec['diagnostic_goals'].get(identity, []) if relative=='hard/000.txt' else []
        players = prescribed+[p for p in players if p not in prescribed]
        oracle=UnitGraphOracle(children)
        goals,labels=[],[]
        for player in players:
            goal=state_id((player,tuple(sorted(env.goals))))
            values=oracle.to_goal(goal)
            if split!='diagnostic' and np.isfinite(values).sum()<spec['minimum_reachable_per_goal']:
                continue
            goals.append(goal); labels.append(values)
            if spec['goals_per_scene']!='all' and len(goals)==spec['goals_per_scene']: break
        expected=spec['goals_per_scene'] if spec['goals_per_scene']!='all' else len(players)
        if len(goals)<expected:
            rejected.append(dict(id=identity,reason='insufficient_exact_reachable_goal_coverage',goals=len(goals)))
            continue
        metadata = dict(task='sokoban', split=split, wall_group=key,
                        source=dict(commit=spec['source_commit'], file=relative, puzzle_id=identity),
                        initial_board=board, original_state=state_id(env.start),
                        training_goals=len(goals)-spec['heldout_goals'], symmetric=False)
        if split=='diagnostic': metadata['diagnostic_goal']=goals[0]
        record = save_scene(out/f'scene_{sid:03d}', packed, children, goals, metadata,
                            distances=np.stack(labels),reverse=np.stack([oracle.from_goal(g) for g in goals]))
        used.add(key); records.append(dict(path=f'scene_{sid:03d}', **record))
        if split != 'diagnostic': selected[split] += 1
        print(json.dumps(dict(task='sokoban', scene=sid, split=split, states=len(packed),
                              finite_max=record['finite_max'])), flush=True)
        del state_id, children, packed, labels, oracle
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
