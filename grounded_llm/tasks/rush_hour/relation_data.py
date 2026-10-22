"""Public no-wall components, grouped by static vehicle catalogue."""
import json
from pathlib import Path
import urllib.request
import numpy as np
from grounded_llm.core.files import atomic_json
from grounded_llm.core.qmap.physical_data import PhysicalCorpus, PhysicalScene, enumerate_component, save_scene,group_seed,group_split
from .environment import RushHour


def decode(rows):
    return [tuple(map(int, row)) for row in rows]


def baseline(source, goal):
    return (np.asarray(source) != np.asarray(goal)).sum(axis=-1)


def catalogue_key(env):
    # Renaming ordinary cars cannot turn the same static layout into a new split.
    a = env.pieces[0][1:]
    rest = sorted(piece[1:] for piece in env.pieces[1:])
    direct = (a, tuple(rest))
    flipped = ((a[0], a[1], 5-a[2]), tuple(sorted((h, n, 5-lane if h else lane) for h, n, lane in rest)))
    return repr(min(direct, flipped))


def build(config, out):
    out=Path(out); out.mkdir(parents=True, exist_ok=True)
    spec, rng=config['data'], np.random.default_rng(config['seed'])
    cache=out/'rush1000.txt'; source=Path(spec['source_cache'])
    if not cache.exists():
        cache.write_bytes(source.read_bytes() if source.exists() else urllib.request.urlopen(spec['source_url'],timeout=60).read())
    pool=[]
    for row,line in enumerate(cache.read_text().splitlines(),1):
        distance,board,size=line.split()
        if 'x' not in board: pool.append((row,int(distance),board,int(size)))
    rng.shuffle(pool)
    diagnostic=set(spec['diagnostic_rows'])
    known_keys={catalogue_key(RushHour(b)) for row,_,b,_ in pool if row in diagnostic}
    used,records,rejected=set(),[],[]
    selected={name:0 for name in spec['scenes']}
    for row,distance,board,size in pool:
        env=RushHour(board); key=catalogue_key(env)
        goal_rng=np.random.default_rng(group_seed(str(row),config['seed'],'goals'))
        if row in diagnostic:
            split='diagnostic'
        else:
            if key in known_keys:
                rejected.append(dict(row=row,reason='equivalent_catalogue_group')); continue
            if size>spec['max_states']:
                rejected.append(dict(row=row,reason='unknown_resource',reported_states=size)); continue
            split=group_split(key,config['seed'],spec['split_fractions'])
            target=spec['scenes'][split]
            if target!='all' and selected[split]>=target: continue
        if board in used: continue
        states,index,children=enumerate_component(env,env.start,spec['max_states'])
        if len(states)!=size:
            raise ValueError(('Public component size mismatch',row,len(states),size))
        terminals=[i for i,s in enumerate(states) if env.terminal(s)]
        if len(terminals)<=spec['heldout_goals']:
            rejected.append(dict(row=row,reason='too_few_complete_goals',count=len(terminals))); continue
        count=len(terminals) if spec['goals_per_scene']=='all' else min(len(terminals),spec['goals_per_scene'])
        goals=goal_rng.choice(terminals,count,replace=False).tolist()
        metadata=dict(task='rush_hour',split=split,catalogue_group=key,
                      source=dict(url=spec['source_url'],row=row,original_minimum=distance,reported_states=size),
                      initial_board=board,original_state=0,training_goals=len(goals)-spec['heldout_goals'],symmetric=True)
        if split=='diagnostic':
            frozen=json.loads(Path(spec['diagnostic_suite']).read_text(encoding='utf-8'))
            case=next(c for c in frozen['cases'] if c['task']=='rush_hour' and c['source']['row']==row)
            metadata['diagnostic_goal']=index[RushHour(case['goal_board']).start]
        sid=len(records)
        record=save_scene(out/f'scene_{sid:03d}',states,children,goals,metadata,symmetric=True)
        # The minimum over ALL original-goal states validates the source move convention.
        from grounded_llm.core.qmap.exact_graph import UnitGraphOracle
        original_distances=UnitGraphOracle(children).from_goal(0)
        if min(original_distances[terminals])!=distance:
            raise ValueError(('Public minimum-slide label mismatch',row))
        used.add(board); records.append(dict(path=f'scene_{sid:03d}',**record))
        if split!='diagnostic': selected[split]+=1
        print(json.dumps(dict(task='rush_hour',scene=sid,split=split,states=len(states),finite_max=record['finite_max'])),flush=True)
    if any((selected[name]!=target if target!='all' else selected[name]==0) for name,target in spec['scenes'].items()):
        raise RuntimeError(('Insufficient grouped public layouts',selected))
    atomic_json(out/'corpus.json',dict(task='rush_hour',scenes=records,rejections=rejected,seed=config['seed'],selection=selected))
    return load(out,feature_encoding=config.get('input_encoding','identity'))


def load(out,feature_encoding='identity'):
    out=Path(out)
    manifest=json.loads((out/'corpus.json').read_text(encoding='utf-8'))
    return PhysicalCorpus([PhysicalScene(out/scene['path'],RushHour(scene['initial_board'],feature_encoding),decode,baseline)
                           for scene in manifest['scenes']])
