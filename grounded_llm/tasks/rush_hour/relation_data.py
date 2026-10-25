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
    if config['data'].get('append_from'):return expand(config,out)
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
        ordinary=spec.get('intermediate_goals_per_scene',0)
        extra_held=spec.get('heldout_intermediate_goals',0)
        if ordinary:
            pool=np.asarray([i for i,s in enumerate(states) if not env.terminal(s)])
            extras=goal_rng.choice(pool,ordinary,replace=False).tolist()
            held=spec['heldout_goals']
            goals=goals[:-held]+extras[:ordinary-extra_held]+goals[-held:]+extras[ordinary-extra_held:]
        metadata=dict(task='rush_hour',split=split,catalogue_group=key,
                      source=dict(url=spec['source_url'],row=row,original_minimum=distance,reported_states=size),
                      initial_board=board,original_state=0,training_goals=len(goals)-spec['heldout_goals']-extra_held,symmetric=True,
                      intermediate_goals=ordinary)
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


def expand(config,out):
    """Append distinct training catalogues without changing reference scenes."""
    import gzip
    import shutil
    from grounded_llm.core.qmap.exact_graph import UnitGraphOracle
    out=Path(out);out.mkdir(parents=True,exist_ok=True);spec=config['data']
    reference=(out.parents[1]/spec['append_from']).resolve()
    manifest=json.loads((reference/'corpus.json').read_text())
    records=[]
    for record in manifest['scenes']:
        destination=out/record['path']
        if not destination.exists():shutil.copytree(reference/record['path'],destination)
        if json.loads((destination/'scene.json').read_text())!=json.loads((reference/record['path']/'scene.json').read_text()):
            raise ValueError('Reference scene changed during expansion')
        records.append(record)
    used={r['catalogue_group'] for r in records}
    train={r['catalogue_group'] for r in records if r['split']=='train'}
    held={r['catalogue_group'] for r in records if r['split']!='train'}
    if train & held:raise ValueError('Reference catalogue splits overlap')
    source=Path(spec['source_cache'])
    if not source.exists():
        source.parent.mkdir(parents=True,exist_ok=True)
        source.write_bytes(urllib.request.urlopen(spec['source_url'],timeout=120).read())
    added=[];strata=spec.get('layout_strata',[])
    def bucket(minimum):
        return next((i for i,s in enumerate(strata) if s['minimum_moves']<=minimum<=s['maximum_moves']),None)
    counts=[len({r['catalogue_group'] for r in records if r['split']=='train' and bucket(r['source']['original_minimum'])==i}) for i in range(len(strata))]
    candidates=_expansion_candidates(source,spec,config['seed'],used,counts)
    for row,line in candidates:
            if len(train)>=spec['minimum_training_catalogues']:break
            minimum,board,size=line.split();size=int(size)
            if 'x' in board or size>spec['max_states']:continue
            group=bucket(int(minimum))
            if strata and (group is None or counts[group]>=strata[group]['training_catalogues']):continue
            env=RushHour(board);key=catalogue_key(env)
            if key in used:continue
            path=f'scene_{len(records):03d}';destination=out/path
            if (destination/'scene.json').exists():
                record=json.loads((destination/'scene.json').read_text())
                if record['initial_board']!=board or record['catalogue_group']!=key:raise ValueError('Partial expansion has different source order')
            else:
                states,index,children=enumerate_component(env,env.start,spec['max_states'])
                if len(states)!=size:raise ValueError(('Public component size mismatch',row,len(states),size))
                oracle=UnitGraphOracle(children)
                terminal=np.asarray([i for i,s in enumerate(states) if env.terminal(s)])
                if not len(terminal) or min(oracle.from_goal(0)[terminal])!=int(minimum):raise ValueError('Public move label mismatch')
                rng=np.random.default_rng(group_seed(str(row),config['seed'],'expanded_goals'))
                nt=min(len(terminal),spec['terminal_goals_per_scene'])
                if nt<=spec['heldout_goals']:continue
                end=rng.choice(terminal,nt,replace=False).tolist()
                ordinary=np.asarray([i for i,s in enumerate(states) if not env.terminal(s)])
                if len(ordinary)<spec['intermediate_goals_per_scene']:continue
                mid=rng.choice(ordinary,spec['intermediate_goals_per_scene'],replace=False).tolist()
                a,b=spec['heldout_goals'],spec['heldout_intermediate_goals']
                goals=end[:-a]+mid[:-b]+end[-a:]+mid[-b:]
                metadata=dict(task='rush_hour',split='train',catalogue_group=key,
                    source=dict(url=spec['source_url'],row=row,original_minimum=int(minimum),reported_states=size),
                    initial_board=board,original_state=0,training_goals=len(goals)-a-b,symmetric=True,
                    intermediate_goals=len(mid))
                record=save_scene(destination,states,children,goals,metadata,symmetric=True)
            records.append(dict(path=path,**record));used.add(key);train.add(key);added.append(row)
            if strata:counts[group]+=1
            print(json.dumps(dict(task='rush_hour',training_catalogues=len(train),scene=path,states=record['states'])),flush=True)
    if len(train)<spec['minimum_training_catalogues']:raise ValueError(('Insufficient independent training catalogues',len(train)))
    if strata and any(n<s['training_catalogues'] for n,s in zip(counts,strata)):raise ValueError(('Unfilled layout difficulty strata',counts))
    atomic_json(out/'corpus.json',dict(task='rush_hour',scenes=records,seed=config['seed'],reference=str(reference),
        appended_source_rows=added,training_catalogues=len(train),reference_scenes_unchanged=True,
        catalogue_stratum_counts=counts,
        selection={s:sum(r['split']==s for r in records) for s in ('train','validation','test','diagnostic')}))
    return load(out,feature_encoding=config.get('input_encoding','identity'))


def _expansion_candidates(source,spec,seed,excluded,counts):
    """Bounded deterministic priority sample from the full public database."""
    import gzip
    import heapq
    strata=spec.get('layout_strata',[])
    if not strata:
        with gzip.open(source,'rt') as stream:
            yield from enumerate(stream,1)
        return
    if sum(s['training_catalogues'] for s in strata)!=spec['minimum_training_catalogues']:
        raise ValueError('Difficulty quotas must sum to the declared catalogue target')
    heaps=[[] for _ in strata];chosen=set(excluded)
    with gzip.open(source,'rt') as stream:
        for row,line in enumerate(stream,1):
            minimum,board,size=line.split();minimum=int(minimum)
            if 'x' in board or int(size)>spec['max_states']:continue
            group=next((i for i,s in enumerate(strata) if s['minimum_moves']<=minimum<=s['maximum_moves']),None)
            if group is None:continue
            capacity=max(0,strata[group]['training_catalogues']-counts[group])*spec.get('selection_reserve',4)
            if not capacity:continue
            priority=group_seed(str(row),seed,'coverage_catalogue')
            heap=heaps[group]
            if len(heap)>=capacity and priority>=-heap[0][0]:continue
            env=RushHour(board)
            if len(env.pieces)<spec.get('minimum_vehicles',1):continue
            key=catalogue_key(env)
            if key in chosen:continue
            if len(heap)>=capacity:
                removed=heapq.heappop(heap);chosen.remove(removed[2])
            heapq.heappush(heap,(-priority,row,key,line));chosen.add(key)
    for heap in heaps:
        for _,row,_,line in sorted(heap,reverse=True):yield row,line
