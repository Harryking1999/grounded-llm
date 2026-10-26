"""Recreate finite Game21/24 and familiar-catalogue short Rush Hour OOD queries.

No Q predictions select an evaluation query. Rush catalogues are reconstructed
from the original complete public source and the unchanged original split rule.
"""
import argparse
from functools import lru_cache
from itertools import combinations
import json
from pathlib import Path
import numpy as np
from grounded_llm.core.files import atomic_json
from grounded_llm.core.qmap.exact_graph import UnitGraphOracle
from grounded_llm.core.qmap.physical_data import enumerate_component,group_split
from grounded_llm.tasks.rush_hour.environment import RushHour
from grounded_llm.tasks.rush_hour.relation_data import catalogue_key
from grounded_llm.tasks.game24.environment import Game24


def game_queries(cfg,out,rng):
    env=Game24();states=set()
    for numbers in combinations(range(1,11),4):
        todo=[env.canonical(numbers)];seen=set(todo)
        for state in todo:
            if len(state)>=2:states.add(state)
            for _,child,_ in env.transitions(state):
                if len(child)>=2 and child not in seen:seen.add(child);todo.append(child)
    @lru_cache(None)
    def reachable(state,target):
        if len(state)==1:return state[0]==target
        return any(reachable(child,target) for _,child,_ in env.transitions(state))
    rows=[];counts={}
    for size,count in [(4,cfg['game_roots_per_goal']),(3,cfg['game_three_suffixes_per_goal']),(2,cfg['game_two_suffixes_per_goal'])]:
        for target in [21,24]:
            pool=sorted((s for s in states if len(s)==size and reachable(s,target)))
            counts[f'{size}_to{target}']=len(pool)
            for i in rng.choice(len(pool),count,replace=False):
                state=pool[int(i)]
                rows.append(dict(task='game24',variant=f'{size}_to{target}_big_ood',start=list(map(str,state)),
                    goal=[str(target)],map_checkpoint='qmap.pt',context={}))
    atomic_json(out/'game24.json',rows);atomic_json(out/'game24_coverage.json',counts)


def rush_queries(cfg,out,rng):
    base=json.loads(Path(cfg['rush_original_config']).read_text())
    source=Path(cfg['rush_source']);lines=source.read_text().splitlines()
    diagnostic=set(base['data']['diagnostic_rows']);known_keys={catalogue_key(RushHour(line.split()[1])) for i,line in enumerate(lines,1) if i in diagnostic}
    boards=[];seen=set();rows=[];scopes=[];oracles=out/'rush_oracles';oracles.mkdir(exist_ok=True)
    for number,line in enumerate(lines,1):
        minimum,board,size=line.split();size=int(size)
        if 'x' in board or size>cfg['rush_maximum_component_states']:continue
        env=RushHour(board);key=catalogue_key(env)
        if key in known_keys or key in seen or group_split(key,base['seed'],base['data']['split_fractions'])!='train':continue
        states,index,children=enumerate_component(env,env.start,base['data']['max_states'])
        if len(states)!=size:raise ValueError('Public state count mismatch')
        terminals=[i for i,s in enumerate(states) if env.terminal(s)]
        if len(terminals)<=base['data']['heldout_goals']:continue
        seen.add(key);sid=len(scopes);name=f'scene_{sid:03d}.npz'
        np.savez(oracles/name,states=np.asarray(states,dtype=np.int8),children=children)
        oracle=UnitGraphOracle(children);pairs=set()
        for _ in range(cfg['rush_queries_per_catalogue']):
            for attempt in range(10000):
                terminal_goal=bool(rng.integers(2));goal=int(rng.choice(terminals)) if terminal_goal else int(rng.integers(len(states)))
                distances=oracle.to_goal(goal);depth=int(rng.integers(1,cfg['rush_maximum_distance']+1));sources=np.flatnonzero(distances==depth)
                if not len(sources):continue
                start=int(rng.choice(sources))
                if (start,goal) in pairs:continue
                pairs.add((start,goal));break
            else:raise ValueError('Insufficient distinct familiar short queries')
            rows.append(dict(task='rush_hour',variant='familiar_complete_goal_big_ood' if terminal_goal else 'familiar_relation_big_ood',
                start=list(states[start]),goal=list(states[goal]),map_checkpoint='qmap.pt',
                context=dict(board=board,oracle='rush_oracles/'+name),original_source_row=number,
                source_original_minimum=int(minimum)))
        scopes.append(dict(source_row=number,catalogue_group=key,states=size,
            familiar_scope='Original base source, train group, retained unchanged in1000-catalogue expansion'))
        if len(scopes)==cfg['rush_catalogues']:break
    if len(scopes)!=cfg['rush_catalogues']:raise ValueError('Not enough recovered familiar catalogues')
    atomic_json(out/'rush_hour.json',rows);atomic_json(out/'rush_hour_scope.json',dict(catalogues=scopes,
        query_filter='Exact distance stratum only; never model success',source_config=cfg['rush_original_config']))


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--config',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();cfg=json.loads(a.config.read_text());a.out.mkdir(parents=True,exist_ok=False);rng=np.random.default_rng(cfg['seed'])
    game_queries(cfg,a.out,rng);rush_queries(cfg,a.out,rng)
    atomic_json(a.out/'ready.json',dict(whole_category_ood=['rush_hour','game24'],no_llm_training=True,no_Q_filter=True))


if __name__=='__main__':main()
