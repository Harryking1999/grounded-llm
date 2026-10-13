"""All-source long-first map-greedy shortest trajectories and common caches."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing as mp
from pathlib import Path
import time
import numpy as np
import torch
from experiments.blocks_distance_map.src.multiboard_data import sample_paths
from experiments.blocks_distance_map.src.oracle import DistanceOracle, OracleBudgetExceeded
from experiments.flamingo_map_reader.src.blocks import FrozenBoardMap
from experiments.flamingo_map_reader.src.transcript import encode_trajectory
from experiments.flamingo_map_reader.src.trajectory_dataset import pack_demo
from .common import atomic_json, sorted_step, demonstration

class CachedMap:
    def __init__(self,qmap):
        self.qmap,self.cache,self.metric=qmap,{},qmap.metric
    def encode(self,states):
        missing=list(dict.fromkeys(s for s in states if s not in self.cache))
        if missing:
            vectors=self.qmap.encode(missing)
            self.cache.update(zip(missing,vectors))
        return torch.stack([self.cache[s] for s in states])

def init_worker(config,q_path,model_path,excluded,out):
    global CFG,QMAP,TOKENIZER,EXCLUDED,OUT
    from transformers import AutoTokenizer
    torch.set_num_threads(1)
    CFG=config;QMAP=FrozenBoardMap.load(Path(q_path));EXCLUDED={tuple(map(int,p)) for p in excluded};OUT=Path(out)
    TOKENIZER=AutoTokenizer.from_pretrained(model_path,local_files_only=True)

def goal_bucket(goal):
    return 0 if not goal else 1 if goal.bit_count()<=2 else 2 if goal.bit_count()<=4 else 3

def choose(candidates,quota,seen,rng):
    selected=[];starts=Counter();sizes=Counter();goals=Counter()
    pool=list(candidates)
    while pool and len(selected)<quota:
        length=max(r['shortest_moves'] for r in pool)
        same=[r for r in pool if r['shortest_moves']==length]
        rng.shuffle(same)
        best=max(same,key=lambda r:(len(set(map(tuple,r['pairs']))-seen),-starts[r['start']],-sizes[goal_bucket(int(r['goal']))],-goals[r['goal']]))
        selected.append(best);pool.remove(best);seen.update(map(tuple,best['pairs']))
        starts[best['start']]+=1;sizes[goal_bucket(int(best['goal']))]+=1;goals[best['goal']]+=1
    return selected

def board_worker(board):
    row,mask,reference=board; board=(row,int(mask),reference)
    rng=np.random.default_rng(CFG['seed']+int(row));q=CachedMap(QMAP)
    oracle=DistanceOracle(cache_limit=CFG['data']['oracle_cache_limit'],seconds=CFG['data']['oracle_seconds_per_board'])
    evaluated=set();candidates=[];unknown=0;started=time.monotonic();selected=[]
    paths=sample_paths(board,0,rng)
    for number in CFG['data']['random_routes']:
        previous=len(paths)-1
        paths.extend(sample_paths(board,number-previous,rng)[1:])
        pairs={(s,g) for path in paths for i,s in enumerate(path[:-1]) for g in path[i+1:]}
        scored=[]
        for start,goal in sorted(pairs-evaluated-EXCLUDED):
            evaluated.add((start,goal))
            try:d=oracle.distance(start,goal)
            except OracleBudgetExceeded:
                unknown+=1;continue
            if d>=CFG['data']['minimum_shortest_moves']:scored.append((start,goal,d))
        # Check every candidate at each length; never let Q-success acceptance order replace coverage selection.
        for start,goal,d in sorted(scored,key=lambda r:-r[2]):
            state=start;route=[];visited=[]
            for index in range(d):
                if (state,goal) in EXCLUDED:break
                visited.append((state,goal));step=sorted_step(q,state,goal)
                if not step.map_minimal_candidates:break
                candidate=step.map_minimal_candidates[0];_,after=step.execute(candidate)
                try:remaining=oracle.distance(after,goal)
                except OracleBudgetExceeded:
                    unknown+=1;break
                if remaining!=d-index-1:break
                route.append(candidate);state=after
            if state==goal and len(route)==d:
                candidates.append(dict(start=str(start),goal=str(goal),shortest_moves=d,route_ids=route,pairs=visited,board_row=int(row)))
        selected=choose(candidates,CFG['data']['ordinary_per_board'],set(),rng)
        unique=set(p for r in selected for p in r['pairs'])
        if len(selected)==CFG['data']['ordinary_per_board'] and len(unique)>=CFG['data']['coverage_per_board_target']:break
    if len(selected)!=CFG['data']['ordinary_per_board']:
        raise ValueError(f'Board {row}: only {len(selected)} qualifying distinct routes; no repetition permitted')
    selected.append(dict(start=str(board[1]),goal=str(board[1]),shortest_moves=0,route_ids=[],pairs=[],board_row=int(row)))
    rows=[]
    for index,record in enumerate(selected):
        name=f'{row}_{index:02d}.pt';record['trajectory_id']=f'aug_{row}_{index:02d}';record['prepared_file']=name
        packed={}
        for with_map,key in [(False,'plain'),(True,'map')]:
            demo=demonstration(record,q,with_map)
            encoded=encode_trajectory(demo,TOKENIZER,CFG['maximum_sequence_tokens'])
            packed[key]=dict(encoded=encoded,steps=[t.step for t in demo.turns])
        torch.save(packed,OUT/'trajectories'/name)
        record['map_tokens']=len(packed['map']['encoded'].input_ids);record['plain_tokens']=len(packed['plain']['encoded'].input_ids)
        rows.append(record)
    return rows,dict(board_row=int(row),random_routes=number,eligible=len(candidates),unique_pairs=len(unique),unknown=unknown,seconds=time.monotonic()-started)

def main():
    p=argparse.ArgumentParser()
    for arg in ('config','sources','model-path','out'):p.add_argument('--'+arg,type=Path,required=True)
    p.add_argument('--workers',type=int,default=24);p.add_argument('--smoke-boards',type=int)
    a=p.parse_args();cfg=json.loads(a.config.read_text());src=a.sources
    boards=json.loads((src/'boards.json').read_text());source=json.loads((src/'source_manifest.json').read_text())
    refs=json.loads((src/'evaluation.json').read_text());test=[r for r in refs if r['split']=='test' and int(r['start'])!=int(r['goal'])]
    if len(test)!=510:raise ValueError('Expected the existing fixed 510 tasks')
    excluded={(int(r['start']),int(r['goal'])) for r in source['records'] if r['split']!='train'}
    from experiments.llm_task_sft.src.protocol import action_from_key
    from experiments.gcml_counterexamples.src import blocks as rules
    for record in refs:
        state,goal=int(record['start']),int(record['goal'])
        for key in record['reference_actions']:
            excluded.add((state,goal));state=rules.apply(state,action_from_key(key))
    if a.smoke_boards:boards=boards[:a.smoke_boards]
    a.out.mkdir(parents=True,exist_ok=True);(a.out/'trajectories').mkdir(exist_ok=True)
    atomic_json(a.out/'evaluation.json',test)
    rng=np.random.default_rng(cfg['seed']);indices=sorted(map(int,rng.choice(510,100,replace=False)))
    atomic_json(a.out/'probe_indices.json',indices)
    rows=[];audit=[];all_pairs=set();began=time.monotonic()
    with ProcessPoolExecutor(max_workers=a.workers,mp_context=mp.get_context('spawn'),initializer=init_worker,
        initargs=(cfg,str(src/'qmap.pt'),str(a.model_path),list(excluded),str(a.out))) as pool:
        for prepared,stats in pool.map(board_worker,boards,chunksize=1):
            rows.extend(prepared);audit.append(stats)
            all_pairs.update(tuple(p) for r in prepared for p in r['pairs'])
            atomic_json(a.out/'progress.json',dict(boards=len(audit),samples=len(rows),unique_pairs=len(all_pairs),last=stats,elapsed=time.monotonic()-began))
            print(json.dumps(dict(boards=len(audit),samples=len(rows),unique_pairs=len(all_pairs),last=stats)),flush=True)
    result=dict(config=cfg,rows=rows,boards=audit,unique_pairs=len(all_pairs),smoke=bool(a.smoke_boards))
    atomic_json(a.out/'dataset.json',result)
    if not a.smoke_boards and (len(rows)!=50000 or len(audit)!=1000 or len(all_pairs)<cfg['data']['coverage_target']):
        raise ValueError(f'Data coverage target not met: boards={len(audit)},rows={len(rows)},pairs={len(all_pairs)}')
    atomic_json(a.out/'ready.json',dict(samples=len(rows),unique_pairs=len(all_pairs),smoke=bool(a.smoke_boards)))

if __name__=='__main__':main()
