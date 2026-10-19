"""Nest the accepted 50k routes in 100k using new certified physical routes."""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing as mp
from pathlib import Path
import shutil
import time

import numpy as np
import torch

from experiments.blocks_distance_map.src.multiboard_data import sample_paths
from experiments.blocks_distance_map.src.oracle import DistanceOracle, OracleBudgetExceeded
from experiments.flamingo_map_reader.src.blocks import FrozenBoardMap
from experiments.flamingo_map_reader.src.transcript import encode_trajectory
from .common import atomic_json, demonstration, sorted_step
from .prepare import CachedMap, choose
from .leakage_audit import reference_path


def heldout_pairs(source_manifest,references):
    excluded={(int(r['start']),int(r['goal'])) for r in source_manifest['records'] if r['split']!='train'}
    for row in references:
        path=reference_path(row)
        excluded.update((state,int(row['goal'])) for state in path)
    return excluded


def select_additional(candidates,original,rng):
    quota=sum(bool(row['shortest_moves']) for row in original)
    if len(candidates)<quota:return None
    seen={(int(s),int(g)) for row in original for s,g in row['pairs']}
    return choose(candidates,quota,seen,rng,prior=[row for row in original if row['shortest_moves']])


def initialize(config,source,out,model_path,excluded,initial_pairs,terminal_states):
    global CFG,SOURCE,OUT,QMAP,TOKENIZER,EXCLUDED,INITIAL,TERMINAL
    from transformers import AutoTokenizer
    torch.set_num_threads(1)
    CFG=config;SOURCE=Path(source);OUT=Path(out)
    QMAP=FrozenBoardMap.load(SOURCE/'qmap.pt')
    TOKENIZER=AutoTokenizer.from_pretrained(model_path,local_files_only=True)
    EXCLUDED=set(map(tuple,excluded));INITIAL=set(map(tuple,initial_pairs));TERMINAL=set(terminal_states)


def board_worker(job):
    board,original=job
    row,mask,reference=board;board=(row,int(mask),reference)
    rng=np.random.default_rng(CFG['seed']+int(row));q=CachedMap(QMAP)
    oracle=DistanceOracle(cache_limit=CFG['data']['oracle_cache_limit'],seconds=CFG['data']['oracle_seconds_per_board'])
    paths=sample_paths(board,0,rng);evaluated=set();candidates=[];unknown=0;began=time.monotonic();selected=None
    for number in CFG['data']['random_routes']:
        previous=len(paths)-1
        paths.extend(sample_paths(board,number-previous,rng)[1:])
        pairs={(s,g) for path in paths for i,s in enumerate(path[:-1]) for g in path[i+1:]}
        for start,goal in sorted(pairs-evaluated-EXCLUDED-INITIAL):
            evaluated.add((start,goal))
            try:d=oracle.distance(start,goal)
            except OracleBudgetExceeded:unknown+=1;continue
            if d<CFG['data']['minimum_shortest_moves']:continue
            state=start;route=[];visited=[]
            for index in range(d):
                if (state,goal) in EXCLUDED:break
                visited.append((state,goal));step=sorted_step(q,state,goal)
                if not step.map_minimal_candidates:break
                candidate=step.map_minimal_candidates[0];_,after=step.execute(candidate)
                try:remaining=oracle.distance(after,goal)
                except OracleBudgetExceeded:unknown+=1;break
                if remaining!=d-index-1:break
                route.append(candidate);state=after
            if state==goal and len(route)==d:
                candidates.append(dict(start=str(start),goal=str(goal),shortest_moves=d,
                    route_ids=route,pairs=visited,board_row=int(row)))
        selected=select_additional(candidates,original,rng)
        if selected is not None:break
    if selected is None:
        raise ValueError(f'Board {row}: {len(candidates)} new certified routes, need {sum(bool(r["shortest_moves"]) for r in original)} after {number} sampled routes')
    terminal_candidates=sorted({s for r in selected for s,g in r['pairs'][1:]
                                if s not in TERMINAL and (s,s) not in EXCLUDED},
                               key=lambda state:(-state.bit_count(),state))
    if not terminal_candidates:raise ValueError(f'Board {row}: no distinct terminal example')
    terminal=terminal_candidates[0]
    selected.append(dict(start=str(terminal),goal=str(terminal),shortest_moves=0,
                         route_ids=[],pairs=[],board_row=int(row)))
    rows=[]
    for index,record in enumerate(selected):
        name=f'additional_{row}_{index:02d}.pt'
        record.update(trajectory_id=f'additional_{row}_{index:02d}',prepared_file=name)
        demo=demonstration(record,q,True,'ranking_first')
        encoded=encode_trajectory(demo,TOKENIZER,CFG['maximum_sequence_tokens'])
        torch.save(dict(map=dict(encoded=encoded,steps=[turn.step for turn in demo.turns])),OUT/'trajectories'/name)
        record['map_tokens']=len(encoded.input_ids);rows.append(record)
    return rows,dict(board_row=int(row),random_routes=number,eligible=len(candidates),
                     unknown=unknown,seconds=time.monotonic()-began)


def main():
    parser=argparse.ArgumentParser()
    for key in ('config','source-data','sources','model-path','out'):
        parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--workers',type=int,default=24)
    parser.add_argument('--smoke-boards',type=int)
    args=parser.parse_args();cfg=json.loads(args.config.read_text())
    meta=json.loads((args.source_data/'dataset.json').read_text())
    if len(meta['rows'])!=50000 or meta['response_order']!='ranking_first':
        raise ValueError('Expected the accepted ranking-first 50k physical-route source')
    by_board=defaultdict(list)
    for row in meta['rows']:by_board[row['board_row']].append(row)
    boards=json.loads((args.sources/'boards.json').read_text())
    if len(boards)!=1000 or any(len(v)!=50 for v in by_board.values()):raise ValueError('Expected 1000 boards with 50 accepted routes each')
    refs=json.loads((args.sources/'evaluation.json').read_text())
    manifest=json.loads((args.sources/'source_manifest.json').read_text())
    excluded=heldout_pairs(manifest,refs)
    initial={(int(r['start']),int(r['goal'])) for r in meta['rows']}
    terminals={int(r['goal']) for r in meta['rows']}
    if args.smoke_boards:boards=boards[:args.smoke_boards]
    args.out.mkdir(parents=True,exist_ok=False);(args.out/'trajectories').mkdir()
    rows=[];audit=[];began=time.monotonic()
    with ProcessPoolExecutor(max_workers=args.workers,mp_context=mp.get_context('spawn'),initializer=initialize,
            initargs=(cfg,str(args.sources),str(args.out),str(args.model_path),list(excluded),list(initial),list(terminals))) as pool:
        for prepared,stats in pool.map(board_worker,[(board,by_board[board[0]]) for board in boards],chunksize=1):
            original=by_board[stats['board_row']]
            for record in original:
                (args.out/'trajectories'/record['prepared_file']).symlink_to((args.source_data/'trajectories'/record['prepared_file']).resolve())
            rows.extend(original);rows.extend(prepared);audit.append(stats)
            atomic_json(args.out/'progress.json',dict(boards=len(audit),samples=len(rows),last=stats,elapsed=time.monotonic()-began))
            print(json.dumps(dict(boards=len(audit),samples=len(rows),last=stats)),flush=True)
    keys=[(int(r['start']),int(r['goal'])) for r in rows]
    if len(keys)!=len(set(keys)):raise ValueError('Duplicate physical initial state-goal pairs in expanded corpus')
    if not args.smoke_boards and (len(rows)!=cfg['data']['samples_per_epoch'] or len(rows)!=100000):
        raise ValueError('Formal expanded corpus must contain exactly 100k routes')
    for name in ('evaluation.json','probe_indices.json'):shutil.copy2(args.source_data/name,args.out/name)
    atomic_json(args.out/'dataset.json',dict(config=cfg,rows=rows,boards=audit,
        response_order='ranking_first',source_data=str(args.source_data),nested_source_samples=50000,
        smoke=bool(args.smoke_boards),lengths=dict(Counter(r['shortest_moves'] for r in rows))))
    atomic_json(args.out/'ready.json',dict(samples=len(rows),smoke=bool(args.smoke_boards),response_order='ranking_first'))


if __name__=='__main__':main()
