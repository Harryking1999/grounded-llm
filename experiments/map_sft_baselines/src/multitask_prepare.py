"""Four matched task cohorts; existing data are uniformly subsampled."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import heapq
import json
import multiprocessing
from pathlib import Path
import shutil
import numpy as np
import torch
from transformers import AutoTokenizer
from grounded_llm.core.files import atomic_json
from grounded_llm.core.sft.transcript import encode_trajectory
from grounded_llm.tasks.restoration_sft import environment, demonstration, make_step
from grounded_llm.tasks.path.reader import FrozenPathMap
from grounded_llm.tasks.path.environment import action_catalog

WORKER = None


def encode_row(item):
    task, index, record, payload = item
    cfg, tokenizer, envs, out, path_tables = WORKER
    if isinstance(payload, str):
        steps = torch.load(payload,map_location='cpu',weights_only=False)['map']['steps']
    else:
        vectors, distances = path_tables
        steps = [make_step(envs[task],None,s,record['goal'],vectors=vectors,
                          distances=distances[:,record['goal']]) for s in payload]
    name = f'{task}/{index:05d}.pt'
    packed, tokens = {}, {}
    for real_map, key in [(True, 'map'), (False, 'plain')]:
        demo = demonstration(task, envs[task], record, steps, with_map=real_map)
        encoded = encode_trajectory(demo, tokenizer, cfg['maximum_sequence_tokens'])
        packed[key] = dict(encoded=encoded, steps=[turn.step for turn in demo.turns])
        tokens[key] = len(encoded.input_ids)
    torch.save(packed, out/'trajectories'/name)
    return dict(record, task=task, prepared_file=name, **{k+'_tokens':v for k,v in tokens.items()})


def path_records(env, qmap, count, heldout, seed):
    """Longest certified map-greedy routes, then new state-goal coverage."""
    rng = np.random.default_rng(seed)
    q = qmap.encode(np.arange(env.size))
    distances = np.stack([qmap.distance(q, q[g]).numpy() for g in range(env.size)], axis=1)
    rows = []
    for goal in range(env.size):
        policy = {}
        for state in range(env.size):
            actions = np.flatnonzero(env.next[state] >= 0)
            children = env.next[state, actions]
            policy[state] = int(actions[np.argmin(distances[children, goal])])
        for start in range(env.size):
            if start == goal:
                continue
            state, actions, states = start, [], [start]
            while state != goal:
                if (state, goal) in heldout:
                    break
                action = policy[state]
                child = int(env.next[state, action])
                if env.distance(child, goal) != env.distance(state, goal)-1:
                    break
                actions.append(env.actions[action]); states.append(child); state = child
            if state == goal:
                rows.append(dict(start=start, goal=goal, reference_actions=actions,
                    shortest_moves=len(actions), pairs=[(s,goal) for s in states[:-1]], states=states))
    pool = [(-r['shortest_moves'], -len(r['pairs']), float(rng.random()), i)
            for i,r in enumerate(rows)]
    heapq.heapify(pool)
    selected, covered = [], set()
    while pool and len(selected) < count:
        length, previous, tie, i = heapq.heappop(pool)
        pairs = set(rows[i]['pairs']); gain = -len(pairs-covered)
        if gain != previous:
            heapq.heappush(pool,(length,gain,tie,i)); continue
        selected.append(rows[i]); covered.update(pairs)
    if len(selected) != count:
        raise ValueError(f'Only {len(selected)} qualified path routes, need {count}')
    return selected, covered, q, distances


def main():
    global WORKER
    parser = argparse.ArgumentParser(__doc__)
    for name in ('config','sources','model-path','out'):
        parser.add_argument('--'+name, required=True, type=Path)
    parser.add_argument('--workers', type=int, default=16)
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text()); sources = json.loads(args.sources.read_text())
    out = args.out
    if out.exists():
        raise ValueError('Use a fresh dataset output')
    out.mkdir(parents=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path,local_files_only=True)
    torch.set_num_threads(2)
    rng = np.random.default_rng(cfg['data']['subset_seed'])
    count = cfg['data']['samples_per_task']; rows, all_eval, audit, envs = [], [], {}, {}
    for task in cfg['data']['tasks']:
        (out/task).mkdir(); (out/'trajectories'/task).mkdir(parents=True)
        if task == 'path':
            continue
        source = Path(sources[task]); meta = json.loads((source/'dataset.json').read_text())
        chosen = sorted(map(int,rng.choice(len(meta['rows']),count,replace=False)))
        selected = [dict(meta['rows'][i],task=task) for i in chosen]
        evaluation = json.loads((source/'evaluation.json').read_text())
        indices = sorted(map(int,rng.choice(len(evaluation),cfg['evaluation']['tasks_per_task'],replace=False)))
        evaluation = [dict(evaluation[i],task=task) for i in indices]
        atomic_json(out/task/'evaluation.json',evaluation); all_eval.extend(evaluation)
        atomic_json(out/task/'probe_indices.json',[])
        for filename in ['qmap.pt','oracle.npy']:
            origin = Path(sources['blocks_qmap']) if task=='blocks' and filename=='qmap.pt' else source/filename
            if origin.exists():
                (out/task/filename).symlink_to(origin.resolve())
        if task == 'blocks':
            for row in selected:
                original = source/'trajectories'/row['prepared_file']
                name = f"blocks/{row['prepared_file']}"
                (out/'trajectories'/name).symlink_to(original.resolve())
                rows.append(dict(row,prepared_file=name))
        else:
            envs[task] = environment(task, out/task/'oracle.npy')
            # Existing map encodings and physical steps are reused below.
        audit[task] = dict(source=str(source),source_samples=len(meta['rows']),samples=count,
                           selection='uniform_without_replacement',source_indices=chosen)
        atomic_json(out/'progress.json',dict(stage='selected_existing',task=task))
        if task != 'blocks':
            WORKER = cfg,tokenizer,envs,out,None
            def items():
                for index,row in enumerate(selected):
                    yield task,index,row,str(source/'trajectories'/row['prepared_file'])
            with ProcessPoolExecutor(args.workers,mp_context=multiprocessing.get_context('fork')) as pool:
                for i,row in enumerate(pool.map(encode_row,items(),chunksize=8),1):
                    rows.append(row)
                    if i%250==0:
                        atomic_json(out/'progress.json',dict(stage='encode_existing',task=task,done=i,total=count))
    task='path'; qmap=FrozenPathMap.load(Path(sources['path_qmap']))
    adjacency=qmap.adjacency;actions,_=action_catalog(adjacency)
    np.savez(out/task/'graph.npz',adjacency=adjacency,actions=actions)
    shutil.copy2(sources['path_qmap'],out/task/'qmap.pt')
    env=environment(task,out/task/'oracle.npy');envs[task]=env
    options=np.argwhere(~np.eye(env.size,dtype=bool))
    chosen=rng.choice(len(options),cfg['evaluation']['tasks_per_task'],replace=False)
    heldout={tuple(map(int,options[i])) for i in chosen};evaluation=[]
    for index,(start,goal) in enumerate(sorted(heldout)):
        state=start;physical=[]
        while state!=goal:
            legal=np.flatnonzero(env.next[state]>=0)
            action=next(int(a) for a in legal if env.distance(env.next[state,a],goal)==env.distance(state,goal)-1)
            physical.append(env.actions[action]);state=int(env.next[state,action])
        evaluation.append(dict(trajectory_id=f'path_eval_{index:04d}',task=task,start=start,goal=goal,
                               reference_actions=physical,shortest_moves=len(physical)))
    atomic_json(out/task/'evaluation.json',evaluation);atomic_json(out/task/'probe_indices.json',[])
    all_eval.extend(evaluation)
    selected,covered,vectors,distances=path_records(env,qmap,count,heldout,cfg['data']['subset_seed'])
    assert not covered & heldout
    WORKER=cfg,tokenizer,envs,out,(vectors.numpy(),distances)
    def items():
        for index,row in enumerate(selected):
            record=dict(row,trajectory_id=f'path_train_{index:05d}')
            yield task,index,record,record['states']
    with ProcessPoolExecutor(args.workers,mp_context=multiprocessing.get_context('fork')) as pool:
        for i,row in enumerate(pool.map(encode_row,items(),chunksize=8),1):
            rows.append(row)
            if i%250==0:atomic_json(out/'progress.json',dict(stage='encode_path',done=i,total=count))
    audit[task]=dict(samples=count,unique_state_goal_pairs=len(covered),evaluation_overlap=0,
        selection='longest_then_new_state_goal_coverage',map_shortest_verified=True,
        graph='adopted fixed graph, seen nodes, new SFT relation holdout; Qmap exposure separate')
    assert len(rows)==cfg['data']['samples_per_epoch']
    assert dict(Counter(r['task'] for r in rows))=={t:count for t in cfg['data']['tasks']}
    assert all((out/'trajectories'/r['prepared_file']).is_file() for r in rows)
    for task in cfg['data']['tasks']:
        task_rows=[r for r in rows if r['task']==task]
        audit[task]['length_histogram']=dict(Counter(r.get('shortest_moves',r.get('length')) for r in task_rows))
        audit[task]['maximum_map_tokens']=max(r.get('map_tokens',r.get('tokens',0)) for r in task_rows)
        audit[task]['maximum_plain_tokens']=max(r.get('plain_tokens',0) for r in task_rows)
    atomic_json(out/'evaluation.json',all_eval);atomic_json(out/'probe_indices.json',[])
    atomic_json(out/'dataset.json',dict(task='multitask',response_order='ranking_first',rows=rows))
    atomic_json(out/'acceptance.json',dict(tasks=audit,total=len(rows),
        limitation='SFT holdouts do not remove Qmap exposure or earlier task/model selection.'))
    atomic_json(out/'ready.json',dict(samples=len(rows),smoke=False))
    atomic_json(out/'progress.json',dict(stage='completed',samples=len(rows)))


if __name__=='__main__':
    main()
