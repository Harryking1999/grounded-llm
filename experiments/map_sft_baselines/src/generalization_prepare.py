"""Prepare matched4/5/6-category50k studies using accepted task trajectories.

Keeps the existing longest/coverage and certified map-greedy data condition.
Physical queries, not candidate renumberings, define distinct demonstrations.
"""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
from pathlib import Path
import shutil
import numpy as np
import torch
from transformers import AutoTokenizer

from grounded_llm.core.files import atomic_json
from grounded_llm.core.sft.transcript import encode_trajectory
from grounded_llm.tasks.state_graph_sft import StateTask, Step, demonstration
from grounded_llm.tasks.text import section_text, prompt as full_prompt
from grounded_llm.tasks.blocks import rules

WORKER=None


def initial_text(row, env):
    task=row['task'];start=env.state(row['start']);goal=env.state(row['goal'])
    if task=='blocks':
        return section_text(task,'共享任务描述与规则').replace('{initial_board_10_rows}',env.render(start)).replace('{goal_board_10_rows}',env.render(goal))
    if task=='bridge':
        return section_text(task,'完整目标规则').format(people_count=env.env.n,
            crossing_times='\n'.join(f'{i}: {t}' for i,t in enumerate(env.env.times)),
            initial_configuration=env.render(start),goal_configuration=env.render(goal),action_limit=32)
    case=dict(task=task,initial=start,goal=goal,action_limit=32)
    if task=='puzzle8':case.update(initial=env.env.positions[start].tolist(),goal=env.env.positions[goal].tolist())
    if task=='path':case['neighbors']={str(i):np.flatnonzero(env.env.adjacency[i]).tolist() for i in range(env.env.size)}
    if task=='token_swapping':case['edges']=row['context']['edges']
    if task=='rush_hour':
        e=env.env
        fields=dict(goal_rule=section_text(task,'指定完整目标文字'),action_limit=32,
            vehicle_catalogue='\n'.join(f'{label}: {"horizontal" if horizontal else "vertical"}, length {size}' for label,horizontal,size,_ in e.pieces),
            initial_board=env.render(start),goal_board_section='\n\nGoal board:\n'+env.render(goal))
        return section_text(task,'共享任务描述与规则').format(**fields)
    if task=='game24':
        return section_text(task,'完整多重集目标规则').format(initial_numbers=env.render(start),goal_numbers=env.render(goal))
    return full_prompt(case).split('\n\n'+section_text(task,'SOTA 组／32B 组'))[0]


def certify(row, root, device='cpu'):
    """No teacher chooses the map's action. Reject a route at its first error."""
    env=StateTask(row,root,device);state=env.state(row['start']);goal=env.state(row['goal'])
    before=env.distance(state,goal)
    if not np.isfinite(before) or before<=0:return None
    steps=[];actions=[];seen={state}
    for _ in range(33):
        step=env.make_step(state,goal);steps.append(step)
        if state==goal:
            row=dict(row,reference_actions=actions,shortest_moves=len(actions),reference_cost=float(before),
                initial_text=initial_text(row,env),graph_sft=True)
            return row,steps
        options=list(env.transitions(state))
        if not options:return None
        index=min(range(len(options)),key=lambda i:options[i][2]+step.candidate_map_distances[i])
        action,child,cost=options[index]
        if not np.isclose(env.distance(state,goal),cost+env.distance(child,goal)):return None
        if child in seen:return None
        actions.append(action);state=child;seen.add(state)
    return None


def encode_job(item):
    row,steps=item;cfg,tokenizer,out=WORKER
    packed={};lengths={}
    for real,key in [(True,'map'),(False,'plain')]:
        demo=demonstration(row,steps,with_map=real)
        encoded=encode_trajectory(demo,tokenizer,cfg['maximum_sequence_tokens'])
        packed[key]=dict(encoded=encoded,steps=steps)
        lengths[key+'_tokens']=len(encoded.input_ids)
    name=row['trajectory_id']+'.pt';torch.save(packed,out/'trajectories'/name)
    return dict(row,prepared_file=row['task']+'/'+name,**lengths)


def block_states(original):
    state=int(original['start']);states=[state]
    if 'reference_actions' not in original:
        for local_id in original['route_ids']:
            index=rules.legal_indices(state)[int(local_id)-1]
            state=rules.apply(state,rules.PLACEMENTS[index][1]);states.append(state)
    else:
        for action in original['reference_actions']:
            if isinstance(action,str):
                from grounded_llm.tasks.blocks.protocol import action_from_key
                action=action_from_key(action)
            if isinstance(action,list):action=dict(zip(('shape_id','row','col'),action))
            state=rules.apply(state,action);states.append(state)
    if state!=0:raise ValueError('Source is not a complete clearing trajectory')
    return states


def block_variant(states,variant,cut=None):
    eight=states[-9]
    start,goal,operation=(eight,0,'remove') if variant=='clear8' else (states[0],0,'remove')
    if variant=='partial8':start,goal=eight,states[-9+(cut or 4)]
    if variant=='partial12_ood':start,goal=states[0],states[cut or max(1,len(states)//2)]
    return dict(task='blocks',variant=variant,start=str(start),goal=str(goal),operation=operation,
        map_checkpoint='qmap.pt',context={})


def teacher_record(row,root,device='cpu'):
    env=StateTask(row,root,device);state=env.state(row['start']);goal=env.state(row['goal'])
    before=env.distance(state,goal)
    if not np.isfinite(before) or state==goal:return None
    actions=[]
    while state!=goal:
        action,state,_=next((a,c,k) for a,c,k in env.transitions(state) if np.isclose(k+env.distance(c,goal),env.distance(state,goal)))
        actions.append(action)
    return dict(row,reference_actions=actions,reference_cost=float(before),shortest_moves=len(actions),
        initial_text=initial_text(row,env),graph_sft=True)


def block_queries(source, rng):
    meta=json.loads((source/'dataset.json').read_text())
    variants=['clear8','clear12','partial8']
    for index in rng.permutation(len(meta['rows'])):
        original=meta['rows'][int(index)];states=block_states(original)
        if len(states)<9:continue
        for variant in variants:
            for cut in (range(7,0,-1) if variant=='partial8' else [None]):
                yield dict(block_variant(states,variant,cut),source_trajectory=original['trajectory_id'])


def bridge_queries(rng, maximum_time):
    variants=['five_all_right','five_partial_goal','five_intermediate_start']
    while True:
        times=sorted(map(int,rng.choice(np.arange(1,maximum_time+1),5,replace=False)))
        for variant in variants:
            goal=63 if variant=='five_all_right' else int(rng.integers(1,63))
            start=0 if variant!='five_intermediate_start' else int(rng.integers(1,63))
            yield dict(task='bridge',variant=variant,start=start,goal=goal,map_checkpoint='qmap.pt',context=dict(times=times))


def token_queries(rng, tree):
    while True:
        for variant in ['original_tree_home','original_tree_arbitrary']:
            yield dict(task='token_swapping',variant=variant,start=rng.permutation(10).tolist(),
                goal=list(range(10)) if variant.endswith('home') else rng.permutation(10).tolist(),
                map_checkpoint='qmap.pt',context=dict(edges=tree['edges'],oracle='distances.bin'))


def existing_queries(task,source,rng):
    rows=[r for r in json.loads((source/'dataset.json').read_text())['rows'] if r.get('task',task)==task]
    for index in rng.permutation(len(rows)):
        row=rows[int(index)]
        yield dict(task=task,variant='fixed_graph' if task=='path' else 'restoration',
            start=row['start'],goal=row['goal'],map_checkpoint='qmap.pt',context=dict(oracle='oracle.npy'),
            source_trajectory=row['trajectory_id'])


def query_key(row):
    return json.dumps([row['task'],row.get('operation'),row.get('context'),row['start'],row['goal']],sort_keys=True)


def evaluation_subset(rows, count, seed):
    """Fixed proportional variant/route-length subset, without model scores."""
    if not 0<count<=len(rows):raise ValueError('Evaluation subset exceeds its frozen source pool')
    if len({r['trajectory_id'] for r in rows})!=len(rows):raise ValueError('Duplicate evaluation IDs')
    if count==len(rows):return list(rows)
    tasks={r['task'] for r in rows}
    if len(tasks)!=1:raise ValueError('Select within one task at a time')
    rng=np.random.default_rng(np.random.SeedSequence([seed,*map(ord,next(iter(tasks)))]))

    def quotas(groups, total):
        keys=sorted(groups);sizes=np.asarray([len(groups[k]) for k in keys])
        expected=sizes*total/sizes.sum();assigned=np.floor(expected).astype(int)
        tie=rng.random(len(keys))
        order=sorted(range(len(keys)),key=lambda i:(-(expected[i]-assigned[i]),tie[i]))
        for i in order[:total-int(assigned.sum())]:assigned[i]+=1
        return dict(zip(keys,map(int,assigned)))

    variants={}
    for i,row in enumerate(rows):variants.setdefault(row['variant'],[]).append(i)
    selected=[]
    for variant,quota in quotas(variants,count).items():
        lengths={}
        for i in variants[variant]:
            row=rows[i];length=int(row.get('shortest_moves',len(row['reference_actions'])))
            lengths.setdefault(length,[]).append(i)
        for length,n in quotas(lengths,quota).items():
            indices=sorted(lengths[length],key=lambda i:rows[i]['trajectory_id'])
            selected.extend(map(int,rng.choice(indices,n,replace=False)))
    return [rows[i] for i in sorted(selected)]


def prepare_task(task,cfg,sources,pool,device,workers,smoke=0):
    global WORKER
    out=pool/task
    if (out/'ready.json').exists():return
    out.mkdir(parents=True,exist_ok=True);(out/'trajectories').mkdir(exist_ok=True)
    rng=np.random.default_rng(np.random.SeedSequence([cfg['seed'],cfg['task_pool'].index(task)]))
    selected=sources['maps'][task];shutil.copy2(selected,out/'qmap.pt')
    source=Path(sources['existing'][task]) if task in sources['existing'] else None
    if task in ('cube2','puzzle8','path'):
        for name in ['oracle.npy','graph.npz']:
            p=source/name
            if not p.exists():p=source/task/name
            if p.exists() and not (out/name).exists():(out/name).symlink_to(p.resolve())
    if task=='token_swapping':
        (out/'distances.bin').symlink_to(Path(sources['token_oracle']).resolve())
    if task=='blocks':
        candidates=block_queries(source,rng);variants=['clear8','clear12','partial8']
    elif task=='bridge':
        candidates=bridge_queries(rng,cfg['bridge_maximum_time']);variants=['five_all_right','five_partial_goal','five_intermediate_start']
    elif task=='token_swapping':
        candidates=token_queries(rng,json.loads(Path(sources['token_tree']).read_text()));variants=['original_tree_home','original_tree_arbitrary']
    else:candidates=existing_queries(task,source,rng);variants=['fixed_graph' if task=='path' else 'restoration']
    requested=max(x['counts'].get(task,0) for x in cfg['cohorts'])
    count=smoke if smoke else requested
    quotas={v:count//len(variants)+(i<count%len(variants)) for i,v in enumerate(variants)}
    protected=[];small_ood=[]
    evaluation_count=cfg.get('evaluation_source_per_group',cfg['evaluation_per_group'])
    if source and (source/'evaluation.json').exists():
        for row in json.loads((source/'evaluation.json').read_text()):
            if row.get('task',task)!=task:continue
            if task=='blocks':
                states=block_states(row)
                r=block_variant(states,variants[len(protected)%len(variants)],1+(len(protected)//len(variants))%7)
                r.update(cohort='id',trajectory_id=f'blocks_id_{len(protected):04d}')
                r=teacher_record(r,out,device);protected.append(r)
                cut=max(1,round((len(states)-1)*[.25,.5,.75][len(small_ood)%3]))
                o=block_variant(states,'partial12_ood',cut)
                o.update(cohort='small_ood',trajectory_id=f'blocks_small_ood_{len(small_ood):04d}')
                small_ood.append(teacher_record(o,out,device))
                if len(protected)==evaluation_count:break
                continue
            r=dict(task=task,variant='id',cohort='id',start=row['start'],goal=row['goal'],
                map_checkpoint='qmap.pt',context={} if task=='blocks' else dict(oracle='oracle.npy'),
                trajectory_id=f'{task}_id_{len(protected):04d}',reference_actions=row['reference_actions'],graph_sft=True)
            env=StateTask(r,out,device);r['initial_text']=initial_text(r,env);protected.append(r)
            if len(protected)==evaluation_count:break
    excluded={(str(r['start']),str(r['goal'])) for r in protected+small_ood};seen=set();rows=[];stats=Counter();attempts=0
    if task=='token_swapping':
        while len(protected)<evaluation_count:
            r=next(candidates);r=teacher_record(r,out,device)
            if r is None:continue
            r.update(trajectory_id=f'token_swapping_id_{len(protected):04d}',cohort='id')
            protected.append(r);excluded.add((str(r['start']),str(r['goal'])))
    tokenizer=AutoTokenizer.from_pretrained(sources['tokenizer'],local_files_only=True)
    WORKER=cfg,tokenizer,out
    # GPU map generation stays in the parent; CPU tokenization uses bounded futures.
    pending=[]
    with ProcessPoolExecutor(workers,mp_context=multiprocessing.get_context('fork')) as executor:
        for candidate in candidates:
            variant=candidate['variant']
            if stats[variant]>=quotas[variant]:continue
            key=query_key(candidate)
            if key in seen or (str(candidate['start']),str(candidate['goal'])) in excluded:continue
            attempts+=1
            result=certify(candidate,out,device)
            if result is None:
                if attempts%100==0:atomic_json(out/'progress.json',dict(accepted=dict(stats),required=quotas,attempts=attempts))
                if attempts>cfg['maximum_candidate_queries']:raise ValueError('Insufficient certified distinct queries; do not duplicate or relax criteria')
                continue
            row,steps=result;seen.add(key);stats[variant]+=1
            if any((str(s.current),str(s.goal)) in excluded for s in steps[:-1]):
                seen.remove(key);stats[variant]-=1;continue
            row['trajectory_id']=f'{task}_train_{sum(stats.values()):05d}'
            pending.append(executor.submit(encode_job,(row,steps)))
            if len(pending)>=workers*2:rows.append(pending.pop(0).result())
            if sum(stats.values())%100==0:atomic_json(out/'progress.json',dict(accepted=dict(stats),required=quotas,attempts=attempts))
            if stats==Counter(quotas):break
        rows.extend(f.result() for f in pending)
    if stats!=Counter(quotas):raise ValueError(f'{task}: insufficient unique accepted data {stats} vs {quotas}')
    if task=='bridge':
        # Evaluation times and queries are fixed before mixing; never Q-success filter evaluation.
        train_times={tuple(r['context']['times']) for r in rows}
        protected_keys={query_key(r) for r in protected}
        while len(protected)<evaluation_count:
            r=next(candidates)
            key=query_key(r)
            if tuple(r['context']['times']) in train_times or key in protected_keys:continue
            env=StateTask(r,out,device);start=env.state(r['start']);goal=env.state(r['goal'])
            if start==goal or not np.isfinite(env.distance(start,goal)):continue
            actions=[];state=start
            while state!=goal:
                action,state,_=next((a,c,k) for a,c,k in env.transitions(state) if np.isclose(k+env.distance(c,goal),env.distance(state,goal)))
                actions.append(action)
            r.update(trajectory_id=f'bridge_id_{len(protected):04d}',cohort='id',reference_actions=actions,graph_sft=True,
                initial_text=initial_text(r,env));protected.append(r)
            protected_keys.add(key)
    atomic_json(out/'dataset.json',dict(task=task,rows=rows,response_order='ranking_first'))
    atomic_json(out/'evaluation.json',protected);atomic_json(out/'probe_indices.json',[])
    if small_ood:atomic_json(out/'small_ood.json',small_ood)
    atomic_json(out/'acceptance.json',dict(samples=len(rows),by_subclass=dict(stats),candidate_queries=attempts,
        unique_queries=len(seen),physical_shortest_map_greedy_verified=True,evaluation_filter='random reachable, never Q success',
        by_subclass_length={v:dict(Counter(r['shortest_moves'] for r in rows if r['variant']==v)) for v in variants},
        limitation='Qmap exposure and LLM exposure are distinct; fixed selected maps can be weak outside their accepted scope.'))
    atomic_json(out/'ready.json',dict(samples=len(rows),smoke=bool(smoke)))


def assemble(cfg,pool,out,smoke=0):
    for study in cfg['cohorts']:
        if not all((pool/t/'ready.json').exists() for t in study['tasks']):continue
        target=out/study['name']
        if (target/'ready.json').exists():continue
        target.mkdir(parents=True,exist_ok=True);all_rows=[];evaluation=[]
        for task in study['tasks']:
            bank=pool/task;meta=json.loads((bank/'dataset.json').read_text())
            count=smoke if smoke else study['counts'][task]
            # Keep each subclass balanced when reducing the shared bank.
            groups={v:[r for r in meta['rows'] if r['variant']==v] for v in sorted({r['variant'] for r in meta['rows']})}
            selected=[]
            while len(selected)<count:
                for variant in groups:
                    if groups[variant] and len(selected)<count:selected.append(groups[variant].pop(0))
            (target/'trajectories'/task).mkdir(parents=True,exist_ok=True)
            for row in selected:
                name=Path(row['prepared_file']).name
                p=target/'trajectories'/task/name
                if not p.exists():p.symlink_to((bank/'trajectories'/name).resolve())
            task_out=target/task;task_out.mkdir(exist_ok=True)
            for asset in bank.iterdir():
                if asset.name in ('qmap.pt','oracle.npy','graph.npz','distances.bin') and not (task_out/asset.name).exists():(task_out/asset.name).symlink_to(asset.resolve())
            records=evaluation_subset(json.loads((bank/'evaluation.json').read_text()),
                cfg['evaluation_per_group'],cfg['evaluation_subset']['seed']);evaluation.extend(records)
            atomic_json(task_out/'evaluation.json',records);atomic_json(task_out/'probe_indices.json',[])
            all_rows.extend(selected)
        atomic_json(target/'dataset.json',dict(task='multitask',rows=all_rows,response_order='ranking_first'))
        atomic_json(target/'evaluation.json',evaluation);atomic_json(target/'probe_indices.json',[])
        atomic_json(target/'acceptance.json',dict(samples=len(all_rows),by_task=dict(Counter(r['task'] for r in all_rows)),
            by_subclass=dict(Counter(r['task']+'/'+r['variant'] for r in all_rows)),shared_task_pool_across_training_sizes=True))
        atomic_json(target/'ready.json',dict(samples=len(all_rows),smoke=bool(smoke)))


def main():
    p=argparse.ArgumentParser(__doc__)
    for name in ['config','sources','out']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--device',default='cpu');p.add_argument('--workers',type=int,default=8);p.add_argument('--smoke',type=int,default=0)
    p.add_argument('--tasks',nargs='+',help='Small validation only; formal preparation uses the configured order')
    args=p.parse_args();cfg=json.loads(args.config.read_text());sources=json.loads(args.sources.read_text())
    if args.smoke:cfg=dict(cfg,evaluation_per_group=min(args.smoke,cfg['evaluation_per_group']),
        evaluation_source_per_group=min(args.smoke,cfg.get('evaluation_source_per_group',cfg['evaluation_per_group'])))
    torch.set_num_threads(1);pool=args.out/'pool';pool.mkdir(parents=True,exist_ok=True)
    if args.tasks and not args.smoke:raise ValueError('Formal preparation must include all configured tasks')
    for task in args.tasks or cfg['preparation_order']:
        prepare_task(task,cfg,sources,pool,args.device,args.workers,args.smoke)
        assemble(cfg,pool,args.out,args.smoke)


if __name__=='__main__':main()
