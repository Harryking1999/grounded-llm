"""Freeze final-only subclass and whole-category OOD cohorts, without Q filtering."""
import argparse
import json
from pathlib import Path
import shutil
import numpy as np
from grounded_llm.core.files import atomic_json
from .generalization_prepare import teacher_record
from grounded_llm.tasks.state_graph_sft import StateTask


def link_assets(source,target):
    target.mkdir(parents=True,exist_ok=True)
    for asset in source.iterdir():
        if asset.name in ('qmap.pt','qmap6.pt','alternate.pt','oracle.npy','graph.npz','distances.bin','alternate_distances.bin'):
            p=target/asset.name
            if not p.exists():p.symlink_to(asset.resolve())
    if (source/'new_graph').exists() and not (target/'new_graph').exists():(target/'new_graph').symlink_to((source/'new_graph').resolve())
    if (source/'rush_oracles').exists() and not (target/'rush_oracles').exists():(target/'rush_oracles').symlink_to((source/'rush_oracles').resolve())


def prepare(cfg,sources,data_root):
    rng=np.random.default_rng(cfg['seed']+1);pool=data_root/'pool';count=cfg['evaluation_per_group'];cohorts={};small={}
    for task in cfg['task_pool']:
        bank=pool/task
        if task in ('rush_hour','game24'):
            bank.mkdir(parents=True,exist_ok=True)
            shutil.copy2(sources['maps'][task],bank/'qmap.pt')
            if task=='rush_hour' and not (bank/'rush_oracles').exists():
                (bank/'rush_oracles').symlink_to((Path(sources['ood_queries'][task]).parent/'rush_oracles').resolve())
            rows=json.loads(Path(sources['ood_queries'][task]).read_text())
            if len(rows)<count:raise ValueError(f'{task}: insufficient independent OOD queries')
            chosen=rng.choice(len(rows),count,replace=False)
            cohorts[task]=[]
            for i in chosen:
                row=teacher_record(rows[int(i)],bank)
                if row is None:raise ValueError('Declared reachable whole-category query is not reachable')
                row.update(trajectory_id=f'{task}_big_ood_{len(cohorts[task]):04d}',cohort='big_ood')
                cohorts[task].append(row)
            atomic_json(bank/'big_ood.json',cohorts[task]);continue
        cohorts[task]=json.loads((bank/'evaluation.json').read_text())
        if len(cohorts[task])!=count:raise ValueError('ID cohort size differs from the contract')
        if task=='blocks':small[task]=json.loads((bank/'small_ood.json').read_text());continue
        small[task]=[]
        for original in cohorts[task]:
            row=dict(original,cohort='small_ood',trajectory_id=original['trajectory_id'].replace('_id_','_small_ood_'))
            if task in ('cube2','puzzle8'):
                # Target pool is absent from SFT target supervision; input exposure is separate.
                row['goal']=int(rng.integers(1,1000))
                row['variant']='noncanonical_goal_ood'
            elif task=='bridge':
                if not (bank/'qmap6.pt').exists():shutil.copy2(sources['bridge6_map'],bank/'qmap6.pt')
                row.update(context={'times':sorted(map(int,rng.choice(np.arange(1,cfg['bridge_maximum_time']+1),6,replace=False)))},
                    start=0,goal=int(rng.integers(1,127)),map_checkpoint='qmap6.pt',variant='six_people_ood')
            elif task=='token_swapping':
                if not (bank/'alternate.pt').exists():shutil.copy2(sources['token_alternate_map'],bank/'alternate.pt')
                if not (bank/'alternate_distances.bin').exists():(bank/'alternate_distances.bin').symlink_to(Path(sources['token_alternate_oracle']).resolve())
                tree=json.loads(Path(sources['token_alternate_tree']).read_text())
                row.update(context=dict(edges=tree['edges'],oracle='alternate_distances.bin'),map_checkpoint='alternate.pt',variant='alternate_tree_ood')
            elif task=='path':
                graph=bank/'new_graph';graph.mkdir(exist_ok=True)
                for source,name in [(sources['path_ood_map'],'qmap.pt'),(sources['path_ood_graph'],'graph.npz')]:
                    if not (graph/name).exists():(graph/name).symlink_to(Path(source).resolve())
                row.update(context={'oracle':'new_graph/oracle.npy'},map_checkpoint='new_graph/qmap.pt',variant='new_graph_ood')
            result=teacher_record(row,bank)
            if result is None:raise ValueError('Invalid frozen subclass query; do not Q-success filter or silently replace')
            small[task].append(result)
        atomic_json(bank/'small_ood.json',small[task])
    for study in cfg['cohorts']:
        target=data_root/study['name']/'final_evaluation';target.mkdir(exist_ok=True)
        all_rows=[]
        for task in cfg['task_pool']:
            rows=[dict(r,cohort='id' if task in study['tasks'] else 'big_ood') for r in cohorts[task]]
            if task in study['tasks']:rows+=small[task]
            task_out=target/task;link_assets(pool/task,task_out)
            atomic_json(task_out/'evaluation.json',rows);atomic_json(task_out/'probe_indices.json',[])
            all_rows+=rows
        atomic_json(target/'evaluation.json',all_rows);atomic_json(target/'probe_indices.json',[])
        atomic_json(target/'ready.json',dict(tasks=len(all_rows),smoke=False,final_only=True,
            filter='Random reachable physical queries, no Q-success filtering; Qmap exposure separate from LLM exposure.'))


def main():
    p=argparse.ArgumentParser(__doc__)
    for name in ['config','sources','data']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--map-reference',action='store_true')
    a=p.parse_args();cfg=json.loads(a.config.read_text())
    prepare(cfg,json.loads(a.sources.read_text()),a.data)
    if a.map_reference:
        from collections import defaultdict
        import torch
        torch.set_num_threads(1)
        for study in cfg['cohorts']:
            root=a.data/study['name']/'final_evaluation';groups=defaultdict(lambda:dict(count=0,solved=0,optimal=0,cycles=0,excess_cost_on_success=0.0))
            for row in json.loads((root/'evaluation.json').read_text()):
                env=StateTask(row,root/row['task']);state=env.state(row['start']);goal=env.state(row['goal']);cost=0;visited={state};cycle=False
                for _ in range(32):
                    if state==goal:break
                    step=env.make_step(state,goal);options=list(env.transitions(state))
                    if not options:break
                    index=min(range(len(options)),key=lambda i:options[i][2]+step.candidate_map_distances[i])
                    _,state,c=options[index];cost+=c
                    if state in visited:cycle=True;break
                    visited.add(state)
                key=row['cohort']+'|'+row['task']+'|'+row['variant'];g=groups[key];g['count']+=1;g['solved']+=int(state==goal);g['cycles']+=int(cycle)
                if state==goal:
                    g['optimal']+=int(np.isclose(cost,row['reference_cost']));g['excess_cost_on_success']+=cost-row['reference_cost']
            atomic_json(root/'map_reference.json',dict(groups=dict(groups),physical_candidates='all',search=False,action_budget=32))


if __name__=='__main__':main()
