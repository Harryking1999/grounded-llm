"""Frozen bridge candidates on new time scenarios, using existing exact evaluators."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from grounded_llm.core.files import atomic_json
from grounded_llm.core.qmap.model import StateEncoder
from grounded_llm.tasks.bridge.relation_data import build,load
from .relation_data import materialize
from .relation_evaluate import evaluate


def run(config,runs,people,device,source_commit):
    name=config['candidates'][str(people)];reference=runs/name
    out=reference/'frozen_acceptance'
    if out.exists():raise FileExistsError('Independent suite already exists; do not overwrite it')
    out.mkdir()
    original=json.loads((reference/'config.json').read_text())
    saved=torch.load(reference/'best.pt',map_location=device,weights_only=False)
    if saved['scope']['people']!=people:raise ValueError('Mismatched personnel count')
    manifest=json.loads((reference/'data/graphs/corpus.json').read_text())
    excluded=[scene['times'] for scene in manifest['scenes']]
    preparation=dict(seed=config['seed']+people,data=dict(people=people,maximum_time=config['maximum_time'],
        scenes=dict(test=config['scenarios_per_map']),goals_per_scene=config['goals_per_scene'],
        heldout_goals=0,excluded_times=excluded))
    # Freeze the data inputs and exact graphs before loading model predictions.
    atomic_json(out/'config.json',dict(acceptance=config,preparation=preparation,checkpoint=str(reference/'best.pt')))
    corpus=build(preparation,out/'graphs');groups={key:[] for key in config['required_cohorts']}
    dead=[];rng=np.random.default_rng(config['seed']+people+100)
    for sid,scene in enumerate(corpus.scenes):
        offset=corpus.offsets[sid];n=len(scene.states)
        groups['all_right_test'].append([offset,offset+scene.env.goal])
        candidates=[int(g) for j,g in enumerate(scene.goals) if g not in (0,scene.env.goal) and np.isfinite(scene.distances[j,0])]
        if len(candidates)<2:raise ValueError('Insufficient complete intermediate targets')
        goal=candidates[0];groups['intermediate_goal_test'].append([offset,offset+goal])
        goal=candidates[1];column=scene.goal_columns[goal]
        pool=np.flatnonzero(np.isfinite(scene.distances[column]) & (scene.distances[column]>0))
        pool=pool[pool!=0]
        groups['intermediate_start_test'].append([offset+int(rng.choice(pool)),offset+goal])
        dead.extend([[offset+1,offset+scene.env.goal],[offset+n-2,offset+scene.env.goal]])
    groups={k:np.asarray(v,dtype=np.int64) for k,v in groups.items()}
    groups['reachability_test']=np.vstack([groups['all_right_test'],np.asarray(dead,dtype=np.int64)])
    atomic_json(out/'suite.json',dict(times=[s.env.times for s in corpus.scenes],groups={k:v.tolist() for k,v in groups.items()}))
    data={}
    for name,rows in groups.items():data.update({k+'_'+name:v for k,v in materialize(corpus,rows).items()})
    evaluation=dict(original)
    evaluation['evaluation']=dict(original['evaluation'],decision_count=4096,rollout_count=config['scenarios_per_map'],horizon=config['horizon'])
    torch.set_num_threads(original['training']['cpu_threads'])
    model=StateEncoder(input_dim=saved['input_dim'],**original['model']).to(device)
    model.load_state_dict(saved['model']);model.requires_grad_(False)
    threshold=json.loads((reference/'summary.json').read_text())['metrics']['reachability_threshold']
    metrics=evaluate(model,corpus,data,evaluation,threshold=threshold)
    required={k:metrics['cohorts'][k]['planning']['q'] for k in config['required_cohorts']}
    oracle_ok=all(metrics['cohorts'][k]['planning']['oracle']['optimal_rate']==1 for k in config['required_cohorts'])
    passed=oracle_ok and all(q['count']==config['scenarios_per_map'] and q['success_rate']>=config['minimum_completion_rate']
        and q['optimal_rate']>=config['minimum_optimal_cost_rate'] for q in required.values())
    result=dict(people=people,source_commit=source_commit,checkpoint_source_commit=saved['source_commit'],step=saved['step'],
        excluded_time_scenarios=len(excluded),novel_scenarios=len(corpus.scenes),oracle_ok=oracle_ok,passed=passed,
        target_reached=passed and all(q['success_rate']>=config['target_completion_rate'] for q in required.values()),metrics=metrics)
    atomic_json(out/'result.json',result)
    atomic_json(reference/'frozen_acceptance.json',result)
    print(json.dumps(dict(people=people,passed=passed,cohorts={k:dict(solved=q['solved'],optimal=q['optimal'],count=q['count']) for k,q in required.items()})),flush=True)


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--config',type=Path,required=True);parser.add_argument('--runs',type=Path,required=True)
    parser.add_argument('--people',type=int,required=True);parser.add_argument('--device',default='cpu')
    parser.add_argument('--source-commit',required=True);args=parser.parse_args()
    run(json.loads(args.config.read_text()),args.runs,args.people,args.device,args.source_commit)


if __name__=='__main__':main()
