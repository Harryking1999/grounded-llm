"""Separate map ranking, decoded action, and true reachability on matched states."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
import torch
from .analyze_blocks_ranking import ndcg
from experiments.flamingo_map_reader.src.blocks import FrozenBoardMap, blocks_step
from experiments.flamingo_map_reader.src.trajectory_dataset import load_record
from experiments.flamingo_map_reader.src.relations import parse_ranking, first_ranked_candidate
from experiments.blocks_distance_map.src.oracle import DistanceOracle


def summary(rows):
    if not rows:
        return dict(n=0)
    booleans=['model_safe','map_best_safe','map_best_all_safe','map_best_any_safe',
              'model_is_map_best','model_follows_first_ranked','model_follows_predicted_top_tie',
              'first_ranked_safe','map_best_safe_model_unsafe','map_best_unsafe_model_safe',
              'both_unsafe','model_action_valid']
    return dict(n=len(rows),**{k:sum(r[k] for r in rows) for k in booleans},
        mean_ndcg=sum(r['ndcg'] for r in rows)/len(rows),
        random_ndcg=sum(r['random_ndcg'] for r in rows)/len(rows),
        mean_map_best_safe_fraction=sum(r['map_best_safe_fraction'] for r in rows)/len(rows))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--run-root',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    torch.set_num_threads(1)
    data=args.run_root/'blocks/data'
    manifest=json.loads((data/'manifest.json').read_text())
    records={r['trajectory_id']:r for r in manifest['records'] if r['split']=='validation'}
    qmap=FrozenBoardMap.load(Path(manifest['q_checkpoint']))
    oracle=DistanceOracle(cache_limit=2000000,seconds=1000000)
    args.out.mkdir(parents=True,exist_ok=True)
    result=dict(run_root=str(args.run_root),split='validation',checkpoint='final',
        definition='All solvable nonterminal saved states. Map-best tie breaker: smallest physical action ID among map-minimal candidates; also report all/any and uniform tie average.',modes={})
    with (args.out/'per_turn.jsonl').open('w') as handle:
        for mode in ['rollout','reference']:
            files=sorted((args.run_root/'evaluation/blocks/final'/f'validation_{mode}_map').glob('*/summary.json'))
            cases=[json.loads(l) for f in files for l in (f.parent/'cases.jsonl').read_text().splitlines() if l.strip()]
            rows=[]
            for ci,case in enumerate(cases):
                rec=records[case['trajectory_id']]
                if mode=='reference':
                    demo=load_record(data/'trajectories',rec,manifest['config'],case['variant'])
                else:
                    rng=np.random.default_rng(rec['sample_seed']+104729*(case['variant']+1))
                for i,row in enumerate(case['turns']):
                    # Advance the saved RNG even on excluded rollout rows.
                    step=demo.turns[i].step if mode=='reference' else blocks_step(qmap,int(row['current']),int(row['goal']),rng=rng)
                    if row['done'] or row['remaining_shortest']<1:
                        continue
                    assert step.current==int(row['current']) and len(step.candidate_actions)==row['candidates']
                    distances=step.candidate_map_distances
                    minimal=step.map_minimal_candidates
                    best=min(minimal,key=lambda j:step.candidate_actions[j-1])
                    def safe(j):
                        return j is not None and 1<=j<=len(distances) and oracle.distance(step.candidate_destinations[j-1],step.goal)>=0
                    map_safe=[safe(j) for j in minimal]
                    chosen=row.get('chosen_id') if row['legal_action'] else None
                    action_safe=safe(chosen)
                    if 'action_keeps_goal_reachable' in row:
                        assert action_safe==row['action_keeps_goal_reachable']
                    ranks=parse_ranking(row['answer'],len(distances),10)
                    first=first_ranked_candidate(row['answer'],len(distances))
                    predicted={} if ranks is None else {j:r for j,r in ranks.items() if j}
                    top=set() if not predicted else {j for j,r in predicted.items() if r==min(predicted.values())}
                    score,chance=ndcg(distances,ranks)
                    r=dict(trajectory_id=case['trajectory_id'],mode=mode,step=i+1,
                        candidates=len(distances),ndcg=score,random_ndcg=chance,
                        model_safe=action_safe,map_best_safe=safe(best),map_best_all_safe=all(map_safe),map_best_any_safe=any(map_safe),
                        map_best_safe_fraction=sum(map_safe)/len(map_safe),model_is_map_best=chosen in minimal,
                        model_follows_first_ranked=chosen is not None and chosen==first,
                        model_follows_predicted_top_tie=chosen is not None and chosen in top,
                        first_ranked_safe=safe(first),model_action_valid=chosen is not None,
                        map_best_safe_model_unsafe=safe(best) and not action_safe,
                        map_best_unsafe_model_safe=not safe(best) and action_safe,both_unsafe=not safe(best) and not action_safe,
                        chosen_id=chosen,map_best_id=best,
                        chosen_distance=None if chosen is None else distances[chosen-1],map_best_distance=distances[best-1])
                    if i==0:
                        r['chosen_true_distance']=None if chosen is None else oracle.distance(step.candidate_destinations[chosen-1],step.goal)
                        r['map_best_true_distance']=oracle.distance(step.candidate_destinations[best-1],step.goal)
                        r['ranking_text']=[s for s in row['answer'].splitlines() if s.startswith('Map-distance ranking')]
                    rows.append(r)
                    handle.write(json.dumps(r)+'\n')
                if (ci+1)%200==0:
                    print(json.dumps(dict(mode=mode,cases=ci+1,scored=len(rows))),flush=True)
            first=[r for r in rows if r['step']==1]
            bins={label:summary([r for r in first if a<=r['ndcg']<b]) for label,a,b in [('below_90',0,.9),('90_to_95',.9,.95),('95_to_99',.95,.99),('99_to_100',.99,1.0001)]}
            examples=sorted([r for r in first if not r['model_safe'] and r['map_best_safe']],key=lambda r:r['ndcg'],reverse=True)[:3]
            result['modes'][mode]=dict(overall=summary(rows),first_step=summary(first),
                 first_step_ndcg_bins=bins,first_step_examples=examples,
                 ndcg_on_unsafe_first=summary([r for r in first if not r['model_safe']]),
                 ndcg_on_safe_first=summary([r for r in first if r['model_safe']]))
            print(json.dumps(dict(mode=mode,first_step=summary(first),overall=summary(rows))),flush=True)
    (args.out/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print('DONE',flush=True)


if __name__=='__main__':
    main()
