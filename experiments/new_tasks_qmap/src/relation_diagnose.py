"""Teacher-only geometry probes; these vectors are never inference features."""
import argparse
import json
from pathlib import Path
import numpy as np
from grounded_llm.core.files import atomic_json
from .relation_data import load


def geometry(corpus,data,config,cohorts):
    """Compare exact shortest-path landmark coordinates to the model's metric.

    With every query goal included, d(s,landmark) coordinates give exact
    directed-max (or symmetric L-infinity) distances to that goal. This is an
    existence check for a scene's target domain, not an oracle-free encoder.
    A 128-landmark subset gives only a lower-bound probe when goals exceed it.
    """
    budget=config['evaluation'].get('landmarks',config['model']['dimension'])
    rng=np.random.default_rng(config['seed']+211)
    landmarks={}
    if hasattr(corpus,'scenes'):
        for sid,scene in enumerate(corpus.scenes):
            goals=scene.goals+corpus.offsets[sid]
            landmarks[sid]=goals if len(goals)<=budget else rng.choice(goals,budget,replace=False)
    else:
        landmarks[0]=rng.choice(corpus.size,min(budget,corpus.size),replace=False)
    result={}
    for name in cohorts:
        rows=data['tasks_'+name]
        truth=data['distances_'+name]
        take=np.arange(len(rows))
        if len(take)>config['evaluation']['decision_count']:
            take=rng.choice(take,config['evaluation']['decision_count'],replace=False)
        rows=rows[take]; truth=truth[take]
        pred=np.empty(len(rows),dtype=np.float32)
        candidate_scores=np.full(data['children_'+name][take].shape,np.inf,dtype=np.float32)
        query_in_bank=np.zeros(len(rows),bool)
        scene_ids=corpus.locate(rows[:,0]) if hasattr(corpus,'scenes') else np.zeros(len(rows),int)
        for sid in np.unique(scene_ids):
            ids=np.flatnonzero(scene_ids==sid); bank=landmarks[int(sid)]
            children=data['children_'+name][take[ids]]; valid=children>=0
            states=np.concatenate([rows[ids].ravel(),children[valid]])
            unique,inverse=np.unique(states,return_inverse=True)
            profiles=np.empty((len(unique),len(bank)),dtype=np.float32)
            for j,g in enumerate(bank):
                d=corpus.distance(unique,np.full(len(unique),g,dtype=np.int64))
                profiles[:,j]=np.where(np.isfinite(d),d,config['training']['unreachable_target'])
            b=len(ids); parent=profiles[inverse[:2*b]].reshape(b,2,-1)
            difference=parent[:,0]-parent[:,1]
            pred[ids]=(np.maximum(difference.max(axis=1),0) if config['model']['metric']=='directed_max'
                       else np.abs(difference).max(axis=1))
            child=profiles[inverse[2*b:]]
            goals=np.broadcast_to(parent[:,1,None],(*children.shape,len(bank)))[valid]
            difference=child-goals
            scores=np.full(children.shape,np.inf,dtype=np.float32)
            scores[valid]=(np.maximum(difference.max(axis=1),0) if config['model']['metric']=='directed_max'
                           else np.abs(difference).max(axis=1))
            candidate_scores[ids]=scores
            query_in_bank[ids]=np.isin(rows[ids,1],bank)
        finite=np.isfinite(truth); decision=finite & (truth>0)
        chosen=candidate_scores.argmin(axis=1)
        optimal=data['optimal_'+name][take]
        result[name]=dict(count=len(rows),finite_count=int(finite.sum()),
            finite_mae=float(np.abs(pred[finite]-truth[finite]).mean()),
            finite_exact_fraction=float((np.abs(pred[finite]-truth[finite])<1e-5).mean()),
            query_goal_in_landmarks=int(query_in_bank.sum()),
            shortest_action_rate=float(optimal[np.flatnonzero(decision),chosen[decision]].mean()) if decision.any() else None,
            dimensions_per_scene={str(sid):len(landmarks[int(sid)]) for sid in np.unique(scene_ids)},
            interpretation='Oracle distance profiles only. Exact fit with all query goals in the bank establishes within-scene representability; it does not establish that the shared MLP learned or can generalize this map. Subset error is not an impossibility proof.')
    return result


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--cohorts',nargs='+',default=['validation'])
    args=parser.parse_args(); config=json.loads(args.config.read_text())
    corpus,data,_=load(config,args.run/'data')
    atomic_json(args.out,dict(teacher_only=True,used_by_encoder_or_planner=False,
        geometry=geometry(corpus,data,config,args.cohorts)))


if __name__=='__main__': main()
