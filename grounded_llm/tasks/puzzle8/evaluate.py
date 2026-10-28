"""Restoration is primary; heldout intermediate goals measure relation completion."""
from grounded_llm.core.qmap.metrics import pair_metrics, decisions, ranking_loss, rollout
from grounded_llm.core.qmap.sampling import task_indices
import numpy as np
import torch


@torch.no_grad()
def goal_normalized_rollouts(model, env, cohorts, device, horizon=62):
    """Score transformed queries while executing each original physical action."""
    model.eval()
    vectors=torch.cat([model.encode(torch.as_tensor(env.features(np.arange(a,min(a+8192,env.size))),device=device))
                       for a in range(0,env.size,8192)])
    result={}
    for name,tasks in cohorts.items():
        current,goal=tasks[:,0].copy(),tasks[:,1]
        solved=current==goal;active=~solved;steps=np.zeros(len(tasks),dtype=int)
        seen=[{int(s)} for s in current]
        for _ in range(horizon):
            ids=np.flatnonzero(active)
            if not len(ids):break
            children=env.successors(current[ids]);valid=children>=0
            source,target=env.goal_frame(np.maximum(children,0),goal[ids,None])
            scores=model.distance(vectors[source],vectors[target]).cpu().numpy()
            scores[~valid]=np.inf
            current[ids]=children[np.arange(len(ids)),scores.argmin(-1)];steps[ids]+=1
            for i in ids:
                if current[i]==goal[i]:solved[i]=True;active[i]=False
                elif int(current[i]) in seen[i]:active[i]=False
                else:seen[i].add(int(current[i]))
        blank=np.argmin(env.positions[goal],axis=1)
        orbit=np.where(np.isin(blank,[0,2,6,8]),0,np.where(blank==4,2,1))
        result[name]=dict(count=len(tasks),solved=int(solved.sum()),success_rate=float(solved.mean()),
            horizon=horizon,coordinate_frame='goal_normalized',
            by_blank_orbit={kind:dict(count=int((orbit==i).sum()),solved=int(solved[orbit==i].sum()))
                           for i,kind in enumerate(('corner','edge','center'))},
            per_task=dict(solved=solved.tolist(),steps=steps.tolist()))
    return result


def evaluate(model, env, data, config, device):
    result = {}
    for split in ("validation", "test"):
        idx = task_indices(env, data, split, config["evaluation"]["decision_count"], config)
        tasks = data[f"tasks_{split}"][idx]
        ridx = task_indices(env, data, split, config["evaluation"]["rollout_count"], config)
        result[split] = {"pairs": pair_metrics(model, env, data[f"pairs_{split}"], device),
                         "decisions": decisions(model, env, tasks, device),
                         "rollout": rollout(model, env, data[f"tasks_{split}"][ridx], device,
                                            config["evaluation"]["budget_multiplier"])}
    result["restoration_seen_states"] = rollout(model, env, data["tasks_restoration"], device,
                                               config["evaluation"]["budget_multiplier"])
    return result
