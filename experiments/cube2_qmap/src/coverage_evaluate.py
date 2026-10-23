"""Select by development cohorts, then open sealed tests once at study end."""
import argparse
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

from grounded_llm.core.files import atomic_json
from grounded_llm.tasks.cube2.environment import Cube2
from grounded_llm.tasks.cube2.evaluate import rollout
from .coverage_data import model_from
from .near_goal_continuation import inspect
from .fixed_goal import exhaustive_policy
from .full_state_joint import fixed_batch,batch_losses
from .coverage_data import predicted_scores
from .coverage_train import edge_loss


def paired_change(before,after):
    a=np.asarray(before,dtype=bool);b=np.asarray(after,dtype=bool)
    gained=int((~a&b).sum());lost=int((a&~b).sum());discordant=gained+lost
    p=min(1.,2*sum(math.comb(discordant,k) for k in range(min(gained,lost)+1))/2**discordant) if discordant else 1.
    differences=b.astype(int)-a.astype(int)
    rng=np.random.default_rng(2026102403)
    samples=np.mean(differences[rng.integers(len(a),size=(10000,len(a)))],axis=1)
    return dict(count=len(a),before=int(a.sum()),after=int(b.sum()),gained=gained,lost=lost,
        both_solved=int((a&b).sum()),both_failed=int((~a&~b).sum()),net=gained-lost,
        paired_bootstrap_95_percent_interval=np.quantile(samples,[.025,.975]).tolist(),
        exact_mcnemar_two_sided_p=p,
        gained_indices=np.flatnonzero(~a&b).tolist(),lost_indices=np.flatnonzero(a&~b).tolist())


@torch.no_grad()
def channel_diagnostic(model,env,parents,device):
    goals=np.flatnonzero(env.depth==1);children=env.successors(parents)
    qg=model.encode(torch.as_tensor(env.features(goals),device=device))
    q=model.encode(torch.as_tensor(env.features(children.ravel()),device=device)).reshape(len(parents),6,-1)
    delta=(q[:,:,None,:]-qg[None,None,:,:]).abs();maximum=delta.amax(-1,keepdim=True);mask=delta==maximum
    invariant=(model.axis_output_action==torch.arange(q.shape[-1],device=device)).all(0)
    actions=maximum.squeeze(-1).argmin(1)
    return dict(parents=len(parents),max_goal1_vector_difference=float((qg-qg[:1]).abs().max()),
        max_goal1_candidate_distance_difference=float((maximum-maximum[:,:,:1]).abs().max()),
        active_max_channels=torch.nonzero(mask.any((0,1,2))).ravel().tolist(),
        fraction_maxima_on_invariant_channels=float(mask[:,:,:,invariant].sum()/mask.sum()),
        parent_fraction_changing_action_across_goal1=float((actions!=actions[:,:1]).any(-1).float().mean()))


def diagnose(args):
    """Training-only fitting, conditioning and gradient-conflict checks."""
    torch.set_num_threads(2);args.out.mkdir(parents=True,exist_ok=False)
    env=Cube2(args.oracle,metric='QTM_URF_6');rng=np.random.default_rng(2026102406)
    pools={}
    for name in ('Original','Cross','OriginalReplay'):
        rows=np.load(args.data/(name+'.npy'))
        if name=='Cross':
            groups=rows.reshape(-1,8,2);pools[name]=groups[rng.choice(len(groups),64,replace=False)].reshape(-1,2)
        else:pools[name]=rows[rng.choice(len(rows),512,replace=False)]
    allowed=np.load(args.data/'edge_allowed.npy',mmap_mode='r');edge_parents=rng.choice(env.size,512,replace=False)
    children=env.successors(edge_parents)
    assert np.array_equal(allowed[edge_parents],allowed[children,np.arange(6)^1]),'Directed edge protection is not symmetric'
    cfg=json.loads((args.run[0]/'config.json').read_text())['study']
    checkpoints=[('initial',args.checkpoint2000)]
    for directory in args.run:
        assert (directory/'complete.json').exists(),str(directory)
        paths=sorted(directory.glob('update_*.pt'))
        checkpoints.append((directory.name,paths[-1]))
    output=dict(source_commit=args.source_commit,training_only=True,models={},
        one_step_edges=dict(directed=int(allowed.sum()),unique_undirected_distance_labels=int(allowed.sum())//2),
        note='A gradient cosine is a local conflict diagnostic, not proof of an unavoidable global tradeoff. Training sample fitting is not independent generalization.')
    for name,path in checkpoints:
        model=model_from(path,env,args.device);gradients={};metrics={}
        for pool,tasks in pools.items():
            batch=fixed_batch(env,tasks);h,r=batch_losses(model,torch.as_tensor(env.features(batch['states']),device=args.device),batch,cfg)
            gradients[pool]={}
            for label,loss in [('huber',h),('rank',r),('total',h+r)]:
                values=torch.autograd.grad(loss,list(model.parameters()),retain_graph=True)
                gradients[pool][label]=torch.cat([v.ravel() for v in values]).detach()
            scores=predicted_scores(model,env,tasks,args.device);truth=env.distance(env.successors(tasks[:,0]),tasks[:,1,None])
            prob=np.exp(-(truth.astype(float)-truth.min(-1,keepdims=True)));prob/=prob.sum(-1,keepdims=True)
            entropy=float(-(prob*np.log(prob)).sum(-1).mean())
            metrics[pool]=dict(huber=float(h.detach()),rank=float(r.detach()),rank_target_entropy=entropy,
                excess_rank_cross_entropy=float(r.detach())-entropy,
                optimal_action_rate=float((truth[np.arange(len(tasks)),scores.argmin(-1)]==truth.min(-1)).mean()),
                training_greedy=inspect(model,env,tasks[:128],args.device,32)['rollout'])
            if pool=='Cross':
                actions=scores.argmin(-1).reshape(-1,8)
                metrics[pool]['parent_fraction_changing_predicted_action_across_goals']=float((actions!=actions[:,:1]).any(-1).mean())
        el,count=edge_loss(model,env,edge_parents,allowed,args.device)
        values=torch.autograd.grad(el,list(model.parameters()))
        gradients['edge']={'total':torch.cat([v.ravel() for v in values]).detach()}
        cosine={}
        names=list(gradients)
        for i,a in enumerate(names):
            for b in names[i+1:]:
                cosine[a+'__'+b]={label:float(torch.nn.functional.cosine_similarity(gradients[a][label],gradients[b][label],dim=0))
                    for label in gradients[a].keys()&gradients[b].keys()}
        output['models'][name]=dict(checkpoint=str(path),pools=metrics,edge_huber=float(el.detach()),
            sampled_effective_edge_labels=count,gradient_cosine=cosine,
            channels=channel_diagnostic(model,env,np.load(args.data/'Cross.npy').reshape(-1,8,2)[:512,0,0],args.device),
            gradient_norms={n:{k:float(v.norm()) for k,v in g.items()} for n,g in gradients.items()})
        atomic_json(args.out/'diagnostics.json',output)
        print(json.dumps(dict(model=name,cross=metrics['Cross']['optimal_action_rate'],edge_huber=float(el.detach()))),flush=True)


def run(args):
    torch.set_num_threads(2);started=time.monotonic()
    args.out.mkdir(parents=True,exist_ok=False)
    candidates=[]
    for directory in args.run:
        if not (directory/'complete.json').exists():raise ValueError(f'Arm unfinished: {directory}')
        summary=json.loads((directory/'summary.json').read_text())
        if summary['best_eligible']:
            candidates.append((tuple(summary['best_validation_and_restoration']),directory,
                summary['best_update']))
    if not candidates:raise ValueError('No restoration-eligible candidate')
    candidates.sort(key=lambda row:(row[0],-row[2]),reverse=True)
    score,directory,update=candidates[0]
    # Save the decision before accessing any sealed test results.
    atomic_json(args.out/'selection.json',dict(selected_run=str(directory),selected_checkpoint=str(directory/'best.pt'),
        validation_and_restoration=score,update=update,
        candidates=[dict(run=str(d),score=s,update=u) for s,d,u in candidates],
        rule='validation first, original restoration >=385; fewer updates breaks exact ties',
        frozen_flan_map_replaced=False,source_commit=args.source_commit))
    env=Cube2(args.oracle,metric='QTM_URF_6');fresh=np.load(args.data/'cohorts.npz')
    old=np.load(args.old_cohorts)
    cohorts={k:fresh[k] for k in ('validation','test','novel_goal_test')}
    cohorts.update(original=fresh['original'],old_small_ood=old['small_ood'],old_validation=old['validation'])
    np.savez_compressed(args.out/'evaluated_tasks.npz',**cohorts)
    results={};oracle={}
    for name,path in [('baseline1000',args.checkpoint1000),('baseline2000',args.checkpoint2000),
                      ('selected',directory/'best.pt')]:
        model=model_from(path,env,args.device)
        results[name]=dict(checkpoint=str(path),scores={k:inspect(model,env,tasks,args.device,32)
            for k,tasks in cohorts.items()})
        if name=='selected':
            for cohort,tasks in cohorts.items():
                oracle[cohort]=rollout(model,env,tasks,args.device,2,horizon=32,return_tasks=True)['oracle']
        if name in ('baseline2000','selected'):
            results[name]['full_restoration']=exhaustive_policy(model,env,fresh['state_mask'],
                {'evaluation':{'horizon':32}},args.device)
        atomic_json(args.out/(name+'.json'),results[name])
        print(json.dumps(dict(stage='tested',model=name,scores={k:v['rollout']['solved']
            for k,v in results[name]['scores'].items()})),flush=True)
    paired={baseline:{name:paired_change(results[baseline]['scores'][name]['rollout']['per_task']['solved'],
        results['selected']['scores'][name]['rollout']['per_task']['solved']) for name in cohorts}
        for baseline in ('baseline1000','baseline2000')}
    atomic_json(args.out/'paired.json',paired);atomic_json(args.out/'oracle_greedy.json',oracle)
    atomic_json(args.out/'complete.json',dict(source_commit=args.source_commit,seconds=time.monotonic()-started,
        horizon=32,selected_run=str(directory),frozen_flan_map_replaced=False,
        note='Development cohorts are selection-biased; heldout depth-2/3 goals exclude continuation labels, not inherited pretraining. Old OOD is regression, never used by this phase for selection or mining.'))


def main():
    p=argparse.ArgumentParser(__doc__)
    for name in ('data','oracle','old-cohorts','checkpoint1000','checkpoint2000','out'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--run',type=Path,action='append',required=True)
    p.add_argument('--device',default='cuda');p.add_argument('--source-commit',required=True)
    p.add_argument('--diagnose',action='store_true')
    args=p.parse_args()
    if args.diagnose:diagnose(args)
    else:run(args)


if __name__=='__main__':main()
