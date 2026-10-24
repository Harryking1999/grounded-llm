"""Frozen Sokoban training-gradient diagnosis; no optimizer or sampling changes."""
import numpy as np
import torch
from torch.nn import functional as F
from grounded_llm.core.qmap.model import soft_rank_loss


def component_gradients(model,batch,features,options):
    """Decompose the existing objective, retaining its actual batch weights.

    Origin masks refer to prepared pool row IDs, including duplicated physical
    relations. Autograd returns gradients without changing parameters or .grad.
    """
    device=next(model.parameters()).device
    pair=batch['pairs'];tasks=batch['tasks'];children=batch['children'];valid=children>=0
    ids=np.r_[pair[:,0],pair[:,1],children[valid],tasks[:,1]]
    unique,inverse=np.unique(ids,return_inverse=True)
    vectors=model.encode(torch.as_tensor(features(unique),dtype=torch.float32,device=device))
    inverse=torch.as_tensor(inverse,device=device);b,n=len(pair),len(tasks);nc=int(valid.sum())
    truth=np.asarray(batch['point_truth']);target=np.where(np.isfinite(truth),truth,options['unreachable_target'])
    point=F.smooth_l1_loss(model.distance(vectors[inverse[:b]],vectors[inverse[b:2*b]]),
        torch.as_tensor(target,device=device),reduction='none')
    mask=torch.as_tensor(valid,device=device);scores=torch.zeros(children.shape,device=device)
    goals=vectors[inverse[2*b+nc:]][:,None].expand(-1,children.shape[1],-1)[mask]
    scores[mask]=1+model.distance(vectors[inverse[2*b:2*b+nc]],goals)
    target=1+np.where(np.isfinite(batch['child_truth']),batch['child_truth'],options['unreachable_target'])
    target=torch.as_tensor(target,device=device)
    rank=soft_rank_loss(scores,target,mask,options['rank_temperature'])
    exact=point.mean()+options['rank_weight']*rank
    losses={};finite=np.isfinite(truth)
    for name,origin in [('broad',~batch['point_is_graph']),('graph',batch['point_is_graph'])]:
        for kind,take in [('finite',finite),('unreachable',~finite)]:
            selected=torch.as_tensor(origin&take,device=device)
            losses[name+'_'+kind+'_distance']=point[selected].sum()/b
    for name,origin in [('broad',~batch['decision_is_graph']),('graph',batch['decision_is_graph'])]:
        selected=torch.as_tensor(origin,device=device);count=int(selected.sum())
        losses[name+'_ranking']=(soft_rank_loss(scores[selected],target[selected],mask[selected],options['rank_temperature'])
            *options['rank_weight']*count/n if count else exact*0)
    parameters=list(model.parameters())
    flatten=lambda values:torch.cat([v.reshape(-1) for v in values])
    gradients={name:flatten(torch.autograd.grad(loss,parameters,retain_graph=True)).detach() for name,loss in losses.items()}
    total=flatten(torch.autograd.grad(exact,parameters)).detach()
    residual=(sum(gradients.values())-total).norm()/total.norm().clamp_min(1e-12)
    if not torch.allclose(sum(losses.values()),exact,rtol=1e-5,atol=1e-6) or float(residual)>1e-5:
        raise ValueError('Original objective did not decompose into the unchanged weighted groups')
    return dict(values={k:float(v.detach()) for k,v in losses.items()},total=float(exact.detach()),
                gradients=gradients,total_gradient=total,decomposition_relative_error=float(residual))


def frozen_adam_direction(parameters,gradient,state):
    """Read the saved original Adam moments to predict one step, without taking it.

    This supports only the existing single-group, ordinary Adam configuration.
    The result is a first-order diagnostic direction, never applied to a model.
    """
    group,=state['param_groups'];parameters=list(parameters)
    if group['weight_decay'] or group['amsgrad'] or group.get('maximize') or group.get('decoupled_weight_decay'):
        raise ValueError('Diagnosis only supports the unchanged original Adam')
    if len(parameters)!=len(group['params']):raise ValueError('Adam parameter identity mismatch')
    beta1,beta2=group['betas'];directions=[];offset=0
    for parameter,key in zip(parameters,group['params']):
        size=parameter.numel();g=gradient[offset:offset+size].reshape(parameter.shape);offset+=size
        history=state['state'][key];step=float(history['step'])+1
        m=history['exp_avg'].to(device=g.device,dtype=g.dtype)
        v=history['exp_avg_sq'].to(device=g.device,dtype=g.dtype)
        next_m=(beta1*m+(1-beta1)*g)/(1-beta1**step)
        next_v=(beta2*v+(1-beta2)*g.square())/(1-beta2**step)
        directions.append((-group['lr']*next_m/(next_v.sqrt()+group['eps'])).reshape(-1))
    if offset!=len(gradient):raise ValueError('Adam gradient shape mismatch')
    return torch.cat(directions)
