"""Fixed-pool controls for the observed invariant-coordinate fitting bottleneck."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from grounded_llm.core.files import atomic_json
from grounded_llm.tasks.cube2.environment import Cube2
from grounded_llm.tasks.cube2.model import QMap
from .coverage_data import CoverageGuard, keys_to_pairs, load_coverage_config, predicted_scores, hist
from .coverage_train import Pool
from .full_state_joint import fixed_batch, batch_losses
from .goal_layers import source_distances, switch_metrics
from .near_goal_continuation import inspect


def initialize(model, mode):
    """Change initialization only; channel126 is invariant under all six axes."""
    with torch.no_grad():
        if mode == 'reset_head':
            torch.nn.init.normal_(model.network[-1].weight, std=.01)
            model.network[-1].bias.zero_()
        elif mode == 'release126':
            if not bool((model.axis_output_action[:,126] == 126).all()):
                raise ValueError('Coordinate126 must be an invariant channel')
            model.network[-1].weight[126].zero_()
            model.network[-1].bias[126].zero_()
        elif mode != 'inherited':
            raise ValueError(mode)


def subset_batch(full, ids):
    """Subset precomputed full stars, preserving parent and all six successors."""
    n = len(full['truth'])
    pairs = full['pairs'].reshape(n,7,2)[ids].reshape(-1,2)
    states, inverse = np.unique(pairs, return_inverse=True)
    pairs = inverse.reshape(-1,2)
    grouped = pairs.reshape(-1,7,2)
    return states, dict(pairs=pairs, labels=full['labels'].reshape(n,7)[ids].ravel(),
        children=grouped[:,1:,0], goals=grouped[:,0,1], truth=full['truth'][ids])


@torch.no_grad()
def fit_metrics(model, env, rows, device):
    scores = predicted_scores(model,env,rows,device)
    truth = env.distance(env.successors(rows[:,0]),rows[:,1,None])
    strict = truth[:,:,None] < truth[:,None,:]
    ordered = scores[:,:,None] < scores[:,None,:]
    nontrivial = truth.max(-1) != truth.min(-1)
    right = truth[np.arange(len(rows)),scores.argmin(-1)] == truth.min(-1)
    states = np.unique(np.r_[env.successors(rows[:,0]).ravel(), rows[:,1]])
    q = model.encode(torch.as_tensor(env.features(states),device=device))
    children = q[np.searchsorted(states,env.successors(rows[:,0]))]
    goal = q[np.searchsorted(states,rows[:,1]),None]
    winners = (children-goal).abs().argmax(-1).cpu().numpy()
    # Saturation is measured on the actual sigmoid outputs before axis averaging.
    features = torch.as_tensor(env.features(states),device=device)
    raw = torch.sigmoid(model.network(features[...,model.axis_feature_gather]))
    return dict(count=len(rows), top1=float(right.mean()),
        nontrivial_count=int(nontrivial.sum()), nontrivial_top1=float(right[nontrivial].mean()),
        distance_mae=float(np.abs(source_distances(model,env,rows,device)-env.distance(*rows.T)).mean()),
        strict_pair_accuracy=float((strict&ordered).sum()/strict.sum()),
        all_strict_orderings=float((~strict|ordered).all((1,2)).mean()),
        winning_coordinates=hist(winners),
        sigmoid_saturation_fraction=float(((raw<.001)|(raw>.999)).float().mean()),
        channel126_saturation_fraction=float(((raw[...,126]<.001)|(raw[...,126]>.999)).float().mean()))


def run(args):
    cfg = load_coverage_config(args.config)
    options = cfg['training']; arm = cfg['fit_arms'][args.arm]
    torch.set_num_threads(options['cpu_threads']); torch.manual_seed(cfg['seed'])
    env = Cube2(args.oracle, metric='QTM_URF_6')
    saved = torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    model = QMap(env.feature_dim,**saved['model_config'],cap=saved['cap']).to(args.device)
    model.load_state_dict(saved['model']); initialize(model,arm['initialization'])
    optimizer = torch.optim.Adam(model.parameters(),lr=arm['learning_rate'])
    rows = np.load(args.data/'Micro.npy'); dev = np.load(args.data/'MicroHoldout.npy')
    d = np.load(args.data/'cohorts.npz')
    guard = CoverageGuard(env,keys_to_pairs(d['excluded_relation_keys'],env.size),d['state_mask'],d['heldout_goals'])
    if guard.task_blocked(env,rows).any():
        raise ValueError('Micro fitting would expose protected labels')
    # Training labels are prepared once, using the original batch constructor.
    full = fixed_batch(env,rows)
    features = torch.as_tensor(env.features(full['states']),device=args.device)
    pool = Pool(np.arange(len(rows))[:,None],options['seed'])
    parents,counts = np.unique(rows[:,0],return_counts=True)
    shared = parents[counts == len(np.load(args.data/'goals.npy'))]
    args.out.mkdir(parents=True,exist_ok=False)
    atomic_json(args.out/'config.json',dict(contract=cfg,arm=args.arm,source_commit=args.source_commit,
        checkpoint=str(args.checkpoint),data=str(args.data),optimizer_reset=True))
    started = time.monotonic(); records = []
    for update in range(options['updates']+1):
        if update:
            ids = pool.draw(options['batch_size']).ravel()
            selected,batch = subset_batch(full,ids)
            optimizer.zero_grad(set_to_none=True)
            h,r = batch_losses(model,features[selected],batch,cfg)
            (h+options['rank_weight']*r).backward()
            head_gradient = model.network[-1].weight.grad.norm(dim=1).detach().cpu().numpy()
            torch.nn.utils.clip_grad_norm_(model.parameters(),options['clip_norm']); optimizer.step()
        if update % options['evaluate_every'] and update != options['updates']:
            continue
        model.eval()
        record = dict(update=update, seconds=time.monotonic()-started,
            training=fit_metrics(model,env,rows,args.device),
            development=fit_metrics(model,env,dev,args.device),
            shared_training_switching=switch_metrics(model,env,shared,args.device),
            exposure=int(pool.seen.sum()),minimum_exposure=int(pool.seen.min()),maximum_exposure=int(pool.seen.max()))
        if update:
            record['last_batch_head_gradient_norms'] = head_gradient.tolist()
            record['last_batch_huber'] = float(h.detach()); record['last_batch_rank'] = float(r.detach())
            payload = {key:saved[key] for key in ('model_type','task','actions','model_config','cap')}
            payload.update(model=model.state_dict(),optimizer=optimizer.state_dict(),pool=pool.state_dict(),
                torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all() if str(args.device).startswith('cuda') else [],
                step=update,updates=update,source_commit=args.source_commit,config=cfg,arm=args.arm,
                initial_checkpoint=str(args.checkpoint),data_directory=str(args.data))
            # Atomic publication prevents consumers from opening a partial checkpoint.
            path = args.out/'last.pt'; torch.save(payload,path.with_suffix('.tmp')); path.with_suffix('.tmp').replace(path)
        records.append(record)
        atomic_json(args.out/'summary.json',dict(arm=args.arm,diagnostic_only=True,records=records,
            source_commit=args.source_commit,sealed_tests_not_opened=True))
        print(json.dumps(dict(arm=args.arm,update=update,train=record['training']['top1'],
            dev=record['development']['top1'],seconds=record['seconds'])),flush=True)
        model.train()
    final = dict(validation=inspect(model,env,d['validation'],args.device,32),
        restoration=inspect(model,env,d['original'],args.device,32))
    atomic_json(args.out/'complete.json',dict(updates=update,source_commit=args.source_commit,
        final_monitors=final,sealed_tests_not_opened=True,diagnostic_only=True))


def main():
    p = argparse.ArgumentParser(__doc__)
    for name in ('config','checkpoint','data','oracle','out'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--arm',required=True); p.add_argument('--device',default='cuda:0')
    p.add_argument('--source-commit',required=True)
    run(p.parse_args())


if __name__ == '__main__':
    main()
