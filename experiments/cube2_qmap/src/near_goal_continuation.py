"""Bounded fine-tuning of the selected Q toward absolute near-solved goals."""
import argparse
import json
from pathlib import Path
import shutil
import time

import numpy as np
import torch

from grounded_llm.core.files import atomic_json
from grounded_llm.tasks.cube2.environment import Cube2
from grounded_llm.tasks.cube2.evaluate import decisions, rollout
from grounded_llm.tasks.cube2.model import MODEL_TYPE, QMap
from .cycle_trial import close_axis_mask
from .cycle_repair import ExitGuard, choose_guard, full_policy, guarded_adam_step
from .fixed_goal import prepare, exhaustive_policy
from .full_state_joint import RelationGuard, task_pairs, fixed_batch, batch_losses


def axis_pairs(env, pairs):
    return np.concatenate([env.axis_symmetry(pairs, transform) for transform in range(6)])


class ProtectedRelations(RelationGuard):
    def __init__(self, env, pairs, state_mask):
        super().__init__(env.size, pairs)
        self.state_mask = state_mask

    def contains(self, pairs):
        return super().contains(pairs) | self.state_mask[np.asarray(pairs)].any(-1)


def sample_tasks(env, pools, count, rng, forbidden):
    tasks = []
    depths = sorted(pools)
    for index in range(count):
        pool = pools[depths[index % len(depths)]]
        for _ in range(10000):
            source = int(rng.integers(env.size))
            goal = int(rng.choice(pool))
            row = np.array([[source, goal]], dtype=np.int64)
            if source != goal and not forbidden.task_blocked(env, row)[0]:
                tasks.append((source, goal))
                break
        else:
            raise ValueError('Cannot sample an unprotected task')
    return np.asarray(tasks, dtype=np.int64)


def training_tasks(env, pools, count, rng, guard, offset, relative_pools=None, local_fraction=0.):
    levels = np.asarray(sorted(pools))[(np.arange(count)+offset) % len(pools)]
    goals = np.array([rng.choice(pools[int(level)]) for level in levels], dtype=np.int64)
    parents = rng.integers(env.size, size=count)
    local = rng.random(count)<local_fraction if relative_pools else np.zeros(count,dtype=bool)
    def local_sources(ids):
        levels=rng.choice(sorted(relative_pools),size=len(ids))
        relative=np.array([rng.choice(relative_pools[int(d)]) for d in levels])
        parents[ids]=env.compose(goals[ids],relative)
    if local.any():
        local_sources(np.flatnonzero(local))
    for _ in range(10000):
        tasks = np.column_stack((parents, goals))
        blocked = (parents == goals) | guard.task_blocked(env, tasks)
        if not blocked.any():
            return tasks
        parents[blocked] = rng.integers(env.size, size=int(blocked.sum()))
        if (blocked & local).any():
            local_sources(np.flatnonzero(blocked & local))
    raise ValueError('Training rejection sampling did not terminate')


@torch.no_grad()
def inspect(model, env, tasks, device, horizon):
    model.eval()
    result = {'rollout': rollout(model, env, tasks, device, 2, horizon=horizon,
                                return_tasks=True)['q'],
              'decisions': decisions(model, env, tasks, device)['q']}
    result['by_goal_depth'] = {}
    for depth in np.unique(env.depth[tasks[:, 1]]):
        keep = env.depth[tasks[:, 1]] == depth
        solved = np.asarray(result['rollout']['per_task']['solved'])[keep]
        result['by_goal_depth'][str(depth)] = {'count': int(keep.sum()), 'solved': int(solved.sum())}
    return result


def run(args):
    config = json.loads(args.contract.read_text())
    if 'parent_contract' in config:
        parent=json.loads(Path(config['parent_contract']).read_text())
        config={**parent,**config,
                'data':{**parent['data'],**config.get('data',{})},
                'training':{**parent['training'],**config.get('training',{})},
                'evaluation':{**parent['evaluation'],**config.get('evaluation',{})}}
    options, opts = config['training'], config['data']
    torch.set_num_threads(options['cpu_threads']);torch.manual_seed(config['seed'])
    saved = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    if saved['task'] != 'cube2_qtm' or 'optimizer' not in saved or saved.get('smoke_steps'):
        raise ValueError('Expected selected formal cube Q with Adam state')
    starting_step=config.get('starting_step',0)
    if starting_step and saved.get('step')!=starting_step:
        raise ValueError('Refinement must start at the configured checkpoint step')
    env = Cube2(args.oracle, metric='QTM_URF_6')
    if env.depth is None:
        raise ValueError('Reuse the existing exact QTM oracle')
    base = json.loads(Path(config['base_contract']).read_text())
    mask, _, _ = prepare(env, base)
    state_mask = close_axis_mask(env, mask > 0)
    records = json.loads((args.evaluation_data/'evaluation.json').read_text())
    cohorts = {name:np.array([(int(r['start']), int(r['goal'])) for r in records
                             if r['cohort']==name], dtype=np.int64)
               for name in ('original', 'small_ood')}
    if any(len(rows) != 500 for rows in cohorts.values()):
        raise ValueError('Expected the paired 500+500 cube regression tasks')
    pools = {d:np.flatnonzero(env.depth==d) for d in opts['goal_depths']}
    relation_pairs = axis_pairs(env, task_pairs(env, np.concatenate(list(cohorts.values()))))
    rng = np.random.default_rng(config['seed'])
    # Fresh monitor tasks have disjoint relation stars, including hard-coded
    # symmetry equivalents. Test and regression never guide checkpoint choice.
    for name in ('validation', 'test'):
        guard = ProtectedRelations(env, relation_pairs, state_mask)
        tasks = sample_tasks(env, pools, opts[name+'_tasks'], rng, guard)
        cohorts[name] = tasks
        relation_pairs = np.concatenate((relation_pairs, axis_pairs(env, task_pairs(env, tasks))))
    if opts.get('final_test_tasks'):
        guard=ProtectedRelations(env,relation_pairs,state_mask)
        cohorts['final_test']=sample_tasks(env,pools,opts['final_test_tasks'],rng,guard)
        relation_pairs=np.concatenate((relation_pairs,axis_pairs(env,task_pairs(env,cohorts['final_test']))))
    guard = ProtectedRelations(env, relation_pairs, state_mask)
    args.out.mkdir(parents=True, exist_ok=False)
    atomic_json(args.out/'config.json', config)
    np.savez_compressed(args.out/'cohorts.npz', state_mask=state_mask,
                        excluded_relation_keys=guard.keys, **cohorts)
    model = QMap(env.feature_dim, **saved['model_config'], cap=saved['cap']).to(args.device)
    model.load_state_dict(saved['model'])
    optimizer = torch.optim.Adam(model.parameters(), lr=options['learning_rate'])
    optimizer.load_state_dict(saved['optimizer'])
    for group in optimizer.param_groups:
        group['lr'] = options['learning_rate']
    horizon = config['evaluation']['horizon']
    evaluation_base = {**base, 'evaluation':{**base['evaluation'], 'horizon':horizon}}
    result = dict(source_commit=args.source_commit, initial_checkpoint=str(args.checkpoint),
        initial_source_commit=saved['source_commit'],
        initial_optimizer_loaded=True, historical_sampling_state_resumed=False,
        protected_states=int(state_mask.sum()), protected_relation_count=len(guard.keys),
        goal_pool_sizes={d:len(pool) for d,pool in pools.items()}, evaluations=[])
    result['initial'] = {name:inspect(model, env, tasks, args.device, horizon)
                         for name,tasks in cohorts.items() if name!='final_test'}
    if config['evaluation']['final_exhaustive_restoration']:
        result['initial_exhaustive_restoration'] = exhaustive_policy(
            model, env, mask, evaluation_base, args.device)
    atomic_json(args.out/'summary.json', result)
    print(json.dumps({'stage':'initial_evaluation', 'cohorts':{
        name:row['rollout']['solved'] for name,row in result['initial'].items()}}), flush=True)
    training_rng = np.random.default_rng(options.get('seed',config['seed']+1))
    seen_goals, exposed = set(), {'near_stars':0, 'restoration_stars':0}
    relative_pools={d:np.flatnonzero(env.depth==d) for d in opts.get('local_distance_levels',[])}
    fixed_guard=None;accepted_scales=[]
    if 'guard' in config:
        allowed=np.zeros(env.size,dtype=bool)
        for begin in range(0,env.size,32768):
            ids=np.arange(begin,min(begin+32768,env.size))
            allowed[ids]=~guard.task_blocked(env,np.column_stack((ids,np.zeros(len(ids),dtype=int))))
        allowed[0]=False
        guard_rng=np.random.default_rng(config['seed']+20)
        _,_,margins=full_policy(model,env,args.device)
        parents=choose_guard(env,margins,allowed,config['guard']['count'],
                             config['guard']['fragile_fraction'],guard_rng)
        floors=np.minimum(margins[parents]*config['guard']['retain_fraction'],config['guard']['floor_cap'])
        fixed_guard=ExitGuard(model,env,parents,floors,args.device)
        np.savez_compressed(args.out/'restoration_guard.npz',parents=parents,floors=floors)
    select=config.get('selection',{})
    selecting=isinstance(select,dict)
    best_step=starting_step
    initial_original=result['initial']['original']['rollout']['solved']
    best_score=(result['initial']['validation']['rollout']['solved'],initial_original)
    if selecting:
        shutil.copy2(args.checkpoint,args.out/'best.pt')
        shutil.copy2(args.checkpoint,args.out/f'step_{starting_step:06d}.pt')
    result['best_step']=best_step
    started = time.monotonic()
    for phase_step in range(1, options['steps']+1):
        step=starting_step+phase_step
        model.train()
        near = training_tasks(env, pools, options['near_goal_batch_size'], training_rng,
                              guard, (phase_step-1)*options['near_goal_batch_size'],
                              relative_pools,opts.get('local_fraction',0.))
        restoration = training_tasks(env, {0:np.array([0])}, options['restoration_batch_size'],
                                     training_rng, guard, 0)
        terms = []
        optimizer.zero_grad(set_to_none=True)
        for tasks,weight in ((near,options['near_goal_weight']),
                             (restoration,1-options['near_goal_weight'])):
            batch = fixed_batch(env, tasks)
            if guard.contains(batch['states'][batch['pairs']]).any():
                raise ValueError('Protected relation entered supervision')
            huber,rank = batch_losses(model, torch.as_tensor(env.features(batch['states']),
                                                          device=args.device), batch, config)
            (weight*(huber+options['rank_weight']*rank)).backward()
            terms.append({'huber':float(huber.detach()), 'rank':float(rank.detach())})
        torch.nn.utils.clip_grad_norm_(model.parameters(), options['clip_norm'])
        if fixed_guard is None:
            optimizer.step();scale=1.
        else:
            scale=guarded_adam_step(model,optimizer,fixed_guard.accepts,config['guard']['scales'])
        accepted_scales.append(scale)
        seen_goals.update(map(int,near[:, 1]))
        exposed['near_stars'] += len(near);exposed['restoration_stars'] += len(restoration)
        if phase_step==1 or phase_step % options['log_every']==0:
            progress=dict(step=step, terms=terms, seconds=time.monotonic()-started,
                          goal_count=len(seen_goals), accepted_scale=scale, **exposed)
            atomic_json(args.out/'progress.json', progress)
            print(json.dumps(progress), flush=True)
        if phase_step % options['evaluate_every']==0 or phase_step==options['steps']:
            metrics = {name:inspect(model, env, cohorts[name], args.device, horizon)
                       for name in ('validation', 'original')}
            result['evaluations'].append(dict(step=step, metrics=metrics, **exposed))
            payload = dict(model_type=MODEL_TYPE, task=env.name, actions=list(env.actions),
                model_config=saved['model_config'], cap=saved['cap'], model=model.state_dict(),
                optimizer=optimizer.state_dict(), rng=training_rng.bit_generator.state,
                torch_rng=torch.get_rng_state(), config=config, step=step,
                source_commit=args.source_commit, initial_checkpoint=str(args.checkpoint),
                seen_goals=sorted(seen_goals), exposure=exposed.copy(),
                cohorts_file=str(args.out/'cohorts.npz'))
            temporary = args.out/'last.pt.tmp';torch.save(payload,temporary)
            temporary.replace(args.out/'last.pt')
            if selecting:
                torch.save(payload,args.out/f'step_{step:06d}.pt')
                score=(metrics['validation']['rollout']['solved'],metrics['original']['rollout']['solved'])
                if score[1]>=initial_original and score>best_score:
                    shutil.copy2(args.out/'last.pt',args.out/'best.pt')
                    best_score,best_step=score,step
                result['best_step']=best_step
                result['best_validation_and_original']=best_score
            result['accepted_scales']={str(s):accepted_scales.count(s) for s in sorted(set(accepted_scales))}
            if fixed_guard is not None:
                result['fixed_guard']=fixed_guard.report()
            atomic_json(args.out/'summary.json', result)
            print(json.dumps({'stage':'validation','step':step, 'solved':{
                name:row['rollout']['solved'] for name,row in metrics.items()}}), flush=True)
    if selecting:
        result['last']={name:inspect(model,env,cohorts[name],args.device,horizon)
                        for name in ('original','small_ood','test')}
        best=torch.load(args.out/'best.pt',map_location=args.device,weights_only=False)
        model.load_state_dict(best['model'])
    result['final']={name:inspect(model,env,tasks,args.device,horizon) for name,tasks in cohorts.items()}
    if config['evaluation']['final_exhaustive_restoration']:
        result['final_exhaustive_restoration']=exhaustive_policy(model,env,mask,evaluation_base,args.device)
    atomic_json(args.out/'summary.json',result)
    atomic_json(args.out/'complete.json', dict(step=step, source_commit=args.source_commit,
        selected_checkpoint='validation_with_original_retention_not_adopted' if selecting else 'final_only_not_adopted'))


def main():
    parser=argparse.ArgumentParser(__doc__)
    for name in ('contract','checkpoint','oracle','evaluation-data','out'):
        parser.add_argument('--'+name, required=True, type=Path)
    parser.add_argument('--source-commit', required=True)
    parser.add_argument('--device', default='cuda')
    run(parser.parse_args())


if __name__=='__main__':main()
