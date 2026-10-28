"""Learn a fixed-goal distance through the existing Q encoder and norm."""
import argparse
import json
from pathlib import Path
import time
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

from grounded_llm.tasks.cube2.evaluate import decisions, pair_metrics, rollout
from grounded_llm.tasks.cube2.model import MODEL_TYPE, QMap
from grounded_llm.tasks.cube2.environment import Cube2
from .full_state_joint import batch_losses
from grounded_llm.core.qmap.parallel import replica_step


def tensor_batch(parents, successors, depth, heldout):
    """The existing complete fixed-goal stars, prepared on the tensor device."""
    stars = torch.cat((parents[:, None], successors[parents]), dim=1)
    states, inverse = torch.unique(torch.cat((parents.new_zeros(1), stars.flatten())),
                                   sorted=True, return_inverse=True)
    indices = inverse[1:].reshape(stars.shape)
    allowed = ~heldout[stars]
    source = indices[allowed]
    live = allowed.all(-1) & (parents != 0)
    children = indices[live, 1:]
    if not len(children):raise ValueError('No complete nonterminal ranking task in replica')
    return dict(states=states, pairs=torch.stack((source, torch.zeros_like(source)), -1),
                labels=depth[stars[allowed]].float(), children=children,
                goals=torch.zeros_like(indices[live, 0]), truth=depth[stars[live, 1:]])


class FixedGoalReplicas:
    """Split one original objective over replicas with device-resident rule data."""
    def __init__(self, model, env, heldout, devices):
        if str(next(model.parameters()).device) != str(torch.device(devices[0])):
            raise ValueError('The primary replica must own the existing optimizer')
        self.models = [model]+[deepcopy(model).to(d) for d in devices[1:]]
        features = torch.as_tensor(env.features(np.arange(env.size)),device=devices[0],
                                   dtype=next(model.parameters()).dtype)
        self.features = [features]+[features.to(d) for d in devices[1:]]
        children = env.successors(np.arange(env.size))
        self.children = [torch.as_tensor(children,device=d) for d in devices]
        self.depth = [torch.as_tensor(env.depth,device=d) for d in devices]
        self.heldout = [torch.as_tensor(heldout,device=d) for d in devices]
        self.pool = ThreadPoolExecutor(max_workers=len(devices))

    def update(self, parents, optimizer, config):
        partitions = np.array_split(parents,len(self.models))
        def prepare_replica(i):
            ids = torch.as_tensor(partitions[i],device=self.features[i].device)
            return tensor_batch(ids,self.children[i],self.depth[i],self.heldout[i])
        batches = list(self.pool.map(prepare_replica,range(len(self.models))))
        pairs = sum(len(b['pairs']) for b in batches)
        tasks = sum(len(b['goals']) for b in batches)
        def worker(i):
            batch = batches[i]
            h,r = batch_losses(self.models[i],self.features[i][batch['states']],batch,config)
            h,r = h*(len(batch['pairs'])/pairs),r*(len(batch['goals'])/tasks)
            (h+config['training']['rank_weight']*r).backward()
            return float(h.detach()),float(r.detach())
        values = replica_step(self.models,optimizer,worker,self.pool,config['training']['clip_norm'])
        device = self.features[0].device
        return (torch.tensor(sum(v[0] for v in values),device=device),
                torch.tensor(sum(v[1] for v in values),device=device),
                dict(goal_pair_count=pairs,path_pair_count=0))

    def close(self):
        self.pool.shutdown()


def prepare(env, config):
    rng = np.random.default_rng(config["data"]["seed"])
    mask = np.zeros(env.size, dtype=np.uint8)
    options = config["data"]
    for depth in range(options["first_heldout_depth"], env.diameter + 1):
        layer = np.flatnonzero(env.depth == depth)
        count = min(len(layer) // 4, max(options["minimum_heldout_per_layer"],
                                        round(len(layer) * options["heldout_fraction_per_split"])))
        selected = rng.choice(layer, count * 2, replace=False)
        mask[selected[:count]], mask[selected[count:]] = 1, 2
    train_layers = [np.flatnonzero((env.depth == depth) & (mask == 0)) for depth in range(env.diameter + 1)]
    cohorts = {}
    for name, split in (("train", 0), ("validation", 1), ("test", 2)):
        rows = []
        for depth in range(1, env.diameter + 1):
            layer = np.flatnonzero((env.depth == depth) & (mask == split))
            rows.extend(rng.choice(layer, min(len(layer), config["evaluation"]["states_per_layer"]), replace=False))
        cohorts[name] = np.asarray(rows, dtype=np.int64)
    if mask[0] or not all(len(layer) for layer in train_layers):
        raise ValueError("Solved state and every distance layer need training labels")
    return mask, train_layers, cohorts


def shortest_paths(env, roots, rng):
    """Sample paths for offline labels only; no path is exposed to inference."""
    roots = np.asarray(roots, dtype=np.int64)
    lengths = env.depth[roots].astype(int)
    paths = np.zeros((len(roots), env.diameter + 1), dtype=np.int64)
    paths[:, 0] = roots
    for step in range(env.diameter):
        active = np.flatnonzero(lengths > step)
        if not len(active):
            break
        current = paths[active, step]
        children = env.successors(current)
        optimal = env.depth[children].astype(int) == env.depth[current, None].astype(int) - 1
        if not optimal.any(-1).all():
            raise ValueError("Shortest restoration path has no advancing successor")
        choice = np.where(optimal, rng.random(optimal.shape), -1).argmax(-1)
        paths[active, step + 1] = children[np.arange(len(active)), choice]
    return paths, lengths


def path_relations(env, roots, heldout, rng):
    """All intermediate pairs on sampled shortest restoration paths."""
    paths, lengths = shortest_paths(env, roots, rng)
    a, b = np.triu_indices(env.diameter + 1, 1)
    valid = b[None] <= lengths[:, None]
    pairs = np.stack((paths[:, a], paths[:, b]), axis=-1)[valid]
    # A subpath of a shortest path is shortest too; the index gap is exact.
    labels = np.broadcast_to(b - a, valid.shape)[valid].astype(np.float32)
    allowed = ~heldout[pairs].any(-1)
    return pairs[allowed], labels[allowed]


def make_batch(env, parents, heldout, path_roots=None, rng=None):
    stars = np.column_stack((parents, env.successors(parents)))
    states, inverse = np.unique(np.r_[0, stars.ravel()], return_inverse=True)
    goal = int(inverse[0])
    indices = inverse[1:].reshape(stars.shape)
    allowed = ~heldout[stars]
    pairs = np.column_stack((indices[allowed], np.full(allowed.sum(), goal)))
    # One unavailable candidate label invalidates the entire soft target.
    live = allowed.all(-1) & (parents != 0)
    if not live.any():
        raise ValueError("No complete nonterminal ranking task in training batch")
    result = {"states": states, "pairs": pairs, "labels": env.depth[stars[allowed]].astype(np.float32),
            "children": indices[live, 1:], "goals": np.full(live.sum(), goal),
            "truth": env.depth[stars[live, 1:]]}
    result["goal_pair_count"] = len(pairs)
    result["path_pair_count"] = 0
    if path_roots is not None:
        physical, labels = path_relations(env, path_roots, heldout, rng)
        expanded = np.unique(np.r_[states, physical.ravel()])
        remap = np.searchsorted(expanded, states)
        for key in ("pairs", "children", "goals"):
            result[key] = remap[result[key]]
        result["states"] = expanded
        result["pairs"] = np.concatenate((result["pairs"], np.searchsorted(expanded, physical)))
        result["labels"] = np.r_[result["labels"], labels]
        result["path_pair_count"] = len(labels)
    return result


def balanced_parents(layers, count, offset, rng):
    levels = (np.arange(count) + offset) % (len(layers) - 1) + 1
    states = np.empty(count, dtype=np.int64)
    for depth in np.unique(levels):
        chosen = levels == depth
        states[chosen] = rng.choice(layers[depth], chosen.sum())
    return states


def reach_within_budget(next_state, goal, horizon):
    """Evaluate a fixed learned policy; no path or action is selected here."""
    steps = np.full(len(next_state), -1, dtype=np.int32)
    steps[goal] = 0
    for _ in range(horizon):
        previous = steps[next_state]
        added = (steps < 0) & (previous >= 0)
        if not added.any():
            break
        steps[added] = previous[added] + 1
    return steps


@torch.no_grad()
def exhaustive_policy(model, env, mask, config, device, feature_table=None):
    """Cache network distances, choose each next state, then score all starts."""
    goal = model.encode(feature_table[:1] if feature_table is not None else
                        torch.as_tensor(env.features([0]), device=device))
    predicted = np.empty(env.size, dtype=np.float32)
    next_state = np.empty(env.size, dtype=np.int32)
    width = config['evaluation'].get('encode_batch_size',8192)
    for start in range(0, env.size, width):
        ids = np.arange(start, min(env.size, start + width))
        q = model.encode(feature_table[start:start+width] if feature_table is not None else
                         torch.as_tensor(env.features(ids), device=device))
        predicted[ids] = model.distance(q, goal).cpu().numpy()
    # Action selection reads learned distances only, never env.depth.
    for start in range(0, env.size, 32768):
        ids = np.arange(start, min(env.size, start + 32768))
        children = env.successors(ids)
        choice = predicted[children].argmin(-1)
        next_state[ids] = children[np.arange(len(ids)), choice]
    next_state[0] = 0
    steps = reach_within_budget(next_state, 0, config["evaluation"]["horizon"])
    error = np.abs(predicted - env.depth)
    good = env.depth[next_state].astype(int) == env.depth.astype(int) - 1
    result = {"horizon": config["evaluation"]["horizon"], "policy": "frozen Q distance greedy; no search", "cohorts": {}}
    for name, keep in (("all_states", np.ones(env.size, dtype=bool)), ("supervised", mask == 0),
                       ("heldout_validation", mask == 1), ("heldout_test", mask == 2)):
        nonterminal = keep & (env.depth > 0)
        result["cohorts"][name] = {"count": int(keep.sum()), "mae": float(error[keep].mean()) if keep.any() else None,
            "shortest_action_rate": float(good[nonterminal].mean()) if nonterminal.any() else None,
            "solved": int((steps[keep] >= 0).sum()), "success_rate": float((steps[keep] >= 0).mean()) if keep.any() else None,
            "by_depth": {str(depth): {"count": int((keep & (env.depth == depth)).sum()),
                "solved": int(((steps >= 0) & keep & (env.depth == depth)).sum())}
                for depth in np.unique(env.depth[keep])}}
    return result


@torch.no_grad()
def evaluate(model, env, cohorts, config, device):
    result = {}
    for name, states in cohorts.items():
        tasks = np.column_stack((states, np.zeros(len(states), dtype=int)))
        rows = np.column_stack((tasks, env.depth[states]))
        result[name] = {"pairs": pair_metrics(model, env, rows, device),
                        "decisions": decisions(model, env, tasks, device)["q"],
                        "rollout": rollout(model, env, tasks, device, 2,
                                           horizon=config["evaluation"]["horizon"])["q"]}
    return result


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--oracle", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--resume-run", type=Path)
    parser.add_argument('--checkpoint',type=Path,help='Warm-start a new fixed-goal condition with its Adam state')
    parser.add_argument('--devices',nargs='+')
    parser.add_argument("--smoke-steps", type=int, default=0)
    args = parser.parse_args()
    config = json.loads(args.contract.read_text())
    if config["rules"]["fixed_goal"] != 0:
        raise ValueError("This task is restoration to canonical solved state zero")
    torch.set_num_threads(2)
    env = Cube2(args.oracle, metric=config["rules"]["metric"])
    mask, layers, cohorts = prepare(env, config)
    heldout = mask > 0
    rng = np.random.default_rng(config["data"]["seed"] + 1)
    torch.manual_seed(config["seed"])
    model = QMap(env.feature_dim, **config["model"], cap=env.diameter).to(args.device)
    options = config["training"]
    optimizer = torch.optim.Adam(model.parameters(), lr=options["learning_rate"])
    starting_epoch, starting_step = 0, 0
    parent = None
    if args.checkpoint is not None:
        if args.resume_run is not None:raise ValueError('Use either warm start or interrupted-run resume')
        saved = torch.load(args.checkpoint,map_location='cpu',weights_only=False)
        if saved['task'] != env.name or saved['model_config'] != config['model'] or 'optimizer' not in saved:
            raise ValueError('Warm start requires the matching complete selected Q and Adam')
        model.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer'])
        for group in optimizer.param_groups:group['lr'] = options['learning_rate']
        parent = dict(checkpoint=str(args.checkpoint),source_commit=saved['source_commit'],
                      optimizer_step=max(int(v['step']) for v in saved['optimizer']['state'].values()))
    if args.resume_run:
        if json.loads((args.resume_run / "contract.json").read_text()) != config:
            raise ValueError("Resume changed fixed-goal contract")
        saved = torch.load(args.resume_run / "last.pt", map_location="cpu", weights_only=False)
        starting_epoch, starting_step = saved["epoch"], saved.get('continuation_step',saved['step'])
        parent = saved.get('parent')
        expected = starting_epoch * ((env.size + options["parent_batch_size"] - 1) // options["parent_batch_size"])
        if saved["fixed_goal"] != 0 or starting_step != expected or saved["smoke_steps"]:
            raise ValueError("Resume requires a complete formal fixed-goal epoch")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        rng.bit_generator.state = saved["rng"]
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "contract.json").write_text(json.dumps(config, indent=2) + "\n")
    np.savez_compressed(args.out / "split.npz", mask=mask, **cohorts)
    started, step = time.monotonic(), starting_step
    result = {"source_commit": args.source_commit, "fixed_goal": 0, "seed": config["seed"],
              "starting_epoch": starting_epoch, "supervised_state_count": int((mask == 0).sum()),
              "heldout_state_count": int(heldout.sum()), "evaluations": []}
    result['parent'] = parent
    relative_cohorts = {}
    if config['evaluation'].get('relative_tasks'):
        with np.load(config['evaluation']['relative_tasks']) as packed:
            for name in ('original','small_ood','validation','test'):
                tasks = packed[name]
                relative_cohorts[name] = np.column_stack((env.relative(tasks[:,1],tasks[:,0]),
                                                         np.zeros(len(tasks),dtype=np.int64)))
    parallel = None
    if args.devices:
        if options['paths_per_batch']:raise ValueError('Device-resident restoration uses solved-goal labels only')
        parallel = FixedGoalReplicas(model,env,heldout,args.devices)
    best_validation = (-1,float('-inf'))

    def inspect(epoch, log):
        nonlocal best_validation
        row = {"epoch": epoch, "step": step, "seconds": time.monotonic() - started,
               "metrics": evaluate(model, env, cohorts, config, args.device)}
        validation = row['metrics']['validation']
        score = (validation['rollout']['solved'],-validation['pairs']['mae'])
        if config.get('selection') == 'validation' and score > best_validation:
            best_validation = score
            save(epoch,'best.pt')
            result['best_epoch'] = epoch
        interval = config['evaluation'].get('exhaustive_every_epochs')
        if interval and epoch % interval == 0:
            row['exhaustive_policy'] = exhaustive_policy(model,env,mask,config,args.device,
                                                        parallel.features[0] if parallel else None)
        if relative_cohorts:
            row['relative_goals'] = {name:rollout(model,env,tasks,args.device,2,
                                                  horizon=config['evaluation']['horizon'])['q']
                                     for name,tasks in relative_cohorts.items()}
        if 'exhaustive_policy' in row and relative_cohorts:
            target = config['evaluation'].get('minimum_success_rate',1.)
            row['target_met'] = (row['exhaustive_policy']['cohorts']['all_states']['success_rate'] >= target
                                 and all(v['success_rate'] >= target for v in row['relative_goals'].values()))
        result["evaluations"].append(row)
        result["last"] = row
        (args.out / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
        log.write(json.dumps(row) + "\n")
        log.flush()
        print(json.dumps({"epoch": epoch, "step": step, "seconds": row["seconds"],
            "test_mae": row["metrics"]["test"]["pairs"]["mae"],
            "test_solved": row["metrics"]["test"]["rollout"]["solved"],
            "test_count": row["metrics"]["test"]["rollout"]["count"]}), flush=True)

    def save(epoch,filename='last.pt'):
        temporary = args.out / (filename+'.tmp')
        torch.save({"model_type": MODEL_TYPE, "task": env.name, "actions": list(env.actions), "cap": env.diameter,
            "model_config": config["model"], "source_commit": args.source_commit, "seed": config["seed"],
            "fixed_goal": 0, "epoch": epoch, "step": step+(parent['optimizer_step'] if parent else 0),
            'continuation_step':step,'parent':parent,"smoke_steps": args.smoke_steps,
            "model": model.state_dict(), "optimizer": optimizer.state_dict(), "rng": rng.bit_generator.state}, temporary)
        temporary.replace(args.out / filename)

    with (args.out / "progress.jsonl").open("w") as log:
        inspect(starting_epoch, log)
        completed = starting_epoch
        for epoch in range(starting_epoch + 1, options["epochs"] + 1):
            order = rng.permutation(env.size)
            for start in range(0, env.size, options["parent_batch_size"]):
                parents = order[start:start + options["parent_batch_size"]]
                replay = balanced_parents(layers, options["balanced_replay_size"], step * options["balanced_replay_size"], rng)
                all_parents = np.r_[parents, replay]
                begin = time.monotonic()
                if parallel:
                    huber,rank,batch = parallel.update(all_parents,optimizer,config)
                else:
                    path_roots = rng.choice(all_parents,min(len(all_parents),options['paths_per_batch']),replace=False)
                    batch = make_batch(env, all_parents, heldout, path_roots, rng)
                    features = torch.as_tensor(env.features(batch["states"]), device=args.device)
                    huber, rank = batch_losses(model, features, batch, config)
                    loss = huber + options["rank_weight"] * rank
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), options["clip_norm"])
                    optimizer.step()
                step += 1
                if step % options["log_every"] == 0:
                    row = {"epoch": epoch, "step": step, "seconds": time.monotonic() - started,
                           "huber": float(huber.detach()), "rank_ce": float(rank.detach()),
                           "goal_pairs": batch["goal_pair_count"], "path_pairs": batch["path_pair_count"],
                           'update_seconds':time.monotonic()-begin,'replicas':len(parallel.models) if parallel else 1}
                    log.write(json.dumps(row) + "\n")
                    log.flush()
                    print(json.dumps(row), flush=True)
                if args.smoke_steps and step - starting_step >= args.smoke_steps:
                    break
            if not args.smoke_steps:
                completed = epoch
            inspect(epoch if not args.smoke_steps else completed, log)
            save(completed)
            if args.smoke_steps:
                break
            if (options.get('stop_on_target') and completed >= options.get('minimum_epochs',0)
                    and result['last'].get('target_met')):
                result['stopped_on_target'] = True
                break
        if config["evaluation"]["final_exhaustive_policy"] and not args.smoke_steps:
            result["exhaustive_policy"] = exhaustive_policy(model, env, mask, config, args.device)
    if parallel:parallel.close()
    result["completed_epochs"] = completed
    result["seconds"] = time.monotonic() - started
    (args.out / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    (args.out / "complete.json").write_text(json.dumps({"completed_epochs": completed, "formal": not bool(args.smoke_steps)}) + "\n")


if __name__ == "__main__":
    main()
