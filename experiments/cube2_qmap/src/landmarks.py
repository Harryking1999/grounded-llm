"""Offline landmark stars and explicit two-leg frozen-Q diagnostics."""
import numpy as np
import torch

from .fixed_goal import shortest_paths


def near_route_bank(env, pools, guard, options, rng):
    """Translate exact restoration routes to nearby goals; keep whole stars."""
    depths=sorted(pools)
    quotas={d:options['pool_size']//len(depths)+int(i<options['pool_size']%len(depths))
            for i,d in enumerate(depths)}
    banks={d:[] for d in depths};seen=set();attempts=accepted=0
    for batch in range(options['maximum_candidate_batches']):
        levels=np.asarray(depths)[(np.arange(options['candidate_batch_size'])+attempts)%len(depths)]
        goals=np.array([rng.choice(pools[int(d)]) for d in levels])
        roots=rng.integers(env.size,size=len(goals))
        relative,lengths=shortest_paths(env,roots,rng)
        paths=env.compose(goals[:,None],relative)
        live=np.arange(env.diameter)[None]<lengths[:,None]
        tasks=np.column_stack((paths[:,:-1][live],np.repeat(goals,lengths)))
        blocked=guard.task_blocked(env,tasks)
        owner=np.repeat(np.arange(len(goals)),lengths)
        bad=np.bincount(owner[blocked],minlength=len(goals))>0
        eligible=np.flatnonzero(~bad & (lengths>0))
        attempts+=len(goals);accepted+=len(eligible)
        for index in eligible:
            depth=int(levels[index]);key=(int(paths[index,0]),int(goals[index]))
            if len(banks[depth])<quotas[depth] and key not in seen:
                banks[depth].append((paths[index],int(lengths[index]),int(goals[index])))
                seen.add(key)
        if all(len(banks[d])==quotas[d] for d in depths):
            rows=[row for depth in depths for row in banks[depth]]
            return dict(paths=np.stack([r[0] for r in rows]),lengths=np.array([r[1] for r in rows]),
                        goals=np.array([r[2] for r in rows]),
                        goal_depths=np.array([d for d in depths for _ in banks[d]]),
                        candidates=np.array(attempts),admissible=np.array(accepted))
    raise ValueError('Not enough complete routes outside protected candidate relations')


def near_route_tasks(bank, count, rng):
    depths=np.unique(bank['goal_depths'])
    indices=[]
    for index,depth in enumerate(depths):
        pool=np.flatnonzero(bank['goal_depths']==depth)
        indices.extend(rng.choice(pool,count//len(depths)+int(index<count%len(depths))).tolist())
    lengths=bank['lengths'][indices];paths=bank['paths'][indices]
    live=np.arange(paths.shape[1]-1)[None]<lengths[:,None]
    return np.column_stack((paths[:,:-1][live],np.repeat(bank['goals'][indices],lengths)))


def landmark_batch(env, roots, heldout, rng, goal_depth=7):
    """Link every pre-landmark path parent and all competitors to one target."""
    paths, lengths = shortest_paths(env, roots, rng)
    selected = lengths > goal_depth
    paths, lengths = paths[selected], lengths[selected]
    parents, goals = [], []
    for path, length in zip(paths, lengths):
        offset = int(length) - goal_depth
        parents.extend(path[:offset])
        goals.extend([path[offset]] * offset)
    if not parents:
        return None
    parents, goals = np.asarray(parents), np.asarray(goals)
    stars = np.column_stack((parents, env.successors(parents)))
    keep = ~heldout[stars].any(-1) & ~heldout[goals]
    stars, goals = stars[keep], goals[keep]
    if not len(stars):
        return None
    states = np.unique(np.r_[stars.ravel(), goals])
    indices, target = np.searchsorted(states, stars), np.searchsorted(states, goals)
    # Off-path candidates require actual shortest distances, not path-index gaps.
    truth = env.distance(stars, goals[:, None]).astype(np.float32)
    return {"states": states, "pairs": np.column_stack((indices.ravel(), np.repeat(target, stars.shape[1]))),
            "labels": truth.ravel(), "children": indices[:, 1:], "goals": target,
            "truth": truth[:, 1:]}


@torch.no_grad()
def greedy_leg(model, env, starts, targets, budget, device="cpu", entry_depth=None):
    current, targets = np.asarray(starts).copy(), np.asarray(targets)
    target_q = model.encode(torch.as_tensor(env.features(targets), device=device))
    reached = current == targets
    active = ~reached
    steps = np.zeros(len(current), dtype=int)
    minimum_depth = env.depth[current].copy()
    entered = minimum_depth <= entry_depth if entry_depth is not None else None
    seen = [{int(s)} for s in current]
    for _ in range(budget):
        ids = np.flatnonzero(active)
        if not len(ids):
            break
        children = env.successors(current[ids])
        q = model.encode(torch.as_tensor(env.features(children.ravel()), device=device)).reshape(len(ids), children.shape[1], -1)
        scores = model.distance(q, target_q[ids, None])
        choices = scores.argmin(-1).cpu().numpy()
        current[ids] = children[np.arange(len(ids)), choices]
        steps[ids] += 1
        minimum_depth[ids] = np.minimum(minimum_depth[ids], env.depth[current[ids]])
        if entered is not None:
            entered[ids] |= env.depth[current[ids]] <= entry_depth
        for i in ids:
            if current[i] == targets[i]:
                reached[i], active[i] = True, False
            elif int(current[i]) in seen[i]:
                active[i] = False
            else:
                seen[i].add(int(current[i]))
    return {"current": current, "reached": reached, "steps": steps,
            "entered": entered, "minimum_depth": minimum_depth}


@torch.no_grad()
def diagnostics(model, env, starts, bank, config, device="cpu"):
    opts, depth = config["evaluation"], config["landmarks"]["goal_depth"]
    starts = np.asarray(starts)
    zero = np.zeros(len(starts), dtype=np.int64)
    direct = greedy_leg(model, env, starts, zero, opts["total_budget"], device, depth)
    entered = direct["entered"]
    model.eval()
    source_q = model.encode(torch.as_tensor(env.features(starts), device=device))
    bank_q = model.encode(torch.as_tensor(env.features(bank), device=device))
    selected = bank[model.distance(source_q[:, None], bank_q[None]).argmin(-1).cpu().numpy()]
    # This diagnostic uses a privileged target. It must never be called Q-selected.
    paths, lengths = shortest_paths(env, starts, np.random.default_rng(config["seed"] + 100))
    oracle_landmark = paths[np.arange(len(starts)), lengths.astype(int) - depth]
    result = {"count": len(starts), "direct_solved": int(direct["reached"].sum()),
              "direct_reached_inner_layer": int(entered.sum()),
              "direct_solved_after_entering": int((entered & direct["reached"]).sum()),
              "bank": {"size": len(bank), "selected_true_distance_mean": float(env.distance(starts, selected).mean()),
                       "selected_within_first_leg_budget": int((env.distance(starts, selected) <= opts["first_leg_budget"]).sum()),
                       "nearest_true_distance_mean": float(env.distance(starts[:, None], bank[None]).min(-1).mean())}}
    for name, targets in (("q_selected_waypoint", selected), ("oracle_waypoint_diagnostic", oracle_landmark)):
        first = greedy_leg(model, env, starts, targets, opts["first_leg_budget"], device)
        arrived = np.flatnonzero(first["reached"])
        solved = 0
        if len(arrived):
            # Give each route only its remaining share of the common total budget.
            for used in np.unique(first["steps"][arrived]):
                group = arrived[first["steps"][arrived] == used]
                second = greedy_leg(model, env, targets[group], np.zeros(len(group), dtype=int),
                                    opts["total_budget"] - int(used), device)
                solved += int(second["reached"].sum())
        result[name] = {"reached_landmark": len(arrived), "solved": solved, "count": len(starts)}
    return result
