"""Offline landmark stars and explicit two-leg frozen-Q diagnostics."""
import numpy as np
import torch

from .fixed_goal import shortest_paths


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
