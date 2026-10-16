"""Linked cycle/exit/branch packs; all labels come from the exact environment."""
import numpy as np


def cycle_pack(env, cycles, heldout, rng, exit_steps=3, branch_goals=2, max_packs=None):
    """Reject a whole pack if protected states occur; never relabel them.

    Each accepted pack contains both complete cycle stars, a shortest exit
    continuation and competing branches. Dense within-pack distances preserve
    lateral relations; shared cross-pack goals prevent disconnected islands.
    """
    if exit_steps < 1 or branch_goals < 1:
        raise ValueError("Need a nonempty exit continuation and branch targets")
    packs = []
    for first, second in np.asarray(cycles, dtype=np.int64).reshape(-1, 2):
        if first == 0 or second == 0 or second not in env.successors(first):
            raise ValueError("Cycle endpoints must be adjacent nonterminal states")
        lower, upper = sorted((int(first), int(second)), key=lambda s: int(env.depth[s]))
        if int(env.depth[upper]) != int(env.depth[lower]) + 1:
            raise ValueError("Expected quarter-turn cycle endpoints at consecutive depths")
        path = [lower]
        for _ in range(exit_steps):
            if path[-1] == 0:
                break
            children = env.successors(path[-1])
            good = children[env.depth[children].astype(int) == int(env.depth[path[-1]]) - 1]
            path.append(int(rng.choice(good)))
        off_path = np.setdiff1d(env.successors(lower), path)
        rng.shuffle(off_path)
        branches = off_path[:branch_goals]
        parents = np.unique(np.r_[lower, upper, path[:-1], branches])
        parents = parents[parents != 0]
        stars = np.column_stack((parents, env.successors(parents)))
        goals = np.unique(np.r_[0, lower, upper, path[1:], branches])
        states = np.unique(np.r_[stars.ravel(), goals])
        if heldout[states].any():
            continue
        packs.append({"states": states, "stars": stars, "goals": goals,
                      "path": path, "cycle": [lower, upper]})
        if max_packs is not None and len(packs) >= max_packs:
            break
    if not packs:
        return None
    physical_pairs, tasks = [], []
    for index, pack in enumerate(packs):
        # No hand-assigned coordinates. Include pairs across different branches.
        a, b = np.triu_indices(len(pack["states"]), 1)
        physical_pairs.append(pack["states"][np.column_stack((a, b))])
        goals = np.unique(np.r_[pack["goals"], packs[(index + 1) % len(packs)]["goals"]])
        for star in pack["stars"]:
            for goal in goals:
                if star[0] != goal:
                    tasks.append(np.r_[star, goal])
    tasks = np.asarray(tasks, dtype=np.int64)
    # Absolute labels for every parent/candidate of every cross-pack goal.
    physical_pairs.append(np.column_stack((tasks[:, :-1].ravel(),
                                          np.repeat(tasks[:, -1], tasks.shape[1] - 1))))
    pairs = np.unique(np.sort(np.concatenate(physical_pairs), axis=-1), axis=0)
    states = np.unique(np.r_[pairs.ravel(), tasks.ravel()])
    return {"states": states, "pairs": np.searchsorted(states, pairs),
            "labels": env.distance(*pairs.T).astype(np.float32),
            "children": np.searchsorted(states, tasks[:, 1:-1]),
            "goals": np.searchsorted(states, tasks[:, -1]),
            "truth": env.distance(tasks[:, 1:-1], tasks[:, -1, None]).astype(np.float32),
            "packs": [{"cycle": pack["cycle"], "exit_path": pack["path"],
                       "goals": pack["goals"].tolist()} for pack in packs]}
