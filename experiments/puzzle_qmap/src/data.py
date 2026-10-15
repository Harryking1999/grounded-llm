"""Exact labels and disjoint supervision, independent of torch and the Q model."""
import json

import numpy as np


SPLITS = ("train", "validation", "test")


def state_split(ids):
    """Stable split of the entire cube state space; 80/10/10, no sampled pool."""
    bucket = ((np.asarray(ids, dtype=np.int64) * 1103515245 + 12345) % 2147483647) % 10
    return np.where(bucket < 2, bucket + 1, 0)


def pair_keys(rows, size):
    rows = np.asarray(rows)
    return np.minimum(rows[:, 0], rows[:, 1]) * size + np.maximum(rows[:, 0], rows[:, 1])


def cube_pairs(env, rng, count, split, buckets):
    accepted, total = [], 0
    while total < count:
        n = max(1024, (count - total) * (2 if split == 0 else 120))
        n = min(n, 200000)
        source = rng.integers(env.size, size=n)
        distance = rng.integers(1, env.diameter + 1, size=n)
        relative = np.empty(n, dtype=np.int64)
        for d in range(1, env.diameter + 1):
            mask = distance == d
            relative[mask] = rng.choice(buckets[d], mask.sum())
        goal = env.compose(source, relative)
        keep = (state_split(source) == split) & (state_split(goal) == split)
        rows = np.column_stack((source[keep], goal[keep], distance[keep]))[:count - total]
        accepted.append(rows)
        total += len(rows)
    return np.concatenate(accepted)


def build(env, config, out):
    """Persist all data once; training/evaluation consume the same saved arrays."""
    options = config["data"]
    rng = np.random.default_rng(config["seed"])
    data = {}
    membership = None
    if env.name == "hanoi3":
        source, goal = np.triu_indices(env.size, k=1)
        pairs = np.column_stack((source, goal, env.distance(source, goal)))
        rng.shuffle(pairs)
        boundaries = (0, int(len(pairs) * .7), int(len(pairs) * .85), len(pairs))
        membership = np.full(env.size**2, -1, dtype=np.int8)
        for i, name in enumerate(SPLITS):
            data[f"pairs_{name}"] = pairs[boundaries[i]:boundaries[i + 1]]
            membership[pair_keys(data[f"pairs_{name}"], env.size)] = i
        # Identity constraints contain no distance information about distinct states.
        identity = np.column_stack((np.arange(env.size), np.arange(env.size), np.zeros(env.size, dtype=int)))
        membership[pair_keys(identity, env.size)] = 0
        data["pairs_train"] = np.concatenate((data["pairs_train"], identity))
    else:
        buckets = [np.flatnonzero(env.depth == d) for d in range(env.diameter + 1)]
        for i, name in enumerate(SPLITS):
            data[f"pairs_{name}"] = cube_pairs(env, rng, options[f"pairs_{name}"], i, buckets)

    for i, name in enumerate(SPLITS):
        pairs = data[f"pairs_{name}"]
        if env.name == "hanoi3":
            tasks = np.concatenate((pairs[pairs[:, 2] > 0, :2], pairs[pairs[:, 2] > 0, :2][:, ::-1]))
        else:
            tasks = pairs[:options[f"decisions_{name}"], :2]
        children = env.successors(tasks[:, 0])
        valid = children >= 0
        if i == 0:
            if membership is not None:
                keys = np.minimum(np.maximum(children, 0), tasks[:, 1, None]) * env.size + np.maximum(np.maximum(children, 0), tasks[:, 1, None])
                keep = ((membership[keys] == 0) | ~valid).all(axis=1)
            else:
                keep = (state_split(children) == 0).all(axis=1)
            tasks, children, valid = tasks[keep], children[keep], valid[keep]
        if not len(tasks):
            raise ValueError(f"No {name} decisions with admissible supervision")
        data[f"tasks_{name}"] = tasks
        data[f"children_{name}"] = children
        data[f"child_distances_{name}"] = np.where(valid, env.distance(np.maximum(children, 0), tasks[:, 1, None]), env.diameter + 1)
        parents = pairs[:, 0]
        all_children = env.successors(parents)
        src = np.repeat(parents, len(env.actions))
        action = np.tile(np.arange(len(env.actions)), len(parents))
        dst = all_children.ravel()
        keep = dst >= 0
        if membership is not None:
            keys = np.minimum(src, np.maximum(dst, 0)) * env.size + np.maximum(src, np.maximum(dst, 0))
            keep &= membership[keys] == i
        else:
            keep &= state_split(dst) == i
        edges = np.unique(np.column_stack((src[keep], action[keep], dst[keep])), axis=0)
        limit = options.get(f"transitions_{name}", len(edges))
        data[f"transitions_{name}"] = edges[rng.choice(len(edges), min(limit, len(edges)), replace=False)]

    # Prove the actual labels fed to losses respect the split, including ranks.
    train = data["pairs_train"]
    train_keys = np.unique(pair_keys(train, env.size))
    for name in SPLITS[1:]:
        if np.intersect1d(train_keys, pair_keys(data[f"pairs_{name}"], env.size)).size:
            raise ValueError("Unordered pair leakage")
    if membership is None:
        ids = np.concatenate((train[:, :2].ravel(), data["children_train"].ravel(), data["transitions_train"][:, [0, 2]].ravel()))
        if np.any(state_split(ids) != 0):
            raise ValueError("Heldout state entered training supervision")
    np.savez_compressed(out / "data.npz", **data)
    summary = {"split_unit": "unordered_pair_seen_states" if membership is not None else "state_endpoints_unseen_in_all_losses",
               "states_in_environment": env.size, "action_count": len(env.actions), "diameter": env.diameter,
               "splits": {name: {"pairs": len(data[f"pairs_{name}"]),
                                  "unique_unordered_pairs": len(np.unique(pair_keys(data[f"pairs_{name}"], env.size))),
                                  "distance_histogram": np.bincount(data[f"pairs_{name}"][:, 2], minlength=env.diameter + 1).tolist(),
                                  "decisions": len(data[f"tasks_{name}"]), "transitions": len(data[f"transitions_{name}"])} for name in SPLITS}}
    (out / "data_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return data, summary
