"""Exact labels and split-safe joint symmetry augmentation; no learned model."""
import json

import numpy as np

from grounded_llm.tasks.cube2.augmentation import augment


SPLITS = ("train", "validation", "test")


def state_split(ids):
    """Stable split of the entire cube state space; 80/10/10, no sampled pool."""
    bucket = ((np.asarray(ids, dtype=np.int64) * 1103515245 + 12345) % 2147483647) % 10
    return np.where(bucket < 2, bucket + 1, 0)


def pair_keys(rows, size):
    rows = np.asarray(rows, dtype=np.int64)
    return np.minimum(rows[:, 0], rows[:, 1]) * size + np.maximum(rows[:, 0], rows[:, 1])


def cube_pairs(env, rng, count, split, buckets):
    accepted, total = [], 0
    while total < count:
        n = min(max(1024, (count - total) * (2 if split == 0 else 120)), 200000)
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
    if config["data"].get("builder") == "local_graph":
        from grounded_llm.tasks.cube2.local_graph import build_local_graph
        return build_local_graph(env, config, out)
    if config["data"].get("builder") not in (None, "local_graph"):
        raise ValueError("Archived data builders require their recorded source revision")
    options, augmentation = config["data"], config["augmentation"]
    if augmentation["kind"] != env.symmetry_kind:
        raise ValueError("Augmentation does not match this task's symmetry convention")
    rng = np.random.default_rng(config["seed"])
    data, membership = {}, None
    buckets = [np.flatnonzero(env.depth == d) for d in range(env.diameter + 1)]
    for i, name in enumerate(SPLITS):
        data[f"pairs_{name}"] = cube_pairs(env, rng, options[f"pairs_{name}"], i, buckets)

    def allowed(rows):
        return ((rows < 0) | (state_split(rows) == 0)).all(axis=1)

    base = data["pairs_train"].copy()
    data["pairs_train_base"] = base
    copies = augmentation["copies"] if augmentation["enabled"] else 0
    augmented, indices, pair_stats = augment(env, base[:, :2], allowed, rng, copies, augmentation["max_attempts"])
    if not np.array_equal(env.distance(augmented[:, 0], augmented[:, 1]), base[indices, 2]):
        raise ValueError("Symmetry changed a true pair distance")
    extra = np.column_stack((augmented, base[indices, 2]))
    data["pairs_train"] = np.unique(np.concatenate((base, extra)), axis=0)

    for i, name in enumerate(SPLITS):
        pairs = base if i == 0 else data[f"pairs_{name}"]
        tasks = pairs[:options[f"decisions_{name}"], :2]
        if i == 0:
            packed = np.column_stack((tasks, env.successors(tasks[:, 0])))
            keep = allowed(packed)
            tasks, packed = tasks[keep], packed[keep]
            data["tasks_train_base"] = tasks.copy()
            transformed, _, task_stats = augment(env, packed, allowed, rng, copies, augmentation["max_attempts"])
            tasks = np.unique(np.concatenate((tasks, transformed[:, :2])), axis=0)
            if not allowed(np.column_stack((tasks, env.successors(tasks[:, 0])))).all():
                raise ValueError("Transformed candidate supervision crossed the split")
        if not len(tasks):
            raise ValueError(f"No {name} decisions with admissible supervision")
        children = env.successors(tasks[:, 0])
        data[f"tasks_{name}"] = tasks
        data[f"children_{name}"] = children
        data[f"child_distances_{name}"] = np.where(children >= 0,
            env.distance(np.maximum(children, 0), tasks[:, 1, None]), env.diameter + 1)

    train_keys = np.unique(pair_keys(data["pairs_train"], env.size))
    for name in SPLITS[1:]:
        if np.intersect1d(train_keys, pair_keys(data[f"pairs_{name}"], env.size)).size:
            raise ValueError("Unordered pair leakage")
    if not allowed(data["pairs_train"][:, :2]).all():
        raise ValueError("Heldout labels entered pair supervision")
    np.savez_compressed(out / "data.npz", **data)
    edge_coverage = None
    summary = {"split_unit": "state_endpoints_unseen_in_all_losses",
               "states_in_environment": env.size, "action_count": len(env.actions), "diameter": env.diameter,
               "legal_edge_relation_coverage": edge_coverage,
               "augmentation": {"kind": env.symmetry_kind, "pairs": pair_stats, "decisions": task_stats,
                                "base_pairs": len(base), "base_decisions": len(data["tasks_train_base"])},
               "splits": {name: {"pairs": len(data[f"pairs_{name}"]),
                                  "unique_unordered_pairs": len(np.unique(pair_keys(data[f"pairs_{name}"], env.size))),
                                  "distance_histogram": np.bincount(data[f"pairs_{name}"][:, 2], minlength=env.diameter + 1).tolist(),
                                  "decisions": len(data[f"tasks_{name}"])} for name in SPLITS}}
    (out / "data_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return data, summary
