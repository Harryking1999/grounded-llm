"""All-state standard and intermediate-goal relations; separate heldout goal cohorts."""
import json
import numpy as np


def build(env, config, out):
    rng = np.random.default_rng(config["seed"])
    counts = config["data"]["intermediate_goals"]
    sampled = rng.choice(np.arange(1, env.size), sum(counts), replace=False)
    a, b = counts[0], counts[0] + counts[1]
    goals = {"train": np.r_[0, sampled[:a]], "validation": sampled[a:b], "test": sampled[b:]}
    env.prepare_goals(np.r_[0, sampled])
    np.savez_compressed(out / "oracle.npz", goals=np.r_[0, sampled], distances=env.tables)
    data = {}
    for split, targets in goals.items():
        sources = np.arange(env.size)
        # Symmetric pair labels to training goal endpoints must not leak into heldout cohorts.
        if split != "train":
            safe = ~np.isin(sources, goals["train"])
            safe &= ~np.isin(env.next, goals["train"]).any(axis=1)
            sources = sources[safe]
        pairs = np.concatenate([np.column_stack((sources, np.full(len(sources), g),
                                env.distance(sources, g))) for g in targets])
        if split != "train":
            # Frozen distance-stratified monitor cohort, from the fully constructed tables.
            selected = []
            for depth in np.unique(pairs[:, 2]):
                pool = np.flatnonzero(pairs[:, 2] == depth)
                selected.extend(rng.choice(pool, min(len(pool), config["data"]["eval_per_depth"]), replace=False))
            pairs = pairs[np.asarray(selected)]
        tasks = pairs[pairs[:, 2] > 0, :2]
        children = env.successors(tasks[:, 0])
        data[f"pairs_{split}"] = pairs.astype(np.int32)
        data[f"tasks_{split}"] = tasks.astype(np.int32)
        data[f"children_{split}"] = children
        data[f"child_distances_{split}"] = np.where(children >= 0,
            env.distance(np.maximum(children, 0), tasks[:, 1, None]), env.diameter + 1).astype(np.int8)
        data[f"parent_distances_{split}"] = env.distance(tasks[:, 0], tasks[:, 1])
    standard = np.arange(1, env.size)
    rng.shuffle(standard)
    data["tasks_restoration"] = np.column_stack((standard[:config["evaluation"]["rollout_count"]],
                                                np.zeros(min(len(standard), config["evaluation"]["rollout_count"]), dtype=int)))
    np.savez_compressed(out / "data.npz", **data)
    summary = {"states": env.size, "standard_goal": list(env.goal_state),
               "goal_ids": {s: g.tolist() for s, g in goals.items()},
               "standard_goal_distance_coverage": env.size,
               "relation_table_shape": list(env.tables.shape),
               "scope": "all states seen; heldout intermediate goal relations; restoration is in-sample",
               "splits": {s: {"pairs": len(data[f"pairs_{s}"]), "decisions": len(data[f"tasks_{s}"])} for s in goals}}
    (out / "data_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return data, summary
