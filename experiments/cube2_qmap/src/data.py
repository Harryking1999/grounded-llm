"""Breadth-complete relative tasks, same-goal paths and complete local forks."""
import json
from pathlib import Path

import numpy as np

from experiments.puzzle_qmap.src.data import cube_pairs, SPLITS


def pose_groups(env, groups, forbidden, rng, max_attempts):
    """One common left multiplier per whole group; no protected Q inputs."""
    result = np.empty_like(groups, dtype=np.int32)
    for begin in range(0, len(groups), 20000):
        rows = groups[begin:begin + 20000]
        pending = np.arange(len(rows))
        for _ in range(max_attempts):
            if not len(pending):
                break
            k = rng.integers(env.size, size=len(pending))
            candidate = env.compose(k[:, None], np.maximum(rows[pending], 0))
            keep = ~forbidden[candidate].any(axis=1)
            result[begin + pending[keep]] = candidate[keep]
            pending = pending[~keep]
        if len(pending):
            raise RuntimeError("Could not pose a complete group without protected states")
    return result


def evaluation_pairs(env, rng, count, split, buckets, forbidden):
    chunks, remaining = [], count
    while remaining:
        rows = cube_pairs(env, rng, max(remaining, 128), split, buckets)
        rows = rows[~forbidden[rows[:, :2]].any(axis=1)][:remaining]
        chunks.append(rows)
        remaining -= len(rows)
    return np.concatenate(chunks).astype(np.int32)


def relative_draw(env, rng, count, buckets, low=1, high=11):
    depth = rng.integers(low, high + 1, size=count)
    result = np.empty(count, dtype=np.int32)
    for d in range(low, high + 1):
        mask = depth == d
        result[mask] = rng.choice(buckets[d], mask.sum())
    return result


def decorate(env, data, split, tasks, pairs):
    tasks = np.asarray(tasks, dtype=np.int32)
    relatives = env.relative(tasks[:, 1], tasks[:, 0])
    data[f"pairs_{split}"] = np.asarray(pairs, dtype=np.int32)
    data[f"tasks_{split}"] = tasks
    data[f"children_{split}"] = env.successors(tasks[:, 0]).astype(np.int32)
    data[f"child_distances_{split}"] = env.depth[env.successors(relatives)]
    data[f"parent_distances_{split}"] = env.depth[relatives]


def build_breadth_depth(env, config, out):
    options = config["data"]
    rng = np.random.default_rng(config["seed"])
    reference_path = Path(options["reference_data"])
    metadata = json.loads((reference_path.parent / "summary.json").read_text())
    if metadata["source_commit"] != options["reference_source_commit"]:
        raise ValueError("Reference data is not from the contracted pure-Q run")
    reference = dict(np.load(reference_path, allow_pickle=False))
    protected = np.zeros(env.size, dtype=bool)
    for split in ("validation", "test"):
        protected[reference[f"pairs_{split}"][:, :2]] = True
        protected[reference[f"tasks_{split}"][:, :2]] = True
    reference_protected = int(protected.sum())
    buckets = [np.flatnonzero(env.depth == d) for d in range(12)]
    data = {}
    for i, split in enumerate(("validation", "test"), start=1):
        rows = evaluation_pairs(env, rng, options[f"pairs_{split}"], i, buckets, protected)
        protected[rows[:, :2]] = True
        decorate(env, data, split, rows[:options[f"decisions_{split}"], :2], rows)
    decorate(env, data, "reference_test", reference["tasks_test"], reference["pairs_test"])

    # Breadth: every nonidentity relative state has a complete nine-way decision.
    relatives = np.arange(1, env.size, dtype=np.int32)
    if options["relative_limit"] is not None:
        relatives = rng.choice(relatives, options["relative_limit"], replace=False)
    n = len(relatives)
    packed = np.column_stack((relatives, np.zeros(n, dtype=np.int32), env.successors(relatives)))
    print(json.dumps({"phase": "pose_all_relative_decisions", "count": n}), flush=True)
    posed = pose_groups(env, packed, protected, rng, config["augmentation"]["max_attempts"])
    broad_tasks = posed[:, :2].copy()
    broad_pairs = np.column_stack((broad_tasks, env.depth[relatives])).astype(np.int32)
    del packed, posed

    # Depth: keep a shared goal through every step of a complete shortest path.
    starts = relative_draw(env, rng, options["trajectories"], buckets, 4, 11)
    path = np.full((len(starts), 11), -1, dtype=np.int32)
    current, lengths = starts.copy(), env.depth[starts].astype(int)
    for step in range(11):
        active = lengths > step
        path[active, step] = current[active]
        children = env.successors(current[active])
        good = env.depth[children] == env.depth[current[active], None] - 1
        noise = np.where(good, rng.random(good.shape), np.inf)
        current[active] = children[np.arange(len(children)), noise.argmin(axis=1)]
    if np.any(current != 0):
        raise RuntimeError("Reference paths did not terminate")
    closure = np.column_stack((np.zeros(len(path), dtype=int), np.maximum(path, 0),
                               env.successors(np.maximum(path, 0)).reshape(len(path), -1)))
    posed = pose_groups(env, closure, protected, rng, config["augmentation"]["max_attempts"])
    path_sources = posed[:, 1:12]
    path_goals = np.broadcast_to(posed[:, :1], path_sources.shape)
    path_tasks = np.column_stack((path_sources[path >= 0], path_goals[path >= 0]))
    data["trajectory_offsets"] = np.r_[0, lengths.cumsum()]
    data["trajectory_goals"] = posed[:, 0]
    del closure, posed

    # Local breadth around each fork: parent and all nine children are parents
    # in the same target frame, with all their candidate edges labeled too.
    forks = relative_draw(env, rng, options["forks"], buckets, 4, 10)
    fork_nodes = np.column_stack((forks, env.successors(forks)))
    closure = np.column_stack((np.zeros(len(forks), dtype=int), fork_nodes,
                               env.successors(fork_nodes).reshape(len(forks), -1)))
    posed = pose_groups(env, closure, protected, rng, config["augmentation"]["max_attempts"])
    fork_tasks = np.column_stack((posed[:, 1:11].ravel(), np.repeat(posed[:, 0], 10)))
    tasks = np.concatenate((broad_tasks, path_tasks, fork_tasks)).astype(np.int32)
    data["broad_task_count"] = np.asarray(n)
    data["context_task_start"] = np.asarray(n)
    data["path_task_count"] = np.asarray(len(path_tasks))
    data["protected_states"] = np.flatnonzero(protected).astype(np.int32)

    # Near-pair contexts must not be limited to nine solved-frame one-hop cases.
    near_parts = []
    for depth in range(1, options["near_max_depth"] + 1):
        r = rng.choice(buckets[depth], options["near_pairs_per_depth"])
        posed = pose_groups(env, np.column_stack((r, np.zeros(len(r), dtype=int))),
                            protected, rng, config["augmentation"]["max_attempts"])
        near_parts.append(np.column_stack((posed, np.full(len(r), depth))).astype(np.int32))
    context = np.concatenate((path_tasks, fork_tasks))
    context_pairs = np.column_stack((context, env.distance(context[:, 0], context[:, 1]))).astype(np.int32)
    pairs = np.concatenate((broad_pairs, *near_parts, context_pairs))
    data["broad_pair_count"] = np.asarray(n)
    decorate(env, data, "train", tasks, pairs)
    for key in ("pairs_train", "tasks_train", "children_train"):
        columns = data[key][:, :2] if key.startswith("pairs") else data[key]
        if protected[columns].any():
            raise ValueError("Protected evaluation state entered a training loss")
    # Check the group law on the entire broad set, not just a few samples.
    if not np.array_equal(env.relative(broad_tasks[:, 1], broad_tasks[:, 0]), relatives):
        raise ValueError("Posing changed a relative decision type")
    data["broad_relative_ids"] = relatives
    np.savez_compressed(out / "data.npz", **data)
    summary = {"split_unit": "explicit_protected_reference_and_fresh_evaluation_endpoints",
               "states_in_environment": env.size, "diameter": env.diameter, "action_count": 9,
               "protected_states": int(protected.sum()), "protected_reference_states": reference_protected,
               "relative_decision_types": n, "relative_decision_coverage": n / (env.size - 1),
               "trajectories": len(starts), "path_decisions": len(path_tasks),
               "closed_forks": len(forks), "fork_decisions": len(fork_tasks),
               "near_pairs": sum(len(x) for x in near_parts),
               "reference_source_commit": metadata["source_commit"],
               "splits": {split: {"pairs": len(data[f"pairs_{split}"]), "decisions": len(data[f"tasks_{split}"]),
                                   "distance_histogram": np.bincount(data[f"pairs_{split}"][:, 2], minlength=12).tolist()}
                          for split in (*SPLITS, "reference_test")}}
    (out / "data_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"data_prepared": summary}), flush=True)
    return data, summary
