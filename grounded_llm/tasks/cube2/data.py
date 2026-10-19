"""Split-safe joint poses and exact labels for cube local graphs."""

import numpy as np

from grounded_llm.tasks.cube2.relation_data import cube_pairs


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


def decorate(env, data, split, tasks, pairs):
    tasks = np.asarray(tasks, dtype=np.int32)
    relatives = env.relative(tasks[:, 1], tasks[:, 0])
    data[f"pairs_{split}"] = np.asarray(pairs, dtype=np.int32)
    data[f"tasks_{split}"] = tasks
    data[f"children_{split}"] = env.successors(tasks[:, 0]).astype(np.int32)
    data[f"child_distances_{split}"] = env.depth[env.successors(relatives)]
    data[f"parent_distances_{split}"] = env.depth[relatives]
