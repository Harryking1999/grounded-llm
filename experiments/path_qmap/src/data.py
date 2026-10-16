"""Freeze linked relation holdouts, protecting the whole initial decision."""
import numpy as np

from experiments.cml_map_scaling.src.core import shortest_distances


def prepare(adjacency, config):
    distances = shortest_distances(adjacency)
    size = len(adjacency)
    options = config["data"]
    rng = np.random.default_rng(options["seed"])
    membership = np.zeros((size, size), dtype=np.int8)
    neighbors = [np.flatnonzero(row) for row in adjacency]
    children = np.full((size, max(map(len, neighbors))), -1, dtype=np.int64)
    for state, row in enumerate(neighbors):
        children[state, :len(row)] = row
    cohorts = {}
    for split, code in (("validation", 1), ("test", 2)):
        tasks = []
        for depth in range(options["first_heldout_distance"], int(distances.max()) + 1):
            candidates = np.argwhere(distances == depth)
            chosen = 0
            for source, goal in candidates[rng.permutation(len(candidates))]:
                endpoints = np.r_[source, neighbors[source]]
                if np.any((membership[endpoints, goal] != 0) &
                          (membership[endpoints, goal] != code)):
                    continue
                membership[endpoints, goal] = code
                membership[goal, endpoints] = code
                tasks.append((int(source), int(goal)))
                chosen += 1
                if chosen == options[f"{split}_tasks_per_distance"]:
                    break
            if not chosen:
                raise ValueError(f"No disjoint {split} decision at distance {depth}")
        cohorts[split] = np.asarray(tasks, dtype=np.int64)
    # Additional relation holdouts are unordered and never remove a physical edge.
    a, b = np.triu_indices(size, k=1)
    available = np.flatnonzero((membership[a, b] == 0) & (distances[a, b] > 1))
    available = rng.permutation(available)
    count = round(len(available) * options["extra_pair_fraction_per_split"])
    for code in (1, 2):
        ids = available[(code - 1) * count:code * count]
        membership[a[ids], b[ids]] = membership[b[ids], a[ids]] = code
    data = {"adjacency": adjacency, "distances": distances, "children": children,
            "membership": membership}
    source, goal = np.where(~np.eye(size, dtype=bool))
    valid = children[source] >= 0
    # A rank label is a distance label: seal candidate relations as well.
    safe = np.maximum(children[source], 0)
    allowed = (membership[source, goal] == 0) & np.all(
        ~valid | (membership[safe, goal[:, None]] == 0), axis=1)
    data["train_tasks"] = np.column_stack((source[allowed], goal[allowed]))
    for name, code in (("train", 0), ("validation", 1), ("test", 2)):
        keep = membership[source, goal] == code
        data[f"{name}_pairs"] = np.column_stack((source[keep], goal[keep],
                                                 distances[source[keep], goal[keep]],
                                                 np.zeros(keep.sum(), dtype=np.int64)))
        if code:
            data[f"{name}_tasks"] = cohorts[name]
    data["train_monitor_tasks"] = data["train_tasks"][rng.choice(
        len(data["train_tasks"]), min(1024, len(data["train_tasks"])), replace=False)]
    if not np.all(membership[adjacency] == 0):
        raise ValueError("Physical edges must remain in training")
    if len(np.unique(data["train_pairs"][:, :2])) != size:
        raise ValueError("Every node must have training relations")
    return data
