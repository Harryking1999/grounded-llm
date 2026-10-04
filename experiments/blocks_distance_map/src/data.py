"""Certified pair supervision, grouped splits, and held-out goal decisions."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import time

import numpy as np

from experiments.gcml_counterexamples.src import blocks
from .oracle import DistanceOracle, successors


def collect(case, episodes, seed):
    rng = np.random.default_rng(seed)
    initial = blocks.from_grid(case["grid"])
    paths, edges = [], set()
    path = [initial]
    for action in case["reference_actions"]:
        tile = blocks.PLACEMENTS[action][0]
        if tile & path[-1] != tile:
            raise ValueError("Invalid official reference")
        edges.add((path[-1], action, path[-1] ^ tile))
        path.append(path[-1] ^ tile)
    if path[-1] != 0:
        raise ValueError("Reference did not reach empty board")
    paths.append(path)
    for _ in range(episodes):
        path = [initial]
        while path[-1]:
            legal = successors(path[-1])
            if not legal:
                break
            action, following = legal[int(rng.integers(len(legal)))]
            edges.add((path[-1], action, following))
            path.append(following)
        paths.append(path)
    return initial, paths, edges


def pair_splits(pairs, forced_test, fractions, seed, forced_train=None):
    """Both directions of an unordered pair belong to the same split."""
    rng = np.random.default_rng(seed)
    forced_train = forced_train or set()
    if forced_train & forced_test:
        raise ValueError("Reserved training and test relations overlap")
    assignment = {}
    result = []
    for s, t, *_ in pairs:
        key = tuple(sorted((int(s), int(t))))
        if key not in assignment:
            assignment[key] = (2 if key in forced_test else 0 if key in forced_train
                               else int(rng.choice(3, p=fractions)))
        result.append(assignment[key])
    return np.asarray(result, dtype=np.int8)


def comparisons(pairs, ids, per_anchor, rng):
    """Store near/far pair-row IDs; outgoing and incoming comparisons stay distinct."""
    result = []
    for axis in (0, 1):
        groups = defaultdict(lambda: defaultdict(list))
        for idx in ids:
            s, t, distance, _ = pairs[idx]
            key = int(distance) if distance >= 0 else 1000000
            groups[int((s, t)[axis])][key].append(int(idx))
        rows = []
        for buckets in groups.values():
            levels = sorted(buckets)
            if len(levels) < 2:
                continue
            for _ in range(per_anchor):
                near, far = sorted(rng.choice(levels, 2, replace=False))
                rows.append((rng.choice(buckets[near]), rng.choice(buckets[far])))
        result.append(np.asarray(rows, dtype=np.int64).reshape(-1, 2))
    return result


def prepare(config, out):
    started = time.monotonic()
    options = config["data"]
    rng = np.random.default_rng(options["seed"])
    initial, paths, edges = collect(config["case"], options["episodes"], options["seed"])
    states = {x for path in paths for x in path}
    oracle = DistanceOracle(cache_limit=options["oracle_cache_limit"],
                            seconds=options["oracle_seconds"])
    initial_distance = oracle.cover(initial)
    eligible = [s for s in sorted(states) if s and oracle.cover(s) >= 0]
    rng.shuffle(eligible)
    parents = eligible[:options["decision_parents"]]
    decisions = []
    forced_goal_states = set()
    for s in parents:
        for action, t in successors(s):
            states.add(t)
            decisions.append((s, action, t, oracle.cover(t)))
            if t:
                forced_goal_states.add(t)
    # Single-tile boards are solvable nonempty landmarks. For a dead successor,
    # reaching any contained landmark is also impossible, while a solvable
    # successor may reach some. This gives indirect goal structure without
    # training on the reserved successor-to-empty relation itself.
    landmarks = sorted({tile for tile, _ in blocks.PLACEMENTS if tile & initial == tile})
    if options.get("landmark_pairs_per_state", 0):
        states.update(landmarks)
    states = sorted(states)
    lookup = {s: i for i, s in enumerate(states)}
    # Every state has goal supervision somewhere; evaluation successors' goal
    # labels are reserved for test in BOTH directions, including rank losses.
    labels = {}

    def add(s, t):
        if s == t or (s, t) in labels:
            return
        d = oracle.distance(s, t)
        kind = 0 if d >= 0 else (1 if t & s != t else 2)
        labels[s, t] = (d, kind)

    for s in states:
        add(s, 0)
        add(0, s)
        # Give each table row a training relation independent of the held-out
        # goal labels. All these states descend from the initial board.
        add(initial, s)
        add(s, initial)
    for s, _, t in sorted(edges):
        add(s, t)
        add(t, s)
    for s, _, t, _ in decisions:
        add(s, t)
        add(t, s)
    landmark_keys = set()
    per_state = options.get("landmark_pairs_per_state", 0)
    if per_state:
        for s in states:
            choices = [t for t in landmarks if t != s and t & s == t]
            if choices:
                chosen = rng.choice(choices, size=min(per_state, len(choices)), replace=False)
                for t in chosen:
                    add(s, int(t))
                    landmark_keys.add(tuple(sorted((lookup[s], lookup[int(t)]))))
    mandatory_count = len(labels)
    # Observed paths certify reachability; the oracle, not time difference,
    # supplies the shortest-distance label.
    path_pairs = sorted({(path[i], path[j]) for path in paths
                         for i in range(len(path)) for j in range(i + 1, len(path))})
    rng.shuffle(path_pairs)
    finite_budget = max(options["target_pairs"] // 2, sum(d >= 0 for d, _ in labels.values()))
    finite_count = sum(d >= 0 for d, _ in labels.values())
    for s, t in path_pairs:
        if finite_count >= finite_budget:
            break
        if (s, t) not in labels:
            add(s, t)
            finite_count += 1
    # Hard negatives must have target contained in source. A missing observed
    # path is never treated as a negative. Bound proposal effort explicitly.
    hard_budget = options["target_pairs"] // 4
    hard_count = sum(k == 2 for _, k in labels.values())
    for _ in range(options["subset_pair_attempts"]):
        if hard_count >= hard_budget or len(labels) >= options["target_pairs"]:
            break
        s, t = (states[i] for i in rng.integers(len(states), size=2))
        if s == t or t & s != t or (s, t) in labels:
            continue
        d = oracle.distance(s, t)
        if d < 0:
            add(s, t)
            hard_count += 1
    for _ in range(options["random_pair_attempts"]):
        if len(labels) >= options["target_pairs"]:
            break
        s, t = (states[i] for i in rng.integers(len(states), size=2))
        add(s, t)
    pairs = np.asarray([(lookup[s], lookup[t], d, kind)
                        for (s, t), (d, kind) in sorted(labels.items())], dtype=np.int32)
    forced_test = {tuple(sorted((lookup[s], lookup[0]))) for s in forced_goal_states}
    initial_keys = {tuple(sorted((lookup[initial], lookup[s]))) for s in states if s != initial}
    forced_train = set(initial_keys)
    forced_train.update(landmark_keys)
    split = pair_splits(pairs, forced_test, options["split_fractions"], options["seed"], forced_train)
    # Ensure every table row is trained via some non-held-out relation. Do not
    # repair a missing row by moving reserved evaluation labels into training.
    covered = set(pairs[split == 0, :2].ravel().tolist())
    if len(covered) != len(states):
        raise ValueError(f"{len(states) - len(covered)} states lack training constraints")
    goal_labels = np.asarray([oracle.cover(s) for s in states], dtype=np.int16)
    immediate = np.asarray([bool(s) and not successors(s) for s in states])
    cap = initial.bit_count() // min(t.bit_count() for t, _ in blocks.PLACEMENTS) + 1
    payload = {"states": np.asarray([str(s) for s in states]), "pairs": pairs,
               "split": split, "goal_labels": goal_labels, "immediate_dead": immediate,
               "decisions": np.asarray([(lookup[s], a, lookup[t], d)
                    for s, a, t, d in decisions], dtype=np.int32).reshape(-1, 4),
               "transitions": np.asarray([(lookup[s], a, lookup[t]) for s, a, t in sorted(edges)],
                                         dtype=np.int32),
               "cap": np.asarray(cap), "goal_id": np.asarray(lookup[0]),
               "initial_id": np.asarray(lookup[initial])}
    summary = {"case_id": config["case"]["id"], "initial_cells": initial.bit_count(),
               "initial_exact_distance": initial_distance, "unreachable_target": cap,
               "states": len(states), "transitions": len(edges), "pairs": len(pairs),
               "mandatory_pairs": mandatory_count, "goal_test_states": len(forced_goal_states),
               "initial_anchor_training_groups": len(initial_keys),
               "nonempty_landmarks": len(landmarks), "landmark_training_groups": len(landmark_keys),
               "decision_parents": len(parents), "decision_candidates": len(decisions),
               "action_count": len(blocks.PLACEMENTS), "shape_rows": list(blocks.CONFIG["blocks"]["shape_rows"]),
               "oracle_cache_entries": len(oracle.cache), "unknown_labels": 0, "splits": {}}
    for sid, name in enumerate(("train", "validation", "test")):
        ids = np.flatnonzero(split == sid)
        outgoing, incoming = comparisons(pairs, ids, options["comparisons_per_anchor"], rng)
        payload[f"{name}_outgoing"] = outgoing
        payload[f"{name}_incoming"] = incoming
        summary["splits"][name] = {"pairs": len(ids),
            "distance_counts": dict(Counter(map(str, pairs[ids, 2].tolist()))),
            "negative_kind_counts": dict(Counter(map(str, pairs[ids, 3].tolist()))),
            "outgoing_comparisons": len(outgoing), "incoming_comparisons": len(incoming)}
    summary["seconds"] = time.monotonic() - started
    out.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(out / "data.npz", **payload)
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    print(json.dumps(summary), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    prepare(json.loads(args.config.read_text()), args.out)


if __name__ == "__main__":
    main()
