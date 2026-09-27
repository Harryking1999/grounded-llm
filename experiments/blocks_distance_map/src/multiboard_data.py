"""Official-board sampling and certified multi-goal supervision for a shared Q map."""
from collections import Counter, defaultdict
import json
from pathlib import Path

import h5py
import numpy as np

from experiments.gcml_counterexamples.src import blocks
from .oracle import DistanceOracle, successors


def mask_from_observation(row):
    occupied = np.flatnonzero(row)
    return sum(1 << int(bit) for bit in occupied)


def official_boards(path, seed, train_count, test_count):
    with h5py.File(path) as archive:
        group = archive["tiling"]
        observations = group["pre_obs"][:, 0, :]
        actions = group["action"][:].astype(np.int32)
    rng = np.random.default_rng(seed)
    selected, seen = [], set()
    for row in rng.permutation(len(observations)):
        mask = mask_from_observation(observations[row])
        if mask in seen:
            continue
        seen.add(mask)
        selected.append((int(row), mask, actions[row].tolist()))
        if len(selected) == train_count + test_count:
            break
    if len(selected) != train_count + test_count:
        raise ValueError("Official dataset has too few distinct initial boards")
    return selected[:train_count], selected[train_count:]


def sample_paths(board, episodes, rng):
    _, initial, reference = board
    path = [initial]
    for action in reference:
        tile = blocks.PLACEMENTS[action][0]
        if tile & path[-1] != tile:
            raise ValueError("Official reference action is illegal")
        path.append(path[-1] ^ tile)
    if path[-1] != 0:
        raise ValueError("Official reference does not clear the board")
    paths = [path]
    for _ in range(episodes):
        path = [initial]
        while path[-1]:
            choices = successors(path[-1])
            if not choices:
                break
            path.append(choices[int(rng.integers(len(choices)))][1])
        paths.append(path)
    return paths


def choose_landmarks(paths, count, rng):
    """Include targets with unsupported isolated cells, then span cell levels."""
    by_size = defaultdict(set)
    for path in paths:
        for mask in path[1:]:
            if mask:
                by_size[mask.bit_count()].add(mask)
    isolated = sorted(mask for values in by_size.values() for mask in values
                      if not blocks.locally_supported(mask))
    quota = min(len(isolated), max(1, count // 3))
    chosen = [isolated[int(i)] for i in rng.choice(len(isolated), quota, replace=False)] if quota else []
    for mask in chosen:
        by_size[mask.bit_count()].remove(mask)
    sizes = sorted(by_size)
    while len(chosen) < count and sizes:
        for size in list(sizes):
            choices = sorted(by_size[size])
            if not choices:
                sizes.remove(size)
                continue
            item = choices[int(rng.integers(len(choices)))]
            by_size[size].remove(item)
            chosen.append(item)
            if len(chosen) == count:
                break
    return chosen


def candidate_pairs(paths, landmarks, pairs_per_state, rng):
    states = sorted({mask for path in paths for mask in path})
    pairs = set()
    for source in states:
        pairs.add((source, 0))
        contained = [goal for goal in landmarks if goal != source and goal & source == goal]
        if contained:
            sample = rng.choice(len(contained), min(pairs_per_state, len(contained)), replace=False)
            pairs.update((source, contained[int(i)]) for i in sample)
        # Non-subset negatives remain goal-conditioned rather than state labels.
        other = [goal for goal in landmarks if goal != source and goal & source != goal]
        if other:
            sample = rng.choice(len(other), min(2, len(other)), replace=False)
            pairs.update((source, other[int(i)]) for i in sample)
    for path in paths:
        for i, source in enumerate(path):
            for goal in path[i + 1:]:
                pairs.add((source, goal))
    return pairs


def contrast_cases(paths, oracle, limit, rng):
    """One parent, two legal successors and two nonempty goals with reversed order."""
    parents = list(dict.fromkeys(mask for path in paths for mask in path[:-2]))
    rng.shuffle(parents)
    result = []
    for parent in parents:
        next_states = list(dict.fromkeys(mask for _, mask in successors(parent) if mask))
        rng.shuffle(next_states)
        for a in next_states[:8]:
            a_goals = [goal for _, goal in successors(a) if goal]
            if not a_goals:
                continue
            for b in next_states[:8]:
                if a == b:
                    continue
                b_goals = [goal for _, goal in successors(b) if goal]
                if not b_goals:
                    continue
                ga = a_goals[int(rng.integers(len(a_goals)))]
                gb = b_goals[int(rng.integers(len(b_goals)))]
                da, db = oracle.distance(b, ga), oracle.distance(a, gb)
                if da < 0 and db < 0:
                    result.append((parent, a, b, ga, gb))
                    break
            if len(result) >= limit:
                return result
    return result


def prepare(config, out):
    """Generate all splits before training; test labels never select a checkpoint."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    options = config["data"]
    rng = np.random.default_rng(options["seed"])
    train_boards, ood_boards = official_boards(
        config["official_dataset"], options["seed"], options["train_boards"], options["ood_boards"])
    oracle = DistanceOracle(cache_limit=options["oracle_cache_limit"], seconds=options["oracle_seconds"])
    board_records = []
    for board in train_boards:
        paths = sample_paths(board, options["train_rollouts"] + options["validation_rollouts"], rng)
        train_paths = paths[:1 + options["train_rollouts"]]
        validation_paths = paths[1 + options["train_rollouts"]:]
        landmarks = choose_landmarks(train_paths, options["landmarks_per_board"] +
                                     options["heldout_landmarks_per_board"], rng)
        board_records.append((board, train_paths, validation_paths, landmarks))
    # A held-out landmark may still be a training source; it is never a training target.
    heldout_goals = {goal for _, _, _, landmarks in board_records
                     for goal in landmarks[:options["heldout_landmarks_per_board"]]}
    train_candidates, validation_candidates, goal_candidates = set(), set(), set()
    train_states = set()
    contrast_train, contrast_test = [], []
    for board_index, (board, train_paths, validation_paths, landmarks) in enumerate(board_records):
        local_states = {s for path in train_paths for s in path}
        train_states.update(local_states)
        train_landmarks = landmarks[options["heldout_landmarks_per_board"]:]
        local = candidate_pairs(train_paths, train_landmarks, options["landmark_pairs_per_state"], rng)
        train_candidates.update((s, t) for s, t in local if t not in heldout_goals)
        validation_candidates.update(candidate_pairs(validation_paths, landmarks, 2, rng))
        for goal in landmarks[:options["heldout_landmarks_per_board"]]:
            goal_candidates.update((source, goal) for source in local_states if source != goal)
        cases = contrast_cases(train_paths, oracle, options["contrast_cases_per_board"], rng)
        if board_index < options["contrast_training_boards"]:
            cases = [case for case in cases if case[3] not in heldout_goals and
                     case[4] not in heldout_goals]
        (contrast_train if board_index < options["contrast_training_boards"] else contrast_test).extend(cases)
        if board_index < options["contrast_training_boards"]:
            for _, a, b, ga, gb in cases:
                train_candidates.update(((a, ga), (b, ga), (a, gb), (b, gb)))
    heldout_relation_keys = {tuple(sorted(pair)) for pair in goal_candidates}
    train_candidates = {pair for pair in train_candidates
                        if pair[1] not in heldout_goals and
                        tuple(sorted(pair)) not in heldout_relation_keys}
    contrast_train_keys = {tuple(sorted((source, goal))) for _, a, b, ga, gb in contrast_train
                           for source, goal in ((a, ga), (b, ga), (a, gb), (b, gb))}
    contrast_test_keys = {tuple(sorted((source, goal))) for _, a, b, ga, gb in contrast_test
                          for source, goal in ((a, ga), (b, ga), (a, gb), (b, gb))}
    train_candidates = {pair for pair in train_candidates
                        if tuple(sorted(pair)) not in contrast_test_keys}
    # Training and validation state membership is global, including contrast goals.
    train_states.update(mask for pair in train_candidates for mask in pair)
    validation_candidates = {(s, t) for s, t in validation_candidates
                             if t not in heldout_goals and
                             (s not in train_states or t not in train_states)}
    goal_candidates = {(s, t) for s, t in goal_candidates if t not in
                       {goal for _, goal in train_candidates}}
    ood_candidates = set()
    for board in ood_boards:
        paths = sample_paths(board, options["ood_rollouts"], rng)
        landmarks = choose_landmarks(paths, options["landmarks_per_board"], rng)
        ood_candidates.update(candidate_pairs(paths, landmarks, options["landmark_pairs_per_state"], rng))
    # Reserve seen-state relations by unordered mask pair so reverse labels cannot leak.
    seen_candidates = []
    for pair in sorted(train_candidates):
        key = tuple(sorted(pair))
        if key[0] == key[1]:
            continue
        if key not in contrast_train_keys and (key[0] * 1315423911 + key[1]) % 10 == 0:
            seen_candidates.append(pair)
    seen_keys = {tuple(sorted(pair)) for pair in seen_candidates}
    train_candidates = {pair for pair in train_candidates if tuple(sorted(pair)) not in seen_keys}
    actually_seen = {mask for pair in train_candidates for mask in pair}
    seen_candidates = [pair for pair in seen_candidates if all(mask in actually_seen for mask in pair)]
    train_states = actually_seen
    validation_candidates = {(s, t) for s, t in validation_candidates
                             if t not in heldout_goals and
                             (s not in train_states or t not in train_states)}
    ood_candidates = {pair for pair in ood_candidates if any(s not in train_states for s in pair)}
    # Evaluation collections are deduplicated against training and each other.
    split_sets = {"train": train_candidates, "validation_unseen_state": validation_candidates,
                  "test_seen_pair": set(seen_candidates), "test_unseen_goal": goal_candidates,
                  "test_ood_board": ood_candidates}
    used = set()
    for name, values in split_sets.items():
        values.difference_update(used)
        used.update(values)
    train_targets = {goal for _, goal in split_sets["train"]}
    assert not (train_targets & heldout_goals)
    assert all(s in train_states and t in train_states
               for s, t in split_sets["test_seen_pair"])
    assert all(s not in train_states or t not in train_states
               for s, t in split_sets["validation_unseen_state"])
    assert all(goal not in train_targets for _, goal in split_sets["test_unseen_goal"])
    all_masks = sorted({s for values in split_sets.values() for pair in values for s in pair} |
                       {x for case in contrast_train + contrast_test for x in case})
    ids = {mask: i for i, mask in enumerate(all_masks)}
    payload = {"states": np.asarray([str(s) for s in all_masks]),
               "train_board_rows": np.asarray([b[0] for b in train_boards], dtype=np.int32),
               "ood_board_rows": np.asarray([b[0] for b in ood_boards], dtype=np.int32),
               "train_state_ids": np.asarray([ids[s] for s in sorted(train_states)], dtype=np.int32),
               "heldout_goal_ids": np.asarray([ids[s] for s in sorted(heldout_goals) if s in ids], dtype=np.int32),
               "contrast_train": np.asarray([[ids[s] for s in case] for case in contrast_train], dtype=np.int32).reshape(-1, 5),
               "contrast_test": np.asarray([[ids[s] for s in case] for case in contrast_test], dtype=np.int32).reshape(-1, 5)}
    summary = {"official_dataset": config["official_dataset"], "train_boards": len(train_boards),
               "ood_boards": len(ood_boards), "train_states": len(train_states),
               "heldout_landmarks": len(heldout_goals), "contrast_train": len(contrast_train),
               "contrast_test": len(contrast_test), "splits": {}}
    summary["heldout_isolated_landmarks"] = sum(not blocks.locally_supported(goal)
                                                for goal in heldout_goals)
    cap = max(board[1].bit_count() for board in train_boards) // 2 + 1
    payload["cap"] = np.asarray(cap)
    for name, values in split_sets.items():
        records = []
        for source, goal in sorted(values):
            if source == goal:
                continue
            d = oracle.distance(source, goal)
            kind = 0 if d >= 0 else 1 if goal & source != goal else 2
            records.append((ids[source], ids[goal], d, kind))
        payload[name] = np.asarray(records, dtype=np.int32).reshape(-1, 4)
        finite_nonempty = [(source, goal) for source, goal, d, _ in records
                           if d >= 0 and all_masks[goal] != 0]
        summary["splits"][name] = {"pairs": len(records), "distance_counts": dict(Counter(
            "unreachable" if row[2] < 0 else str(row[2]) for row in records)),
            "hard_unreachable": sum(row[3] == 2 for row in records),
            "finite_nonempty_goal_pairs": len(finite_nonempty),
            "finite_dead_goal_pairs": sum(oracle.cover(all_masks[goal]) < 0
                                          for _, goal in finite_nonempty)}
    unseen_goals = payload["test_unseen_goal"]
    isolated_ids = {ids[goal] for goal in heldout_goals if goal in ids and
                    not blocks.locally_supported(goal)}
    isolated_rows = unseen_goals[np.isin(unseen_goals[:, 1], list(isolated_ids))]
    payload["test_isolated_goal"] = isolated_rows
    summary["splits"]["test_isolated_goal"] = {
        "pairs": len(isolated_rows), "finite_pairs": int((isolated_rows[:, 2] >= 0).sum()),
        "hard_unreachable": int((isolated_rows[:, 3] == 2).sum())}
    task_pool = isolated_rows[isolated_rows[:, 2] >= 2]
    task_indices = rng.choice(len(task_pool), min(len(task_pool), options["isolated_goal_rollouts"]),
                              replace=False)
    task_rows = task_pool[task_indices]
    payload["isolated_goal_tasks"] = task_rows[:, :3]
    summary["isolated_goal_rollouts"] = len(task_rows)
    summary["states"] = len(all_masks)
    summary["oracle_cache_entries"] = len(oracle.cache)
    np.savez_compressed(out / "data.npz", **payload)
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    print(json.dumps(summary), flush=True)
    return summary
