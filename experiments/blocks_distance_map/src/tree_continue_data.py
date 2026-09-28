"""Extend tree supervision with complete empty-goal forks and long-goal forks."""
from collections import Counter
import json
from pathlib import Path

import numpy as np

from experiments.gcml_counterexamples.src import blocks
from .multiboard_data import official_boards, sample_paths
from .oracle import DistanceOracle, successors


CLEAR_FORK = 64
LONG_FORK = 128


def unordered_pair(source, goal):
    return (min(source, goal), max(source, goal))


def prepare(parent_data, official_path, out, options, max_boards=None):
    """Keep old rows/splits, add exact labels for entire legal successor sets."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    base = dict(np.load(parent_data, allow_pickle=False))
    states = [int(mask) for mask in base["states"]]
    state_ids = {mask: index for index, mask in enumerate(states)}
    old_train = base["train"]
    old_tags = base["train_relation_tags"].copy()
    pair_ids = {(int(source), int(goal)): index
                for index, (source, goal, _, _) in enumerate(old_train[:, :])}
    heldout_goals = {states[int(index)] for index in base["heldout_goal_ids"]}
    trained = set(base["train_state_ids"].tolist())
    protected_states = {states[int(index)]
                        for name in ("validation_unseen_state", "test_ood_board")
                        for row in base[name] for index in row[:2] if index not in trained}
    heldout_pairs = {unordered_pair(states[int(s)], states[int(g)])
                     for name in ("validation_unseen_state", "test_seen_pair",
                                  "test_unseen_goal", "test_ood_board")
                     for s, g, _, _ in base[name]}
    heldout_pairs.update(unordered_pair(states[int(child)], states[int(goal)])
                         for _, a, b, ga, gb in base["contrast_test"]
                         for child, goal in ((a, ga), (b, ga), (a, gb), (b, gb)))

    train_boards, ood_boards = official_boards(
        official_path, options["board_seed"], len(base["train_board_rows"]),
        len(base["ood_board_rows"]))
    if [row for row, _, _ in train_boards] != base["train_board_rows"].tolist():
        raise ValueError("The 1000 training boards changed")
    if [row for row, _, _ in ood_boards] != base["ood_board_rows"].tolist():
        raise ValueError("The held-out boards changed")
    if max_boards is not None:
        train_boards = train_boards[:max_boards]
    rng = np.random.default_rng(options["sample_seed"])
    oracle = DistanceOracle(cache_limit=options["oracle_cache_limit"],
                            seconds=options["oracle_seconds"])
    new_rows, new_tags = [], []
    clear_pair_ids, long_pair_ids = set(), set()
    clear_groups, long_groups = [], []
    coverage = Counter()

    def state_id(mask):
        if mask not in state_ids:
            state_ids[mask] = len(states)
            states.append(mask)
        return state_ids[mask]

    def add_pair(source, goal, label, tag):
        if (source == goal or source in protected_states or goal in protected_states
                or goal in heldout_goals or unordered_pair(source, goal) in heldout_pairs):
            coverage["protected_pairs_skipped"] += 1
            return None
        source_id, goal_id = state_id(source), state_id(goal)
        key = (source_id, goal_id)
        index = pair_ids.get(key)
        if index is None:
            index = len(old_train) + len(new_rows)
            pair_ids[key] = index
            kind = 0 if label >= 0 else 1 if goal & source != goal else 2
            new_rows.append((source_id, goal_id, label, kind))
            new_tags.append(tag)
        elif index < len(old_train):
            if int(old_train[index, 2]) != label:
                raise AssertionError("Existing oracle label differs")
            old_tags[index] |= tag
        else:
            if new_rows[index - len(old_train)][2] != label:
                raise AssertionError("Repeated oracle label differs")
            new_tags[index - len(old_train)] |= tag
        (clear_pair_ids if tag == CLEAR_FORK else long_pair_ids).add(index)
        return index

    def add_fork(parent, goal, tag):
        parent_distance = oracle.distance(parent, goal)
        if parent_distance < 2:
            return
        children = sorted({child for _, child in successors(parent)})
        candidates = []
        for child in children:
            label = oracle.distance(child, goal)
            index = add_pair(child, goal, label, tag)
            if index is not None:
                candidates.append((index, child, label))
        add_pair(parent, goal, parent_distance, tag)
        if not candidates:
            return
        best = [(index, child, label) for index, child, label in candidates
                if label == parent_distance - 1]
        worse = [(index, child, label) for index, child, label in candidates
                 if label != parent_distance - 1]
        if not best or not worse:
            coverage["forks_without_contrast"] += 1
            return
        if tag == CLEAR_FORK:
            coverage["clear_forks"] += 1
            coverage["clear_children"] += len(candidates)
        else:
            coverage["long_forks"] += 1
            coverage["long_children"] += len(candidates)
        coverage["hard_unreachable_children"] += sum(
            label < 0 and blocks.locally_supported(child ^ goal)
            for _, child, label in worse if child & goal == goal)
        # Every worse successor competes with a shortest-path successor.
        rng.shuffle(best)
        rng.shuffle(worse)
        worse.sort(key=lambda item: (item[2] < 0 and not blocks.locally_supported(
            item[1] ^ goal) if item[1] & goal == goal else True))
        groups = clear_groups if tag == CLEAR_FORK else long_groups
        for offset in range(0, len(worse), 3):
            alternatives = worse[offset:offset + 3]
            if len(alternatives) < 3:
                alternatives += worse[:3 - len(alternatives)]
            while len(alternatives) < 3:
                alternatives.append(worse[0])
            groups.append([best[(offset // 3) % len(best)][0],
                           *(item[0] for item in alternatives)])

    for _, board, reference in train_boards:
        path = sample_paths((0, board, reference), 0, rng)[0]
        clear_parents = list(dict.fromkeys(path[:-1]))
        for parent in path[:options["extra_clear_parents_per_board"]]:
            reachable_alternatives = [child for _, child in successors(parent)
                                      if child not in clear_parents and oracle.distance(child, 0) >= 2]
            if reachable_alternatives:
                clear_parents.append(reachable_alternatives[
                    int(rng.integers(len(reachable_alternatives)))])
        for parent in clear_parents:
            add_fork(parent, 0, CLEAR_FORK)
        for depth, parent in enumerate(path[:-1]):
            far_goals = [goal for goal in path[depth + 1:-1]
                         if oracle.distance(parent, goal) >= options["minimum_long_distance"]]
            if not far_goals:
                continue
            selected = far_goals[:1] + far_goals[-1:]
            for goal in dict.fromkeys(selected[:options["long_goals_per_parent"]]):
                add_fork(parent, goal, LONG_FORK)
        coverage["boards"] += 1

    # Historical 200 OOD boards serve only as development decisions. Final
    # confirmation uses later, untouched official boards.
    dev_children, dev_distances, dev_offsets, dev_parent_distances = [], [], [0], []
    for _, board, _ in ood_boards:
        parent_distance = oracle.distance(board, 0)
        children = sorted({child for _, child in successors(board)})
        dev_children.extend(state_id(child) for child in children)
        dev_distances.extend(oracle.distance(child, 0) for child in children)
        dev_offsets.append(len(dev_children))
        dev_parent_distances.append(parent_distance)

    payload = dict(base)
    payload["states"] = np.asarray([str(mask) for mask in states])
    extra = np.asarray(new_rows, dtype=np.int32).reshape(-1, 4)
    payload["train"] = np.concatenate([old_train, extra])
    payload["train_relation_tags"] = np.concatenate([
        old_tags, np.asarray(new_tags, dtype=np.uint8)])
    payload["train_state_ids"] = np.unique(np.concatenate([
        base["train_state_ids"], extra[:, :2].reshape(-1)]))
    payload["clear_pair_ids"] = np.asarray(sorted(clear_pair_ids), dtype=np.int32)
    payload["long_pair_ids"] = np.asarray(sorted(long_pair_ids), dtype=np.int32)
    payload["clear_rank_groups"] = np.asarray(clear_groups, dtype=np.int32).reshape(-1, 4)
    payload["long_rank_groups"] = np.asarray(long_groups, dtype=np.int32).reshape(-1, 4)
    payload["dev_clear_child_ids"] = np.asarray(dev_children, dtype=np.int32)
    payload["dev_clear_child_distances"] = np.asarray(dev_distances, dtype=np.int16)
    payload["dev_clear_offsets"] = np.asarray(dev_offsets, dtype=np.int32)
    payload["dev_clear_parent_distances"] = np.asarray(dev_parent_distances, dtype=np.int16)
    if set(payload["train"][:, 1].tolist()) & set(base["heldout_goal_ids"].tolist()):
        raise AssertionError("A held-out goal became a training target")
    if not len(clear_groups) or not len(long_groups):
        raise ValueError("New clear or long-distance contrast groups are empty")
    np.savez_compressed(out / "data.npz", **payload)
    summary = {"old_train_pairs": int(len(old_train)), "new_train_pairs": int(len(new_rows)),
               "all_train_pairs": int(len(payload["train"])),
               "clear_pair_ids": len(clear_pair_ids), "long_pair_ids": len(long_pair_ids),
               "clear_rank_groups": len(clear_groups), "long_rank_groups": len(long_groups),
               "all_states": len(states), "coverage": dict(coverage)}
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)
    return summary
