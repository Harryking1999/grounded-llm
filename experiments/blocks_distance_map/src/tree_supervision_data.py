"""Goal-relative ancestor and contradictory-branch supervision on sampled removal DAGs."""
from collections import Counter, defaultdict
import json
from pathlib import Path

import numpy as np

from .multiboard_data import official_boards, sample_paths
from .oracle import DistanceOracle, successors


ANCESTOR = 1
GOAL_SWAP = 2
CROSS_LEAF = 4
SIBLING_GOAL = 8


def add_relation(relations, source, goal, tag):
    if source != goal:
        key = (source, goal)
        relations[key] = relations.get(key, 0) | tag


def add_path(relations, path):
    for index, source in enumerate(path):
        for goal in path[index + 1:]:
            add_relation(relations, source, goal, ANCESTOR)


def fork_parents(paths, limit, rng):
    """Spread selected fork roots across depths of the sampled trajectories."""
    levels = defaultdict(list)
    seen = set()
    for path in paths:
        for depth, parent in enumerate(path[:-1]):
            if parent in seen or depth > 5 or parent.bit_count() < 6:
                continue
            seen.add(parent)
            levels[depth].append((path[:depth + 1], path[depth + 1]))
    for items in levels.values():
        rng.shuffle(items)
    selected = []
    while len(selected) < limit and any(levels.values()):
        for depth in sorted(levels):
            if levels[depth]:
                selected.append((depth, *levels[depth].pop()))
                if len(selected) >= limit:
                    break
    return selected


def grow_branch(child, depth, rng):
    branch = [child]
    for _ in range(depth - 1):
        choices = sorted({mask for _, mask in successors(branch[-1]) if mask})
        if not choices:
            break
        branch.append(choices[int(rng.integers(len(choices)))])
    return branch


def add_board_graph(relations, board, options, rng, oracle):
    paths = sample_paths(board, options["backbone_rollouts"], rng)
    for path in paths:
        add_path(relations, path)
    stats = Counter(backbone_paths=len(paths))
    accepted = []
    for depth, prefix, preferred_child in fork_parents(
            paths, options["fork_attempts_per_board"], rng):
        if stats["accepted_parents"] >= options["accepted_parents_per_board"]:
            break
        parent = prefix[-1]
        children = sorted({mask for _, mask in successors(parent) if mask})
        if len(children) < 2:
            continue
        rng.shuffle(children)
        chosen = ([preferred_child] if preferred_child in children else [])
        chosen.extend(mask for mask in children if mask not in chosen)
        chosen = chosen[:options["branch_width"]]
        branches = []
        for child in chosen:
            branch = grow_branch(child, options["branch_depth"], rng)
            if len(branch) < 2:
                continue
            add_path(relations, prefix + branch)
            branches.append((child, branch[-1]))
        if len(branches) < 2:
            continue
        goals = set()
        local_cases = []
        for index, (a, ga) in enumerate(branches):
            for b, gb in branches[index + 1:]:
                if ga == gb or oracle.distance(b, ga) >= 0 or oracle.distance(a, gb) >= 0:
                    stats["merged_or_reachable_pairs"] += 1
                    continue
                if oracle.distance(a, ga) < 0 or oracle.distance(b, gb) < 0:
                    raise AssertionError("A sampled branch lost its own descendant")
                for source, goal in ((a, ga), (b, gb)):
                    add_relation(relations, source, goal, GOAL_SWAP)
                for source, goal in ((a, gb), (b, ga)):
                    add_relation(relations, source, goal, GOAL_SWAP)
                if oracle.distance(ga, gb) < 0 and oracle.distance(gb, ga) < 0:
                    add_relation(relations, ga, gb, CROSS_LEAF)
                    add_relation(relations, gb, ga, CROSS_LEAF)
                    stats["mutually_unreachable_leaf_pairs"] += 1
                local_cases.append((parent, a, b, ga, gb, depth))
                goals.update((ga, gb))
        if not local_cases:
            continue
        stats["accepted_parents"] += 1
        stats[f"fork_depth_{depth}"] += 1
        stats["goal_swap_cases"] += len(local_cases)
        accepted.extend(local_cases)
        selected_goals = sorted(goals)
        rng.shuffle(selected_goals)
        for goal in selected_goals[:options["goals_per_parent"]]:
            for child in children:
                add_relation(relations, child, goal, SIBLING_GOAL)
    return accepted, stats


def pair_key(source, goal):
    return (min(source, goal), max(source, goal))


def prepare_tree(base_path, official_path, out, options, max_boards=None):
    """Replace training rows only; preserve the original validation and OOD tasks."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    base = dict(np.load(base_path, allow_pickle=False))
    original = [int(mask) for mask in base["states"]]
    train_boards, ood_boards = official_boards(
        official_path, options["board_seed"], len(base["train_board_rows"]),
        len(base["ood_board_rows"]))
    if not np.array_equal([board[0] for board in train_boards], base["train_board_rows"]):
        raise ValueError("Training initial boards differ from the original 1000-board split")
    if not np.array_equal([board[0] for board in ood_boards], base["ood_board_rows"]):
        raise ValueError("OOD initial boards differ from the original 200-board split")
    if max_boards is not None:
        train_boards = train_boards[:max_boards]
    heldout_goals = {original[int(i)] for i in base["heldout_goal_ids"]}
    old_trained = set(base["train_state_ids"].tolist())
    protected_states = {original[int(i)]
                        for name in ("validation_unseen_state", "test_ood_board")
                        for row in base[name] for i in row[:2] if i not in old_trained}
    heldout_keys = {pair_key(original[int(s)], original[int(g)])
                    for name in ("validation_unseen_state", "test_seen_pair",
                                 "test_unseen_goal", "test_ood_board")
                    for s, g, _, _ in base[name]}
    heldout_keys.update(pair_key(original[int(child)], original[int(goal)])
                        for _, a, b, ga, gb in base["contrast_test"]
                        for child, goal in ((a, ga), (b, ga), (a, gb), (b, gb)))
    relations = {}
    rng = np.random.default_rng(options["sample_seed"])
    oracle = DistanceOracle(cache_limit=options["oracle_cache_limit"],
                            seconds=options["oracle_seconds"])
    board_stats = []
    cases = []
    for board in train_boards:
        local_cases, stats = add_board_graph(relations, board, options, rng, oracle)
        cases.extend(local_cases)
        board_stats.append(stats)
    for _, a, b, ga, gb in base["contrast_train"]:
        a, b, ga, gb = (original[int(i)] for i in (a, b, ga, gb))
        for source, goal in ((a, ga), (b, ga), (a, gb), (b, gb)):
            add_relation(relations, source, goal, GOAL_SWAP)
    proposal_count = len(relations)
    relations = {(s, g): tags for (s, g), tags in relations.items()
                 if g not in heldout_goals and s not in protected_states
                 and g not in protected_states and pair_key(s, g) not in heldout_keys}
    if max_boards is None and (len(cases) < options["minimum_goal_swap_cases"] or
                               len(relations) < options["minimum_train_relations"]):
        raise ValueError("Tree supervision did not reach its stated coverage")
    masks = original[:]
    ids = {mask: i for i, mask in enumerate(masks)}
    for source, goal in relations:
        for mask in (source, goal):
            if mask not in ids:
                ids[mask] = len(masks)
                masks.append(mask)
    rows, tags_array = [], []
    source_counts = Counter()
    label_counts = Counter()
    for (source, goal), tags in sorted(relations.items()):
        steps = oracle.distance(source, goal)
        kind = 0 if steps >= 0 else 1 if goal & source != goal else 2
        rows.append((ids[source], ids[goal], steps, kind))
        tags_array.append(tags)
        label_counts["finite" if steps >= 0 else "unreachable"] += 1
        if steps < 0:
            label_counts["missing_goal_cells" if kind == 1 else
                         "contained_but_unreachable"] += 1
        for flag, name in ((ANCESTOR, "ancestor"), (GOAL_SWAP, "goal_swap"),
                           (CROSS_LEAF, "cross_leaf"), (SIBLING_GOAL, "sibling_goal")):
            if tags & flag:
                source_counts[name] += 1
                source_counts[f"{name}_{'finite' if steps >= 0 else 'unreachable'}"] += 1
                if steps < 0:
                    source_counts[f"{name}_{'missing_goal_cells' if kind == 1 else 'contained_but_unreachable'}"] += 1
        if tags & ANCESTOR and steps < 0:
            raise AssertionError("An ancestor-descendant relation is unreachable")
        if tags & CROSS_LEAF and steps >= 0:
            raise AssertionError("A certified cross-branch leaf relation is reachable")
    trained_ids = {i for row in rows for i in row[:2]}
    retained_cases = [case for case in cases
                      if all(pair in relations for pair in
                             ((case[1], case[3]), (case[2], case[4]),
                              (case[1], case[4]), (case[2], case[3])))]
    payload = dict(base)
    payload["states"] = np.asarray([str(mask) for mask in masks])
    payload["train"] = np.asarray(rows, dtype=np.int32).reshape(-1, 4)
    payload["train_relation_tags"] = np.asarray(tags_array, dtype=np.uint8)
    payload["train_state_ids"] = np.asarray(sorted(trained_ids), dtype=np.int32)
    payload["contrast_train"] = np.asarray(
        [[ids[mask] for mask in case[:5]] for case in retained_cases],
        dtype=np.int32).reshape(-1, 5)
    seen = base["test_seen_pair"]
    payload["test_seen_pair"] = seen[np.isin(seen[:, 0], list(trained_ids)) &
                                     np.isin(seen[:, 1], list(trained_ids))]
    if set(payload["train"][:, 1].tolist()) & set(base["heldout_goal_ids"].tolist()):
        raise AssertionError("A held-out goal became a training target")
    if any(all(int(i) in trained_ids for i in row[:2])
           for row in base["validation_unseen_state"]):
        raise AssertionError("Training erased an unseen-state validation row")
    if any(all(int(i) in trained_ids for i in row[:2])
           for row in base["test_ood_board"]):
        raise AssertionError("Training erased an OOD row")
    summary = {"base_data": str(base_path), "official_dataset": str(official_path),
               "train_boards": len(train_boards), "ood_boards": len(ood_boards),
               "proposed_relations": proposal_count, "train_relations": len(rows),
               "train_states": len(trained_ids), "all_states": len(masks),
               "goal_swap_cases": len(cases), "retained_goal_swap_cases": len(retained_cases),
               "retained_test_seen_pairs": len(payload["test_seen_pair"]),
               "relation_sources": dict(source_counts), "labels": dict(label_counts),
               "board_coverage": {"min_accepted_parents": min(
                   (item["accepted_parents"] for item in board_stats), default=0),
                   "median_accepted_parents": float(np.median(
                       [item["accepted_parents"] for item in board_stats])) if board_stats else 0,
                   "total_accepted_parents": sum(item["accepted_parents"] for item in board_stats),
                   "fork_depths": dict(sum((Counter({key: value for key, value in item.items()
                                                     if key.startswith("fork_depth_")})
                                       for item in board_stats), Counter()))},
               "oracle_cache_entries": len(oracle.cache)}
    np.savez_compressed(out / "data.npz", **payload)
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)
    return summary
