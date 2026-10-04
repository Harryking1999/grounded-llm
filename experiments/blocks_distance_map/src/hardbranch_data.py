"""Find goal-conditioned branches that survive cheap local feasibility checks."""
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np

from experiments.gcml_counterexamples.src import blocks
from .oracle import DistanceOracle, successors


def find_cases(data, split, max_pairs, max_cases, seed, oracle):
    """Return (parent, good, bad, goal, good_distance) masks, never IDs."""
    rows = data[split]
    states = [int(mask) for mask in data["states"]]
    eligible = np.flatnonzero((rows[:, 2] >= 2) & (data["states"][rows[:, 1]] != "0"))
    rng = np.random.default_rng(seed)
    rng.shuffle(eligible)
    cases = []
    counts = Counter(eligible_total=len(eligible), pair_budget=min(max_pairs, len(eligible)))
    for index in eligible[:max_pairs]:
        parent_id, goal_id, parent_distance, _ = rows[index]
        parent, goal = states[parent_id], states[goal_id]
        by_area = {}
        for _, child in successors(parent):
            if child & goal != goal or not blocks.locally_supported(child ^ goal):
                continue
            area = child.bit_count()
            finite, dead = by_area.setdefault(area, ([], []))
            child_distance = oracle.distance(child, goal)
            if child_distance == parent_distance - 1:
                finite.append(child)
            elif child_distance < 0:
                dead.append(child)
        counts["scanned"] += 1
        eligible_areas = [area for area, (finite, dead) in by_area.items() if finite and dead]
        if not eligible_areas:
            continue
        area = eligible_areas[int(rng.integers(len(eligible_areas)))]
        finite, dead = by_area[area]
        good = finite[int(rng.integers(len(finite)))]
        bad = dead[int(rng.integers(len(dead)))]
        cases.append((parent, good, bad, goal, int(parent_distance - 1)))
        counts["cases"] += 1
        if len(cases) >= max_cases:
            break
    counts["distinct_parents"] = len({case[0] for case in cases})
    counts["distinct_goals"] = len({case[3] for case in cases})
    return cases, dict(counts)


def pair_key(source, goal):
    return (min(source, goal), max(source, goal))


def prepare_augmented(base_data_path, out, options):
    """Keep the original split intact and add only certified train-split branches."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    base = dict(np.load(base_data_path, allow_pickle=False))
    old_masks = [int(mask) for mask in base["states"]]
    oracle = DistanceOracle(cache_limit=options["oracle_cache_limit"],
                            seconds=options["oracle_seconds"])
    found, scan = {}, {}
    for name, split in (("validation", "validation_unseen_state"),
                        ("ood_board", "test_ood_board"),
                        ("unseen_goal", "test_unseen_goal"), ("train", "train")):
        spec = options["sampling"][name]
        found[name], scan[name] = find_cases(base, split, spec["max_pairs"],
                                             spec["max_cases"], spec["seed"], oracle)
    train_keys = {pair_key(old_masks[s], old_masks[g]) for s, g, _, _ in base["train"]}
    heldout_keys = {pair_key(old_masks[s], old_masks[g])
                    for name in ("validation_unseen_state", "test_seen_pair",
                                 "test_unseen_goal", "test_ood_board")
                    for s, g, _, _ in base[name]}
    heldout_keys.update(pair_key(old_masks[child], old_masks[goal])
                        for _, a, b, ga, gb in base["contrast_test"]
                        for child, goal in ((a, ga), (b, ga), (a, gb), (b, gb)))
    originally_trained = set(base["train_state_ids"].tolist())
    protected_new_masks = {old_masks[i]
                           for name in ("validation_unseen_state", "test_ood_board")
                           for row in base[name] for i in row[:2]
                           if i not in originally_trained}
    for name in ("validation", "ood_board", "unseen_goal"):
        found[name] = [case for case in found[name]
                       if all(pair_key(child, case[3]) not in train_keys
                              for child in case[1:3])]
        heldout_keys.update(pair_key(child, case[3])
                            for case in found[name] for child in case[1:3])
    found["train"] = [case for case in found["train"]
                      if all(pair_key(child, case[3]) not in heldout_keys
                             and child not in protected_new_masks
                             for child in case[1:3])]
    if len(found["train"]) < options["minimum_train_cases"] or any(
            len(found[name]) < options["minimum_test_cases"]
            for name in ("validation", "ood_board", "unseen_goal")):
        raise ValueError("Not enough disjoint hard-branch cases")
    masks = old_masks[:]
    ids = {mask: i for i, mask in enumerate(masks)}
    for cases in found.values():
        for case in cases:
            for mask in case[:4]:
                if mask not in ids:
                    ids[mask] = len(masks)
                    masks.append(mask)
    payload = dict(base)
    payload["states"] = np.asarray([str(mask) for mask in masks])
    for name, cases in found.items():
        payload[f"hard_{name}_cases"] = np.asarray(
            [[ids[parent], ids[good], ids[bad], ids[goal], steps]
             for parent, good, bad, goal, steps in cases], dtype=np.int32).reshape(-1, 5)
    additions = {(ids[child], ids[goal], child_steps, kind)
                 for parent, good, bad, goal, good_steps in found["train"]
                 for child, child_steps, kind in ((good, good_steps, 0), (bad, -1, 2))}
    new_rows = np.asarray(sorted(additions), dtype=np.int32).reshape(-1, 4)
    base_rows = {tuple(row) for row in base["train"].tolist()}
    payload["train"] = np.concatenate((base["train"],
                                        np.repeat(new_rows, options["hard_pair_repeat"], axis=0)))
    payload["train_state_ids"] = np.asarray(sorted(
        set(base["train_state_ids"].tolist()) |
        {ids[mask] for case in found["train"] for mask in case[:4]}), dtype=np.int32)
    trained_ids = set(payload["train_state_ids"].tolist())
    if any(all(int(i) in trained_ids for i in row[:2])
           for name in ("validation_unseen_state", "test_ood_board")
           for row in base[name]):
        raise ValueError("Augmentation erased an unseen-state validation or OOD relation")
    if set(payload["train"][:, 1].tolist()) & set(base["test_unseen_goal"][:, 1].tolist()):
        raise ValueError("A held-out goal became a training target")
    summary = {"base_data": str(base_data_path), "base_train_pairs": len(base["train"]),
               "hard_unique_pairs": len(new_rows),
               "previously_unseen_train_pairs": sum(tuple(row) not in base_rows for row in new_rows.tolist()),
               "hard_pair_repeat": options["hard_pair_repeat"],
               "augmented_train_rows": len(payload["train"]), "base_states": len(old_masks),
               "augmented_states": len(masks), "scan": scan,
               "kept_cases": {name: len(cases) for name, cases in found.items()},
               "oracle_cache_entries": len(oracle.cache)}
    np.savez_compressed(out / "data.npz", **payload)
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)
    return summary


def probe(data_path, split, max_pairs, max_cases, seed):
    data = dict(np.load(data_path, allow_pickle=False))
    oracle = DistanceOracle(cache_limit=2_000_000, seconds=3600)
    cases, counts = find_cases(data, split, max_pairs, max_cases, seed, oracle)
    result = {"split": split, "counts": counts,
              "sample": [{"parent": str(parent), "good": str(good), "bad": str(bad),
                          "goal": str(goal), "good_distance": steps}
                         for parent, good, bad, goal, steps in cases[:3]]}
    print(json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("probe", "prepare"), default="probe")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--split", default="train")
    parser.add_argument("--max-pairs", type=int, default=10000)
    parser.add_argument("--max-cases", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260929)
    args = parser.parse_args()
    if args.mode == "prepare":
        if args.config is None or args.out is None:
            parser.error("--mode prepare requires --config and --out")
        prepare_augmented(args.data, args.out, json.loads(args.config.read_text()))
    else:
        probe(args.data, args.split, args.max_pairs, args.max_cases, args.seed)


if __name__ == "__main__":
    main()
