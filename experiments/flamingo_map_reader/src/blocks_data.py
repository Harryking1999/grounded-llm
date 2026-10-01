"""Sample reachable blocks pairs and split against all states visible in SFT."""

from collections import Counter
import json
from pathlib import Path

import numpy as np

from experiments.blocks_distance_map.src.multiboard_data import official_boards, sample_paths
from experiments.blocks_distance_map.src.oracle import DistanceOracle
from .blocks_sft import greedy_demonstration


TEST_GROUPS = ("unseen_start_unseen_goal", "seen_start_unseen_goal", "unseen_start_seen_goal")


def reachable_pairs(board, rollouts, rng, oracle, minimum_moves):
    # Actual rollout order certifies reachability, including nonempty goals.
    paths = sample_paths(board, rollouts, rng)
    pairs = sorted({(source, goal) for path in paths
                    for i, source in enumerate(path[:-1]) for goal in path[i + 1:]})
    rng.shuffle(pairs)
    scored = [(start, goal, oracle.distance(start, goal)) for start, goal in pairs]
    # Stable sorting randomizes ties; shortest distance measures task length,
    # not how many redundant moves the sampled rollout happened to take.
    return sorted((pair for pair in scored if pair[2] >= minimum_moves),
                  key=lambda pair: -pair[2])


def visible_states(demonstration):
    return {state for turn in demonstration.turns
            for state in (turn.step.current, turn.step.goal, *turn.step.candidate_destinations)}


def coverage_group(start, goal, seen):
    if start in seen and goal in seen:
        return None
    return ("seen_start_unseen_goal" if start in seen else
            "unseen_start_seen_goal" if goal in seen else "unseen_start_unseen_goal")


def build_manifest(boards, fresh_boards, qmap, config):
    options = config["data"]
    if len(boards) != options["training_boards"]:
        raise ValueError("training board count differs from configuration")
    rng = np.random.default_rng(config["seed"])
    records, seen, pairs_used = [], set(), set()
    rejected = Counter()
    oracle = DistanceOracle(cache_limit=4_000_000, seconds=14_400)
    for board in boards:
        accepted = 0
        for start, goal, shortest_moves in reachable_pairs(board, options["sampling_rollouts"], rng,
                                                          oracle, options["minimum_shortest_moves"]):
            if (start, goal) in pairs_used:
                continue
            sample_seed = int(rng.integers(2**32))
            demonstration = greedy_demonstration(qmap, start, goal,
                rng=np.random.default_rng(sample_seed),
                max_actions=config["maximum_demonstration_actions"])
            if not demonstration.success:
                rejected["map_greedy_failed"] += 1
                continue
            records.append({"split": "train", "board_row": int(board[0]),
                            "start": str(start), "goal": str(goal),
                            "sample_seed": sample_seed, "shortest_moves": shortest_moves,
                            "demonstration_moves": len(demonstration.executed_path) - 1})
            seen.update(visible_states(demonstration))
            pairs_used.add((start, goal))
            accepted += 1
            if accepted == options["pairs_per_training_board"]:
                break
        if accepted != options["pairs_per_training_board"]:
            raise ValueError(f"Board {board[0]} supplied only {accepted} successful SFT pairs")

    # Freeze the full SFT state set before choosing test examples. Never use Q
    # scores or rollout success to accept/reject a test task.
    pools = {group: [] for group in TEST_GROUPS}
    test_pairs = set()
    for board in [*boards, *fresh_boards]:
        for start, goal, shortest_moves in reachable_pairs(board, options["sampling_rollouts"], rng,
                                                          oracle, options["minimum_shortest_moves"]):
            group = coverage_group(start, goal, seen)
            if group is None or (start, goal) in pairs_used or (start, goal) in test_pairs:
                continue
            test_pairs.add((start, goal))
            pools[group].append({"split": "test", "group": group,
                                 "board_row": int(board[0]), "start": str(start),
                                 "goal": str(goal), "shortest_moves": shortest_moves,
                                 "sample_seed": int(rng.integers(2**32))})
    for group, pool in pools.items():
        count = options["test_pairs_per_group"]
        if len(pool) < count:
            raise ValueError(f"Only {len(pool)} eligible pairs for {group}; need {count}")
        rng.shuffle(pool)
        pool.sort(key=lambda record: -record["shortest_moves"])
        records.extend(pool[:count])
    return {"config": config, "records": records,
            "training_visible_states": [str(s) for s in sorted(seen)],
            "training_board_rows": [int(b[0]) for b in boards],
            "fresh_board_rows": [int(b[0]) for b in fresh_boards],
            "rejected_training": dict(rejected),
            "length_counts": {split: dict(Counter(r["shortest_moves"] for r in records
                                                   if r["split"] == split))
                              for split in ("train", "test")},
            "test_pool_sizes": {group: len(pool) for group, pool in pools.items()},
            "test_goal_nonempty_counts": dict(Counter(record["group"] for record in records
                 if record["split"] == "test" and int(record["goal"]) != 0))}


def record_q_training_coverage(manifest, archive_path):
    """Keep map-training exposure separate from interface SFT exposure."""
    with np.load(archive_path, allow_pickle=False) as data:
        ids = set(map(int, data["train"][:, :2].ravel()))
        if "contrast_train" in data:
            ids.update(map(int, data["contrast_train"].ravel()))
        states = data["states"]
        seen = {int(states[i]) for i in ids}
    for record in manifest["records"]:
        record["q_training_start_seen"] = int(record["start"]) in seen
        record["q_training_goal_seen"] = int(record["goal"]) in seen
    manifest["q_training_data"] = str(Path(archive_path).resolve())
    manifest["q_training_seen_states"] = len(seen)


def demonstration_from_record(qmap, record, max_actions):
    return greedy_demonstration(qmap, int(record["start"]), int(record["goal"]),
                                rng=np.random.default_rng(record["sample_seed"]),
                                max_actions=max_actions)


def main():
    from argparse import ArgumentParser
    from .blocks import FrozenBoardMap
    parser = ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--official-data", type=Path, required=True)
    parser.add_argument("--q-checkpoint", type=Path, required=True)
    parser.add_argument("--q-training-data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    options = config["data"]
    boards, fresh = official_boards(args.official_data, options["board_seed"],
                                    options["training_boards"], options["fresh_board_pool"])
    qmap = FrozenBoardMap.load(args.q_checkpoint, args.device)
    manifest = build_manifest(boards, fresh, qmap, config)
    manifest["q_checkpoint"] = str(args.q_checkpoint.resolve())
    record_q_training_coverage(manifest, args.q_training_data)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"records": len(manifest["records"]),
                      "rejected_training": manifest["rejected_training"],
                      "test_pool_sizes": manifest["test_pool_sizes"]}))


if __name__ == "__main__":
    main()
