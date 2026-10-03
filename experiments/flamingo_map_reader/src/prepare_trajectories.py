"""Select physical tasks before augmentation; prepare long full-trajectory SFT."""

from argparse import ArgumentParser
from collections import Counter, defaultdict
import json
import multiprocessing
from pathlib import Path

import numpy as np
import torch

from . import data as graph_data
from . import blocks_data
from .blocks import FrozenBoardMap
from .trajectory_dataset import prepare_record


def select_graph(config, source):
    options = config["data"]
    env, qmap, _ = graph_data.load_graph(source, options["graph_id"])
    distance = graph_data.shortest_move_counts(env.adjacency)
    rng = np.random.default_rng(config["seed"])
    goals = rng.permutation(len(env.adjacency)).tolist()
    records, failures, offset, partitions = [], Counter(), 0, {}
    for split, goal_count in options["goal_partition"].items():
        selected_goals = goals[offset:offset + goal_count]
        offset += goal_count
        partitions[split] = selected_goals
        pools = {}
        for goal in selected_goals:
            starts = rng.permutation(len(env.adjacency)).tolist()
            pools[goal] = sorted((s for s in starts if distance[s, goal] >= options["minimum_shortest_moves"]),
                                  key=lambda s: -int(distance[s, goal]))
        accepted, target = 0, options["pairs"][split]
        while accepted < target:
            progressed = False
            for goal in selected_goals:
                while pools[goal]:
                    progressed = True
                    start = pools[goal].pop(0)
                    row = dict(split=split, group="same_map_new_goal" if split != "train" else "train",
                        graph_id=options["graph_id"], start=start, goal=goal,
                        shortest_moves=int(distance[start, goal]), sample_seed=int(rng.integers(2**32)))
                    if split == "train":
                        demo = graph_data.demonstration_from_record(env, qmap, row)
                        if not demo.success or len(demo.executed_path) - 1 > config["maximum_demonstration_actions"]:
                            failures["training_greedy_failed_or_over_limit"] += 1
                            continue
                    records.append(row)
                    accepted += 1
                    break
                if accepted == target:
                    break
            if not progressed:
                raise ValueError(f"Insufficient distinct long {split} tasks: {accepted}/{target}")
        records.extend(dict(split=split, group="initial_goal", graph_id=options["graph_id"],
            start=g, goal=g, shortest_moves=0, sample_seed=int(rng.integers(2**32))) for g in selected_goals)
    if offset != len(goals):
        raise ValueError("Goal partition must cover the fixed graph exactly")
    return records, dict(source_root=str(source.resolve()), goal_partition=partitions,
                         rejected_training=dict(failures))


def select_blocks(config, source_manifest, official, q_checkpoint):
    """Reuse certified long physical pairs, never reuse old reader weights/targets."""
    from experiments.blocks_distance_map.src.multiboard_data import official_boards
    from experiments.blocks_distance_map.src.oracle import DistanceOracle
    options = config["data"]
    old = json.loads(source_manifest.read_text())
    if Path(old["q_checkpoint"]).resolve() != q_checkpoint.resolve():
        raise ValueError("Physical seed trajectories require their original frozen Q")
    boards, fresh = official_boards(official, options["board_seed"], options["training_boards"],
                                    options["validation_new_boards"] + options["test_new_boards"])
    train = [dict(r, group="train") for r in old["records"] if r["split"] == "train"]
    counts = Counter(r["board_row"] for r in train)
    if (set(counts) != {b[0] for b in boards} or
            set(counts.values()) != {options["pairs_per_training_board"]} or
            any(r["shortest_moves"] < options["minimum_shortest_moves"] for r in train)):
        raise ValueError("Seed manifest does not provide the contracted 1000-board long coverage")
    records = train
    used = {(int(r["start"]), int(r["goal"])) for r in train}
    # A later SFT turn already supervises (current, goal), even when that pair
    # was not an initial task. Do not relabel that suffix as a new held-out pair.
    qmap = FrozenBoardMap.load(q_checkpoint)
    for index, record in enumerate(train):
        demo = blocks_data.demonstration_from_record(qmap, record, config["maximum_demonstration_actions"])
        if not demo.success:
            raise ValueError("Seed physical trajectory is no longer successful")
        used.update((t.step.current, t.step.goal) for t in demo.turns)
        if (index + 1) % 1000 == 0:
            print(json.dumps(dict(phase="exclude_training_suffixes", trajectories=index + 1)), flush=True)
    rng = np.random.default_rng(config["seed"])
    oracle = DistanceOracle(cache_limit=4_000_000, seconds=14_400)

    def add(row, split, group):
        records.append(dict(split=split, group=group, board_row=int(row[0]), start=str(row[1]),
            goal=str(row[2]), shortest_moves=int(row[3]), sample_seed=int(rng.integers(2**32))))
        used.add((row[1], row[2]))

    for i, board in enumerate(boards):
        needs = (["validation"] if i < options["validation_same_board_pairs"] else []) + ["test"]
        pool = blocks_data.reachable_pairs(board, options["sampling_rollouts"], rng, oracle,
                                           options["minimum_shortest_moves"])
        for split in needs:
            candidate = next((p for p in pool if p[:2] not in used), None)
            if candidate is None:
                raise ValueError(f"Board {board[0]} lacks a distinct long {split} pair")
            add((board[0], *candidate), split, "same_initial_board_new_pair")
        add((board[0], board[1], board[1], 0), "train", "initial_goal")
        if (i + 1) % 100 == 0:
            print(json.dumps(dict(phase="select_blocks_seen", boards=i + 1)), flush=True)
    if len(boards) != options["test_same_board_pairs"]:
        raise ValueError("Same-board test samples one pair from every training board")
    for i, board in enumerate(fresh):
        split = "validation" if i < options["validation_new_boards"] else "test"
        pool = blocks_data.reachable_pairs(board, options["sampling_rollouts"], rng, oracle,
                                           options["minimum_shortest_moves"])
        accepted = 0
        for candidate in pool:
            if candidate[:2] in used:
                continue
            add((board[0], *candidate), split, "new_initial_board")
            accepted += 1
            if accepted == options["pairs_per_new_board"]:
                break
        if accepted != options["pairs_per_new_board"]:
            raise ValueError(f"Board {board[0]} lacks enough distinct long evaluation pairs")
        add((board[0], board[1], board[1], 0), split, "initial_goal")
    return records, dict(q_checkpoint=str(q_checkpoint.resolve()),
        physical_pair_source=str(source_manifest.resolve()), official_data=str(official.resolve()),
        training_board_rows=[b[0] for b in boards], validation_board_rows=[b[0] for b in fresh[:options["validation_new_boards"]]],
        test_board_rows=[b[0] for b in fresh[options["validation_new_boards"]:]],
        rejected_training=old.get("rejected_training", {}))


# Per-record preparation is independent, so it runs in forked workers. The parent
# fills this in before creating the pool; children inherit it without pickling the
# tokenizer or the map evaluator.
_WORKER = {}


def prepare_one(record, index):
    """Prepare one trajectory and return everything the manifest aggregation needs."""
    config, tokenizer, root = _WORKER["config"], _WORKER["tokenizer"], _WORKER["root"]
    make_demo = _WORKER["make_demo"]
    record = dict(record, trajectory_id=f"{config['task']}_{index:06d}")
    demo = make_demo(record)
    if record["split"] == "train" and not demo.success:
        raise ValueError("Training pair no longer produces its certified complete trajectory")
    row = prepare_record(demo, record, config, tokenizer, root / (record["trajectory_id"] + ".pt"))
    payload = dict(row=row, candidate_counts=Counter(), step_counts=Counter(), state_goal_pairs=set())
    for step_id, turn in enumerate(demo.turns):
        payload["candidate_counts"][len(turn.step.candidate_actions)] += 1
        payload["step_counts"][step_id] += 1
        payload["state_goal_pairs"].add((turn.step.current, turn.step.goal))
    if record["split"] == "train":
        payload["visible"] = blocks_data.visible_states(demo)
    if demo.turns and not demo.turns[0].step.done:
        step = demo.turns[0].step
        payload["first_best"] = {step.candidate_actions[i - 1] for i in step.map_minimal_candidates}
        if record["split"] == "test":
            payload["by_start"] = int(record["start"])
    return payload


def prepare_star(job):
    """Module-level so the pool can reference it by name."""
    return prepare_one(*job)


def coverage(records, config):
    return {split: dict(physical_tasks=sum(r["split"] == split for r in records),
        length_counts=dict(Counter(r["shortest_moves"] for r in records if r["split"] == split)),
        action_turns=sum(r["decision_turns"] for r in records if r["split"] == split),
        numbered_trajectories=sum(r["variants"] for r in records if r["split"] == split),
        max_tokens=max(r["max_tokens"] for r in records if r["split"] == split),
        greedy_success=sum(r["greedy_success"] for r in records if r["split"] == split),
        repeats_per_fixed_variant=config["training"]["epochs"] if split == "train" else None)
        for split in ("train", "validation", "test")}


def main():
    parser = ArgumentParser()
    for arg in ("config", "model-path", "out", "source-root", "source-manifest", "official-data", "q-checkpoint", "q-training-data"):
        parser.add_argument("--" + arg, type=Path, required=arg in ("config", "model-path", "out"))
    parser.add_argument("--workers", type=int, default=1, help="Forked processes for per-record preparation")
    args = parser.parse_args()
    torch.set_num_threads(1)
    config = json.loads(args.config.read_text())
    if args.out.exists():
        raise FileExistsError(args.out)
    args.out.mkdir(parents=True)
    root = args.out / "trajectories"
    root.mkdir()
    if config["task"] == "graph":
        records, metadata = select_graph(config, args.source_root)
        env, qmap, _ = graph_data.load_graph(args.source_root, config["data"]["graph_id"])
        make_demo = lambda r: graph_data.demonstration_from_record(env, qmap, r)
    else:
        records, metadata = select_blocks(config, args.source_manifest, args.official_data, args.q_checkpoint)
        qmap = FrozenBoardMap.load(args.q_checkpoint)
        make_demo = lambda r: blocks_data.demonstration_from_record(qmap, r, config["maximum_demonstration_actions"])
    pairs = [(int(r["start"]), int(r["goal"])) for r in records]
    if len(set(pairs)) != len(pairs):
        raise ValueError("Physical start-goal pair leakage or duplicate task")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    _WORKER.update(config=config, tokenizer=tokenizer, root=root, make_demo=make_demo)
    result, seen, first_best, by_start = [], set(), {}, defaultdict(list)
    turn_coverage = {split: dict(candidate_counts=Counter(), step_counts=Counter(),
                                state_goal_pairs=set()) for split in ("train", "validation", "test")}
    jobs = [(record, index) for index, record in enumerate(records)]
    pool = multiprocessing.get_context("fork").Pool(args.workers) if args.workers > 1 else None
    stream = pool.imap(prepare_star, jobs, chunksize=1) if pool else map(prepare_star, jobs)
    try:
        # imap preserves job order, so records land in the manifest exactly as before.
        for index, payload in zip(range(len(records)), stream):
            record, row = records[index], payload["row"]
            stats = turn_coverage[record["split"]]
            stats["candidate_counts"].update(payload["candidate_counts"])
            stats["step_counts"].update(payload["step_counts"])
            stats["state_goal_pairs"].update(payload["state_goal_pairs"])
            if "visible" in payload:
                seen.update(payload["visible"])
            if "first_best" in payload:
                first_best[row["trajectory_id"]] = payload["first_best"]
                if "by_start" in payload:
                    by_start[payload["by_start"]].append(row["trajectory_id"])
            result.append(row)
            if (index + 1) % 100 == 0:
                print(json.dumps(dict(phase="prepare", completed=index + 1, total=len(records),
                                     latest_tokens=row["max_tokens"])), flush=True)
    finally:
        if pool is not None:
            pool.terminate()
            pool.join()
    for row in result:
        row["sft_start_seen"] = int(row["start"]) in seen
        row["sft_goal_seen"] = int(row["goal"]) in seen
    target_pairs, used_ids = [], set()
    for ids in by_start.values():
        for i, left in enumerate(ids):
            right = next((r for r in ids[i + 1:] if r not in used_ids and
                          first_best[left].isdisjoint(first_best[r])), None)
            if left not in used_ids and right:
                target_pairs.append([left, right])
                used_ids.update((left, right))
    target_pairs = target_pairs[:config["evaluation"]["target_pairs"]]
    manifest = dict(config=config, records=result, **metadata,
        coverage=coverage(result, config), natural_target_pairs=target_pairs,
        natural_target_pair_shortfall=max(0, config["evaluation"]["target_pairs"] - len(target_pairs)))
    manifest["turn_coverage"] = {split: dict(candidate_counts=dict(stats["candidate_counts"]),
        step_counts=dict(stats["step_counts"]), distinct_state_goal_pairs=len(stats["state_goal_pairs"]))
        for split, stats in turn_coverage.items()}
    manifest["greedy_baseline"] = {split: dict(attempts=sum(r["split"] == split for r in result),
        reached=sum(r["split"] == split and r["greedy_success"] for r in result),
        shortest=sum(r["split"] == split and r["greedy_success"] and r["action_count"] == r["shortest_moves"] for r in result))
        for split in ("validation", "test")}
    if args.q_training_data:
        blocks_data.record_q_training_coverage(manifest, args.q_training_data)
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest["coverage"]), flush=True)


if __name__ == "__main__":
    main()
