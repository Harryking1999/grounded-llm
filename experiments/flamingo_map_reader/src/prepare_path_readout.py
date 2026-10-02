"""Broader first-turn comparisons with goals held out before Q augmentation."""

from argparse import ArgumentParser
from collections import Counter
from itertools import combinations
import json
from pathlib import Path

import numpy as np

from .data import first_turn_from_record, load_graph, shortest_move_counts


def non_tied_pairs(step):
    distances = step.candidate_map_distances
    return [(left, right) for left, right in combinations(range(1, len(distances) + 1), 2)
            if not np.isclose(distances[left - 1], distances[right - 1], rtol=1e-10, atol=1e-12)]


def select_records(environment, qmap, suite, graph_id, config, prior_records, seed):
    rng = np.random.default_rng(seed)
    options = config["readout_data"]
    distances = shortest_move_counts(environment.adjacency)
    reserved = set()
    previous_goals = set()
    for record in [*suite["cases"], *prior_records]:
        start, goal = int(record["start"]), int(record["goal"])
        reserved.update(((start, goal), (goal, start)))
    for record in prior_records:
        if record["split"] == "train":
            previous_goals.add(int(record["goal"]))
    if graph_id == config["unseen_test_graph"]:
        goals = {"test": set(range(len(distances)))}
    else:
        available = sorted(set(range(len(distances))) - previous_goals)
        held_out = set(rng.choice(available, options["held_out_goals_per_graph"], replace=False).tolist())
        goals = {"train": set(range(len(distances))) - held_out, "validation": held_out}
    records = []
    for split, allowed in goals.items():
        for length_text, requested in options[f"{split}_per_length_per_graph"].items():
            length = int(length_text)
            pairs = [(int(start), int(goal)) for start, goal in np.argwhere(distances == length)
                     if int(goal) in allowed and (int(start), int(goal)) not in reserved]
            rng.shuffle(pairs)
            accepted = []
            for start, goal in pairs:
                record = dict(graph_id=graph_id, start=start, goal=goal,
                              shortest_moves=length, split=split,
                              sample_seed=int(rng.integers(0, 2**63)))
                turn = first_turn_from_record(environment, qmap, record)
                candidates = non_tied_pairs(turn.step)
                if not candidates:
                    continue
                # Neither pair membership nor question order depends on which is closer.
                pair = candidates[int(rng.integers(len(candidates)))]
                if rng.integers(2):
                    pair = pair[::-1]
                record["candidate_pair"] = list(pair)
                record["evaluate"] = split != "train"
                accepted.append(record)
                if len(accepted) == requested:
                    break
            if len(accepted) != requested:
                raise ValueError(f"Insufficient {graph_id}/{split}/length-{length}: {len(accepted)}/{requested}")
            if split == "train":
                count = options["training_evaluation_per_length_per_graph"][length_text]
                for index in rng.choice(len(accepted), count, replace=False):
                    accepted[int(index)]["evaluate"] = True
            records.extend(accepted)
    train_goals = {r["goal"] for r in records if r["split"] == "train"}
    valid_goals = {r["goal"] for r in records if r["split"] == "validation"}
    assert not train_goals & valid_goals
    return records, {split: sorted(values) for split, values in goals.items()}


def main():
    parser = ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--prior-manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    prior = json.loads(args.prior_manifest.read_text(encoding="utf-8"))
    if Path(prior["source_root"]).resolve() != args.source_root.resolve():
        raise ValueError("Prior pilot must use the same frozen maps")
    if args.out.exists():
        raise FileExistsError(args.out)
    records, partitions = [], {}
    graph_ids = [*config["train_validation_graphs"], config["unseen_test_graph"]]
    for index, graph in enumerate(graph_ids):
        selected, goals = select_records(*load_graph(args.source_root, graph), graph, config,
            [r for r in prior["records"] if r["graph_id"] == graph], config["readout_data"]["seed"] + index)
        records.extend(selected)
        partitions[graph] = goals
    coverage = {}
    for split in ("train", "validation", "test"):
        subset = [r for r in records if r["split"] == split]
        coverage[split] = dict(records=len(subset),
            evaluation_records=sum(r["evaluate"] for r in subset),
            unique_goals=len({(r["graph_id"], r["goal"]) for r in subset}),
            unique_current_states=len({(r["graph_id"], r["start"]) for r in subset}),
            shortest_moves=dict(Counter(r["shortest_moves"] for r in subset)),
            graphs=dict(Counter(r["graph_id"] for r in subset)))
    manifest = dict(config=config, source_root=str(args.source_root.resolve()),
        prior_manifest=str(args.prior_manifest.resolve()), records=records,
        goal_partitions=partitions, coverage=coverage,
        split_boundary="goal identity within graph; Q swaps remain inside their source split",
        supervision="one arbitrary non-tied candidate pair and its Q swap per training record",
        evaluation="all non-tied candidate pairs, original and Q swap, in marked records")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(coverage), flush=True)


if __name__ == "__main__":
    main()
