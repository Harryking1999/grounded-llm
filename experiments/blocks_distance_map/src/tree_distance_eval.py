"""Exact state-pair distances on sealed fresh boards for Q distance plots."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

import numpy as np
import torch

from .model import BoardEncoder, board_bits
from .multiboard_data import official_boards, sample_paths
from .multiboard_eval import encoded_values, listwise_groups, pair_scores, split_metrics
from .oracle import DistanceOracle


def build(data_path, config_path, official_path, comparison_path, output_path):
    base = dict(np.load(data_path, allow_pickle=False))
    comparison = json.loads(Path(comparison_path).read_text())
    config = json.loads(Path(config_path).read_text())
    old_count = len(base["ood_board_rows"])
    skip = int(comparison.get("skip_fresh_boards", 0))
    count = len(comparison["board_rows"])
    _, ood = official_boards(official_path, config["board_seed"],
                             len(base["train_board_rows"]), old_count + skip + count)
    if [board[0] for board in ood[:old_count]] != base["ood_board_rows"].tolist():
        raise ValueError("Historical OOD split changed")
    fresh = ood[old_count + skip:]
    if [board[0] for board in fresh] != comparison["board_rows"]:
        raise ValueError("Distance evaluation boards differ from main evaluation")
    rng = np.random.default_rng(20261013)
    oracle = DistanceOracle(cache_limit=2_000_000, seconds=3600)
    masks, ids, pairs, pair_boards = [], {}, [], []

    def state_id(mask):
        if mask not in ids:
            ids[mask] = len(masks)
            masks.append(mask)
        return ids[mask]

    for board_index, board in enumerate(fresh):
        paths = sample_paths(board, 4, rng)
        by_size = defaultdict(list)
        for mask in {mask for path in paths for mask in path}:
            by_size[mask.bit_count()].append(mask)
        selected = {board[1], 0}
        for states in by_size.values():
            rng.shuffle(states)
            selected.update(states[:3])
        for source in selected:
            for goal in selected:
                if source == goal:
                    continue
                steps = oracle.distance(source, goal)
                kind = 0 if steps >= 0 else 1 if goal & source != goal else 2
                pairs.append((state_id(source), state_id(goal), steps, kind))
                pair_boards.append(board_index)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = np.asarray(pairs, dtype=np.int32)
    np.savez_compressed(output_path, states=np.asarray([str(mask) for mask in masks]),
                        test_ood_board=rows,
                        pair_board=np.asarray(pair_boards, dtype=np.int16))
    counts = Counter(str(step) if step >= 0 else "unreachable" for step in rows[:, 2])
    summary = {"boards": count, "states": len(masks), "pairs": len(rows),
               "true_distances": dict(counts)}
    output_path.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)
    return summary


def score(data_path, checkpoint_path):
    data = dict(np.load(data_path, allow_pickle=False))
    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model_config = saved["config"]["model"]
    model = BoardEncoder(model_config["state_dim"], model_config["encoder_hidden_dim"],
                         saved["cap"], saved["metric"])
    model.load_state_dict(saved["model"])
    values = encoded_values(model, board_bits(data["states"]))
    pairs = data["test_ood_board"]
    ranks = listwise_groups(pairs, 20261013, per_anchor=4)
    metrics = split_metrics(pairs, pair_scores(values, pairs, saved["metric"]),
                            ranks, saved["cap"])
    data_path = Path(data_path)
    data_path.with_name(data_path.stem + "_metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics), flush=True)
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--official", type=Path, required=True)
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    args = parser.parse_args()
    build(args.data, args.config, args.official, args.comparison, args.out)
    if args.checkpoint:
        score(args.out, args.checkpoint)


if __name__ == "__main__":
    main()
