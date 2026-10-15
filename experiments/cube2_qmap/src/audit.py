"""Inspect exact decision structure, relative coverage and frozen-Q fit."""
import argparse
import json
from pathlib import Path

import numpy as np

from .environment import Cube2


def audit(run, out):
    env = Cube2(run / "oracle_depth.npy")
    data = dict(np.load(run / "data.npz", allow_pickle=False))
    layer_edges = np.zeros((12, 12), dtype=np.int64)
    optimal_counts = np.zeros((12, 10), dtype=np.int64)
    for begin in range(0, env.size, 100000):
        ids = np.arange(begin, min(env.size, begin + 100000))
        depth = env.depth[ids]
        child_depth = env.depth[env.successors(ids)]
        layer_edges += np.bincount((depth[:, None] * 12 + child_depth).ravel(), minlength=144).reshape(12, 12)
        good = (child_depth.astype(int) == depth[:, None].astype(int) - 1).sum(axis=1)
        optimal_counts += np.bincount(depth * 10 + good, minlength=120).reshape(12, 10)
    relatives = {}
    for kind in ("pairs_train_base", "pairs_train", "tasks_train_base", "tasks_train"):
        rows = data[kind]
        relatives[kind] = np.unique(env.relative(rows[:, 1], rows[:, 0]))
    represented = np.zeros(env.size, dtype=bool)
    represented[relatives["tasks_train"]] = True
    rng = np.random.default_rng(20261016)
    by_depth = {}
    for d in range(1, 12):
        layer = np.flatnonzero(env.depth == d)
        sampled = rng.choice(layer, min(1000, len(layer)), replace=False)
        current = sampled.copy()
        covered, all_covered = 0, np.ones(len(sampled), dtype=bool)
        for _ in range(d):
            seen = represented[current]
            covered += int(seen.sum())
            all_covered &= seen
            children = env.successors(current)
            child_depth = env.depth[children]
            current = children[np.arange(len(current)), child_depth.argmin(axis=1)]
        by_depth[str(d)] = {
            "relative_states": len(layer),
            "pair_relative_types": int(np.isin(relatives["pairs_train"], layer).sum()),
            "decision_relative_types": int(represented[layer].sum()),
            "decision_relative_coverage": float(represented[layer].mean()),
            "optimal_action_count_histogram": optimal_counts[d].tolist(),
            "uniform_random_shortest_action_rate": float((optimal_counts[d] * np.arange(10)).sum() / (len(layer) * 9)),
            "outgoing_layer_counts": layer_edges[d].tolist(),
            "sampled_shortest_paths": len(sampled),
            "supervised_fraction_on_sampled_path_nodes": covered / (len(sampled) * d),
            "fully_supervised_sampled_paths": int(all_covered.sum())}
    result = {"reference_run": str(run), "relative_types": {k: len(v) for k, v in relatives.items()},
              "by_depth": by_depth,
              "structure": {"states": env.size, "degree": 9, "diameter": 11,
                            "maximum_neighbor_depth_difference": int(max(abs(i - j) for i, j in zip(*np.nonzero(layer_edges))))}}
    # Fit on sampled training rows versus the actual heldout set locates a data
    # failure without treating a larger data set as proof of representability.
    import torch
    from experiments.puzzle_qmap.src.model import MapReader
    from experiments.puzzle_qmap.src.evaluate import pair_metrics, decisions
    torch.set_num_threads(4)
    reader = MapReader(env, run / "best.pt")
    result["frozen_q"] = {}
    for split in ("train", "test"):
        pairs, tasks = data[f"pairs_{split}"], data[f"tasks_{split}"]
        pairs = pairs[rng.choice(len(pairs), min(6000, len(pairs)), replace=False)]
        tasks = tasks[rng.choice(len(tasks), min(1000, len(tasks)), replace=False)]
        result["frozen_q"][split] = {"pairs": pair_metrics(reader.model, env, pairs, "cpu"),
                                         "decisions": decisions(reader.model, env, tasks, "cpu")["q"],
                                         "decisions_by_depth": {str(d): decisions(reader.model, env,
                                             tasks[env.distance(tasks[:, 0], tasks[:, 1]) == d], "cpu")["q"]
                                             for d in range(1, 12)}}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"relative_types": result["relative_types"],
                      "fit": {k: {"mae": v["pairs"]["mae"], "shortest": v["decisions"]["shortest_action_rate"]}
                              for k, v in result["frozen_q"].items()},
                      "deep": {k: {name: value for name, value in v.items() if name in (
                          "relative_states", "decision_relative_coverage", "fully_supervised_sampled_paths")}
                               for k, v in by_depth.items() if int(k) >= 7}}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    audit(args.run, args.out)
