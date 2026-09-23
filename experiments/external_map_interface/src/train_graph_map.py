"""Reuse Step 1 local Q/V training on the graph in a frozen evaluation suite."""

import argparse
import json
from pathlib import Path
import subprocess
import time

import numpy as np

from experiments.cml_map_scaling.src.core import (
    action_catalog, geometry, sample_walks, shortest_distances, train_epoch)
from .transitions import environment_from_suite

ROOT = Path(__file__).resolve().parents[3]


def candidate_ranking(q, v, actions, outgoing, truth):
    correct = total = 0
    by_distance = {}
    for current in range(len(q)):
        ids = outgoing[current]
        predicted = q[current].astype(np.float64) + v[ids].astype(np.float64)
        scores = np.linalg.norm(predicted[:, None, :] - q[None, :, :], axis=-1)
        for goal in range(len(q)):
            if current == goal:
                continue
            following = int(actions[ids[int(np.argmin(scores[:, goal]))], 1])
            good = bool(truth[following, goal] == truth[current, goal] - 1)
            bucket = by_distance.setdefault(int(truth[current, goal]), {"pairs": 0, "correct": 0})
            bucket["pairs"] += 1
            bucket["correct"] += int(good)
            correct += int(good)
            total += 1
    return {"pairs": total, "best_distance_action_on_shortest_path": correct,
            "accuracy": correct / total, "by_true_distance": by_distance}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    suite = json.loads(args.suite.read_text(encoding="utf-8"))
    base = json.loads((ROOT / config["map"]["step1_config"]).read_text(encoding="utf-8"))
    base["model"]["state_dim"] = config["map"]["state_dim"]
    env = environment_from_suite(suite)
    if len(env.adjacency) != config["node_count"]:
        raise ValueError("Suite node count does not match config")
    actions, outgoing = action_catalog(env.adjacency)
    train, model = base["training"], base["model"]
    seed, offsets = config["map"]["seed"], train["seed_offsets"]
    count = train["walks_at_32_nodes"] * len(env.adjacency) // 32
    walks = sample_walks(actions, outgoing, count, train["transitions_per_walk"], seed + offsets["walks"])
    visits = np.bincount(walks[:, :, 1].ravel(), minlength=len(actions))
    rng = np.random.default_rng(seed + offsets["initialization"])
    q = rng.normal(0, model["q_init_std"], (len(env.adjacency), model["state_dim"])).astype(model["dtype"])
    v = rng.normal(0, model["v_init_std"], (len(actions), model["state_dim"])).astype(model["dtype"])
    args.out.mkdir(parents=True)
    np.savez_compressed(args.out / "inputs.npz", adjacency=env.adjacency, actions=actions,
                        walks=walks, visits=visits, q_initial=q, v_initial=v)
    # Truth is used only below for diagnostics, never by train_epoch.
    truth = shortest_distances(env.adjacency)
    initial, _ = geometry(q, v, actions, truth, visits)
    replay = np.random.default_rng(seed + offsets["replay"])
    started = time.monotonic()
    for epoch in range(1, train["epochs"] + 1):
        train_epoch(q, v, walks, replay.permutation(count), train["eta_q"], train["eta_v"])
    elapsed = time.monotonic() - started
    final, _ = geometry(q, v, actions, truth, visits)
    np.savez_compressed(args.out / "map.npz", q=q, v=v)
    summary = {"source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
               "suite": str(args.suite.resolve()), "map_config": config["map"],
               "step1_config": base, "training_seconds": elapsed,
               "action_coverage": float(np.mean(visits > 0)), "initial": initial, "final": final,
               "candidate_ranking": candidate_ranking(q, v, actions, outgoing, truth)}
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"complete": str(args.out), "seconds": elapsed,
                      "spearman": final["spearman"], "mse": final["transition_mse_all_actions"],
                      "candidate_accuracy": summary["candidate_ranking"]["accuracy"]}), flush=True)


if __name__ == "__main__":
    main()
