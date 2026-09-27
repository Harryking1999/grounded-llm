"""Inspect where frozen-Q greedy paths to held-out nonempty goals fail."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

import numpy as np
import torch

from .model import BoardEncoder, board_bits, distance
from .multiboard_post_eval import matched_supported_tasks
from .oracle import DistanceOracle, successors


@torch.no_grad()
def diagnose(model, tasks, states, metric, device, oracle, preserve_goal_cells=False):
    cache = {}

    def encode(masks):
        missing = [mask for mask in dict.fromkeys(masks) if mask not in cache]
        for start in range(0, len(missing), 2048):
            chunk = missing[start:start + 2048]
            cache.update(zip(chunk, model.encode(board_bits(chunk).to(device)).split(1)))
        return torch.cat([cache[mask] for mask in masks], dim=0)

    outcomes = Counter()
    decisions = Counter()
    first_error_step = Counter()
    by_distance = defaultdict(Counter)
    for source_id, goal_id, true_distance in tasks:
        state, goal = int(states[source_id]), int(states[goal_id])
        true_distance = int(true_distance)
        assert oracle.distance(state, goal) == true_distance
        first_error = None
        outcome = "step_limit"
        for step in range(12):
            if state == goal:
                outcome = "reached"
                break
            choices = successors(state)
            if preserve_goal_cells:
                choices = [(action, mask) for action, mask in choices if mask & goal == goal]
            if not choices:
                outcome = "no_action"
                break
            masks = [mask for _, mask in choices]
            scores = distance(encode(masks), encode([goal]).expand(len(masks), -1), metric)
            chosen = masks[int(scores.argmin())]
            before = oracle.distance(state, goal)
            after = oracle.distance(chosen, goal)
            if after == before - 1:
                decisions["optimal"] += 1
            else:
                if first_error is None:
                    first_error = step + 1
                if after < 0:
                    kind = "removed_goal_cell" if goal & chosen != goal else "uncoverable_difference"
                    decisions[kind] += 1
                    outcome = kind
                    break
                decisions["reachable_detour"] += 1
            state = chosen
        else:
            if state == goal:
                outcome = "reached"
        outcomes[outcome] += 1
        by_distance[true_distance]["tasks"] += 1
        by_distance[true_distance][outcome] += 1
        if first_error is not None:
            first_error_step[first_error] += 1
    return {"tasks": len(tasks), "outcomes": dict(outcomes),
            "decisions": dict(decisions), "first_error_step": dict(first_error_step),
            "by_true_distance": {str(k): dict(v) for k, v in sorted(by_distance.items())}}


def evaluate(run_dir, device):
    run_dir = Path(run_dir)
    data = dict(np.load(run_dir / "data" / "data.npz", allow_pickle=False))
    saved = torch.load(run_dir / "best.pt", map_location=device, weights_only=True)
    config = saved["config"]
    model = BoardEncoder(config["model"]["state_dim"],
                         config["model"]["encoder_hidden_dim"], saved["cap"],
                         saved["metric"]).to(device)
    model.load_state_dict(saved["model"])
    model.eval()
    isolated = data["isolated_goal_tasks"]
    isolated_targets = set(data["test_isolated_goal"][:, 1].tolist())
    unseen = data["test_unseen_goal"]
    supported = unseen[~np.isin(unseen[:, 1], list(isolated_targets))]
    supported_tasks = matched_supported_tasks(supported, isolated,
                                               config["training"]["seed"] + 1907)
    known_targets = set(data["train"][:, 1].tolist())
    seen_pairs = data["test_seen_pair"]
    seen_pairs = seen_pairs[np.isin(seen_pairs[:, 1], list(known_targets)) &
                            (seen_pairs[:, 2] >= 2) & (data["states"][seen_pairs[:, 1]] != "0")]
    known_goal_tasks = matched_supported_tasks(seen_pairs, isolated,
                                                config["training"]["seed"] + 1973)
    if len(supported_tasks) != len(isolated) or len(known_goal_tasks) != len(isolated):
        raise ValueError("Distance-matched goal pools are incomplete")
    oracle = DistanceOracle(cache_limit=1_000_000, seconds=3600)
    result = {"source_commit": saved["source_commit"], "best_step": saved["step"],
              "isolated": diagnose(model, isolated, data["states"], saved["metric"], device, oracle),
              "supported": diagnose(model, supported_tasks, data["states"],
                                    saved["metric"], device, oracle),
              "known_goal_unseen_pair": diagnose(model, known_goal_tasks, data["states"],
                                                  saved["metric"], device, oracle),
              "preserve_goal_cells": {
                  "isolated": diagnose(model, isolated, data["states"], saved["metric"],
                                       device, oracle, preserve_goal_cells=True),
                  "supported": diagnose(model, supported_tasks, data["states"], saved["metric"],
                                        device, oracle, preserve_goal_cells=True),
                  "known_goal_unseen_pair": diagnose(model, known_goal_tasks, data["states"],
                                                      saved["metric"], device, oracle,
                                                      preserve_goal_cells=True)}}
    (run_dir / "trajectory_diagnosis.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    evaluate(args.run, args.device)


if __name__ == "__main__":
    main()
