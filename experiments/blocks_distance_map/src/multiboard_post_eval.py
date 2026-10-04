"""Frozen-checkpoint comparison of isolated and supported unseen goals."""
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
import torch

from .model import BoardEncoder, board_bits
from .multiboard_eval import (encoded_values, goal_rollouts, listwise_groups,
                              pair_scores, split_metrics)


def matched_supported_tasks(supported, isolated_tasks, seed):
    rng = np.random.default_rng(seed)
    chosen = []
    for steps, count in sorted(Counter(isolated_tasks[:, 2].tolist()).items()):
        pool = supported[supported[:, 2] == steps]
        if not len(pool):
            continue
        indices = rng.choice(len(pool), min(count, len(pool)), replace=False)
        chosen.extend(pool[indices, :3].tolist())
    return np.asarray(chosen, dtype=np.int32).reshape(-1, 3)


def evaluate(run_dir, analysis_commit, device):
    run_dir = Path(run_dir)
    torch.set_num_threads(4)
    data = dict(np.load(run_dir / "data" / "data.npz", allow_pickle=False))
    saved = torch.load(run_dir / "best.pt", map_location=device, weights_only=True)
    config = saved["config"]
    model = BoardEncoder(config["model"]["state_dim"],
                         config["model"]["encoder_hidden_dim"], saved["cap"],
                         saved["metric"]).to(device)
    model.load_state_dict(saved["model"])
    model.eval()
    isolated = data["test_isolated_goal"]
    isolated_targets = set(isolated[:, 1].tolist())
    unseen = data["test_unseen_goal"]
    supported = unseen[~np.isin(unseen[:, 1], list(isolated_targets))]
    bits = board_bits(data["states"]).to(device)
    values = encoded_values(model, bits)
    groups = listwise_groups(supported, config["training"]["seed"] + 1907,
                             per_anchor=config["training"]["listwise_groups_per_anchor"])
    scores = pair_scores(values, supported, saved["metric"])
    tasks = matched_supported_tasks(supported, data["isolated_goal_tasks"],
                                    config["training"]["seed"] + 1907)
    result = {"model_source_commit": saved["source_commit"],
              "analysis_commit": analysis_commit, "best_step": saved["step"],
              "supported_goal_pairs": split_metrics(supported, scores, groups, int(data["cap"])),
              "isolated_task_distance_counts": dict(Counter(
                  data["isolated_goal_tasks"][:, 2].tolist())),
              "supported_task_distance_counts": dict(Counter(tasks[:, 2].tolist())),
              "supported_goal_rollout": goal_rollouts(model, tasks[:, :2], data["states"],
                                                      saved["metric"], device),
              "isolated_goal_rollout_replay": goal_rollouts(model,
                  data["isolated_goal_tasks"][:, :2], data["states"], saved["metric"], device)}
    (run_dir / "post_eval.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--analysis-commit", required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    evaluate(args.run, args.analysis_commit, args.device)


if __name__ == "__main__":
    main()
