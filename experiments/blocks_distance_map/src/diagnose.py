"""Read-only diagnosis of label coverage and trained versus held-out goal pairs."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .evaluate import auc, predict_pairs
from .model import BoardEncoder, QMap, board_bits
from .train_encoder import encoded_view


def diagnose(data_path, checkpoint):
    torch.set_num_threads(4)
    data = dict(np.load(data_path, allow_pickle=False))
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    encoded = saved.get("model_type") == "board_mlp"
    if encoded:
        model = BoardEncoder(saved["config"]["model"]["state_dim"],
                             saved["config"]["model"]["encoder_hidden_dim"],
                             saved["cap"], saved["metric"])
    else:
        model = QMap(saved["state_count"], saved["config"]["model"]["state_dim"],
                     saved["cap"], saved["metric"], saved["config"]["model"]["init_std"])
    model.load_state_dict(saved["model"])
    model.eval()
    view = encoded_view(model, board_bits(data["states"])) if encoded else model
    pairs, goal = data["pairs"], int(data["goal_id"])
    scores = predict_pairs(view, np.column_stack((np.arange(len(data["states"])),
                                                 np.full(len(data["states"]), goal))))
    rows = pairs[data["split"] == 0]
    outgoing = np.bincount(rows[rows[:, 2] == 1, 0], minlength=len(scores))
    decision_states = np.unique(data["decisions"][:, 2])
    decision_states = decision_states[decision_states != goal]
    rollout_states = np.unique(data["transitions"][:, [0, 2]])
    result = {"metric": saved["metric"], "model_type": saved.get("model_type", "table"),
              "source_commit": saved["source_commit"],
              "decision_nonempty_successors": len(decision_states),
              "decision_successors_outside_original_rollouts": int(
                  (~np.isin(decision_states, rollout_states)).sum()),
              "decision_successors_without_trained_outgoing_one_step": int(
                  (outgoing[decision_states] == 0).sum()),
              "train_hard_negatives": int((rows[:, 3] == 2).sum()),
              "train_non_goal_hard_negatives": int(((rows[:, 3] == 2) & (rows[:, 1] != goal)).sum()),
              "decision_successors_with_trained_non_goal_hard_negative": int(np.isin(
                  decision_states, rows[(rows[:, 3] == 2) & (rows[:, 1] != goal), 0]).sum()),
              "goal_pairs": {}}
    for sid, name in enumerate(("train", "validation", "test")):
        ids = pairs[(data["split"] == sid) & (pairs[:, 1] == goal), 0]
        truth = data["goal_labels"][ids]
        finite = truth >= 0
        result["goal_pairs"][name] = {"count": len(ids), "finite": int(finite.sum()),
            "unreachable_auc": auc(~finite, scores[ids]),
            "finite_mae": float(np.abs(scores[ids][finite] - truth[finite]).mean()) if finite.any() else None,
            "unreachable_distance_mean": float(scores[ids][~finite].mean()) if (~finite).any() else None,
            "unreachable_distance_max": float(scores[ids][~finite].max()) if (~finite).any() else None}
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = diagnose(args.data, args.checkpoint)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
