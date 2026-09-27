"""Read-only greedy rollout with exact legal successors and learned Q scores."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .model import BoardEncoder, QMap, board_bits, distance
from .oracle import successors


def rollout(data_path, checkpoint, seed):
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
    states = [int(s) for s in data["states"]]
    index = {state: i for i, state in enumerate(states)}
    encoded_cache = {}

    def q_values(masks):
        if encoded:
            missing = list(dict.fromkeys(mask for mask in masks if mask not in encoded_cache))
            for start in range(0, len(missing), 2048):
                chunk = missing[start:start + 2048]
                values = model.encode(board_bits(chunk))
                encoded_cache.update(zip(chunk, values))
            return torch.stack([encoded_cache[mask] for mask in masks])
        return model.q.weight[[index[mask] for mask in masks]]

    original = states[int(data["initial_id"])]
    starts = [int(data["initial_id"])] + sorted({int(x) for x in data["decisions"][:, 0]}
                                             - {int(data["initial_id"])})
    max_steps = original.bit_count() // 2 + 1
    records = []
    with torch.no_grad():
        for start_id in starts:
            rng = np.random.default_rng(seed + start_id)
            state, actions = states[start_id], []
            result = "step_limit"
            for _ in range(max_steps):
                if state == 0:
                    result = "solved"
                    break
                legal = successors(state)
                if not legal:
                    result = "dead_end"
                    break
                missing = [(action, target) for action, target in legal if target not in index]
                if missing and not encoded:
                    result = "candidate_q_missing"
                    break
                q = q_values([target for _, target in legal])
                goal_q = q_values([0])[0]
                scores = distance(q, goal_q, saved["metric"]).numpy()
                tied = np.flatnonzero(np.isclose(scores, scores.min(), rtol=0, atol=1e-7))
                choice = int(rng.choice(tied))
                action, state = legal[choice]
                actions.append(int(action))
            if state == 0:
                result = "solved"
            records.append({"start_state_id": start_id, "start_cells": states[start_id].bit_count(),
                            "exact_start_distance": int(data["goal_labels"][start_id]),
                            "outcome": result, "executed_steps": len(actions), "actions": actions})
    counts = {key: sum(row["outcome"] == key for row in records)
              for key in ("solved", "dead_end", "candidate_q_missing", "step_limit")}
    solved = [row for row in records if row["outcome"] == "solved"]
    return {"metric": saved["metric"], "model_type": saved.get("model_type", "table"),
            "seed": seed, "source_commit": saved["source_commit"],
            "starts": len(records), "outcomes": counts, "solve_rate_all_starts": counts["solved"] / len(records),
            "encoded_states_outside_training_pool": sum(mask not in index for mask in encoded_cache),
            "mean_extra_steps_on_solved": float(np.mean([row["executed_steps"] - row["exact_start_distance"]
                                                        for row in solved])) if solved else None,
            "initial_board": records[0], "records": records}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260927)
    args = parser.parse_args()
    result = rollout(args.data, args.checkpoint, args.seed)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "records"}), flush=True)


if __name__ == "__main__":
    main()
