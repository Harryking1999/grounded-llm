"""Frozen policy interventions: symmetry averaging and localized oracle repairs.

No training or optimizer update. Averaged readouts and oracle repairs are
diagnostic controls, never presented as the original 128-dimensional Q policy.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from experiments.puzzle_qmap.src.model import QMap
from .environment import Cube2
from .fixed_goal import reach_within_budget


def policy(env, height):
    next_state = np.empty(env.size, dtype=np.int32)
    oracle = np.empty_like(next_state)
    optimal = np.zeros(env.size, dtype=bool)
    margins = np.full(env.size, np.nan, dtype=np.float32)
    for start in range(0, env.size, 16384):
        ids = np.arange(start, min(env.size, start + 16384))
        children = env.successors(ids)
        scores = height[children]
        chosen = scores.argmin(-1)
        next_state[ids] = children[np.arange(len(ids)), chosen]
        good = env.depth[children].astype(int) == env.depth[ids, None].astype(int) - 1
        optimal[ids] = good[np.arange(len(ids)), chosen]
        oracle[ids] = children[np.arange(len(ids)), good.argmax(-1)]
        competing = good.any(-1) & (~good).any(-1)
        good_min = np.where(good, scores, np.inf).min(-1)
        bad_min = np.where(~good, scores, np.inf).min(-1)
        margins[ids[competing]] = (bad_min - good_min)[competing]
    next_state[0] = oracle[0] = 0
    return next_state, oracle, optimal, margins


def summarize_policy(env, next_state):
    reached = reach_within_budget(next_state, 0, 28) >= 0
    ids = np.arange(env.size)
    pairs = (next_state[next_state] == ids) & (next_state != ids) & (ids < next_state)
    low = np.minimum(env.depth[pairs], env.depth[next_state[pairs]])
    return {"count": env.size, "solved": int(reached.sum()), "success_rate": float(reached.mean()),
            "two_cycles": int(pairs.sum()),
            "two_cycle_lower_depth": np.bincount(low, minlength=env.diameter + 1).tolist(),
            "by_depth": {str(d): {"count": int((env.depth == d).sum()),
                                   "success_rate": float(reached[env.depth == d].mean())}
                         for d in range(env.diameter + 1)}}


@torch.no_grad()
def audit_checkpoint(env, path, device, per_layer=256):
    saved = torch.load(path, map_location="cpu", weights_only=False)
    if saved["model_config"].get("metric") != "linf":
        raise ValueError("This active-coordinate audit requires the recorded Linf model")
    model = QMap(env.feature_dim, **saved["model_config"], cap=saved["cap"]).to(device)
    model.load_state_dict(saved["model"])
    model.eval()
    goal = model.encode(torch.as_tensor(env.features([0]), device=device))
    height = np.empty(env.size, dtype=np.float32)
    winner = np.empty(env.size, dtype=np.int16)
    for start in range(0, env.size, 8192):
        ids = np.arange(start, min(env.size, start + 8192))
        q = model.encode(torch.as_tensor(env.features(ids), device=device))
        values, indices = (q - goal).abs().max(-1)
        height[ids], winner[ids] = values.cpu().numpy(), indices.cpu().numpy()
    baseline, oracle, optimal, margins = policy(env, height)
    averaged = np.empty_like(height)
    frame_ranges = []
    for start in range(0, env.size, 8192):
        ids = np.arange(start, min(env.size, start + 8192))
        frames = env.axis_symmetry(ids[:, None], np.arange(6)[None])
        if not np.array_equal(env.depth[frames], np.broadcast_to(env.depth[ids, None], frames.shape)):
            raise ValueError("Axis transformation changed exact depth")
        scores = height[frames]
        averaged[ids] = scores.mean(-1)
        frame_ranges.append(float(np.ptp(scores, axis=-1).sum()))
    symmetry_policy, _, symmetry_optimal, _ = policy(env, averaged)
    far_repaired = np.where(env.depth > 7, oracle, baseline)
    near_repaired = np.where(env.depth <= 7, oracle, baseline)
    rng = np.random.default_rng(20261030)
    sampled = np.concatenate([rng.choice(np.flatnonzero(env.depth == d),
                                         min(per_layer, int((env.depth == d).sum())), replace=False)
                              for d in range(1, 15)])
    frames = env.axis_symmetry(sampled[:, None], np.arange(6)[None])
    frame_good = env.depth[baseline[frames]].astype(int) == env.depth[frames].astype(int) - 1
    layers = {}
    for d in range(1, 15):
        mask = env.depth == d
        coordinate_counts = np.bincount(winner[mask], minlength=128)
        gap = margins[mask];gap = gap[np.isfinite(gap)]
        layers[str(d)] = {"count": int(mask.sum()), "height_mean": float(height[mask].mean()),
                           "mae": float(np.abs(height[mask] - d).mean()),
                           "optimal_action_rate": float(optimal[mask].mean()),
                           "symmetry_averaged_optimal_action_rate": float(symmetry_optimal[mask].mean()),
                           "active_coordinates": int((coordinate_counts > 0).sum()),
                           "dominant_coordinate": int(coordinate_counts.argmax()),
                           "dominant_coordinate_share": float(coordinate_counts.max() / mask.sum()),
                           "mean_good_bad_margin": float(gap.mean()) if len(gap) else None}
    # The same geometric relation in common-left translated presentations.
    source = sampled[:]
    poses = rng.integers(env.size, size=(len(source), 4))
    moved = env.compose(poses, source[:, None])
    def encode(ids):
        flat = ids.ravel()
        return torch.cat([model.encode(torch.as_tensor(env.features(chunk), device=device))
                          for chunk in np.array_split(flat, max(1, (len(flat) + 8191) // 8192))]).reshape(*ids.shape, -1)
    translated = model.distance(encode(moved), encode(poses)).cpu().numpy()
    if not np.array_equal(env.distance(moved, poses), np.broadcast_to(env.depth[source, None], moved.shape)):
        raise ValueError("Common-left multiplication changed exact distance")
    return {"checkpoint": str(path), "epoch": saved["epoch"], "source_commit": saved["source_commit"],
            "layers": layers,
            "axis_mean_distance_range_all_states": sum(frame_ranges) / env.size,
            "axis_sampled_states": len(sampled),
            "axis_states_correctness_changes": int((frame_good.any(-1) & ~frame_good.all(-1)).sum()),
            "common_left_mean_absolute_distance_change": float(np.abs(translated - height[source, None]).mean()),
            "baseline": summarize_policy(env, baseline),
            "symmetry_averaged_diagnostic": summarize_policy(env, symmetry_policy),
            "oracle_far_only_diagnostic": summarize_policy(env, far_repaired),
            "oracle_near_only_diagnostic": summarize_policy(env, near_repaired)}


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--oracle", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    torch.set_num_threads(2)
    env = Cube2(args.oracle, metric="QTM_URF_6")
    result = {"analysis_source_commit": args.source_commit, "scope": "frozen diagnostics; no optimization", "models": []}
    for path in args.checkpoint:
        model = audit_checkpoint(env, path, args.device)
        result["models"].append(model)
        args.out.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps({"checkpoint": str(path), "baseline": model["baseline"]["success_rate"],
                          "symmetry_averaged_diagnostic": model["symmetry_averaged_diagnostic"]["success_rate"],
                          "oracle_far_only_diagnostic": model["oracle_far_only_diagnostic"]["success_rate"],
                          "oracle_near_only_diagnostic": model["oracle_near_only_diagnostic"]["success_rate"]}), flush=True)


if __name__ == "__main__":
    main()
