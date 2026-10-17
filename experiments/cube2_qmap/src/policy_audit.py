"""Frozen policy interventions: symmetry averaging and localized oracle repairs.

No training or optimizer update. Averaged readouts and oracle repairs are
diagnostic controls, never presented as the original 128-dimensional Q policy.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .model import QMap
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


def two_cycle_basins(next_state):
    """Classify eventual outcomes exactly, independent of the rollout budget.

    -1 means solved, -2 means a different cycle, otherwise the smaller ID of
    the two-cycle. Pointer doubling crosses every possible transient tail.
    """
    pointer = np.asarray(next_state)
    endpoint = pointer.copy()
    for _ in range((len(pointer) - 1).bit_length()):
        endpoint = endpoint[endpoint]
    two = (pointer[pointer[endpoint]] == endpoint) & (pointer[endpoint] != endpoint)
    labels = np.full(len(pointer), -2, dtype=np.int32)
    labels[two] = np.minimum(endpoint[two], pointer[endpoint[two]])
    labels[endpoint == 0] = -1
    return labels


def cycle_diagnostics(env, height, next_state, oracle, examples=8):
    """Locate traps and their true exits; oracle edits are diagnostic only."""
    labels = two_cycle_basins(next_state)
    representatives, counts = np.unique(labels[labels >= 0], return_counts=True)
    partners = next_state[representatives]
    low = np.where(env.depth[representatives] < env.depth[partners], representatives, partners)
    high = np.where(low == representatives, partners, representatives)
    if not np.all(env.depth[high].astype(int) == env.depth[low].astype(int) + 1):
        raise ValueError("Expected quarter-turn two-cycles to join adjacent exact depths")
    children = env.successors(low)
    good = env.depth[children].astype(int) == env.depth[low, None].astype(int) - 1
    best_exit = children[np.arange(len(low)), np.where(good, height[children], np.inf).argmin(-1)]
    gap = height[best_exit] - height[high]
    repaired = next_state.copy()
    repaired[low] = oracle[low]
    repaired_labels = two_cycle_basins(repaired)
    rows = []
    for index in np.argsort(-counts)[:examples]:
        a, b, exit_state = map(int, (low[index], high[index], best_exit[index]))
        rows.append({"cycle": [a, b], "true_goal_distances": [int(env.depth[a]), int(env.depth[b])],
                     "predicted_goal_distances": [float(height[a]), float(height[b])],
                     "basin_states": int(counts[index]), "exit": exit_state,
                     "exit_true_goal_distance": int(env.depth[exit_state]),
                     "exit_predicted_goal_distance": float(height[exit_state]),
                     "exit_disadvantage": float(gap[index]),
                     "both_complete_candidate_sets": [
                         {"parent": parent, "children": env.successors(np.array([parent]))[0].tolist(),
                          "true_distances": env.depth[env.successors(np.array([parent]))[0]].tolist(),
                          "predicted_distances": height[env.successors(np.array([parent]))[0]].tolist()}
                         for parent in (a, b)]})
    reached = reach_within_budget(next_state, 0, 28) >= 0
    return {"two_cycles": len(low), "two_cycle_basin_states": int((labels >= 0).sum()),
            "other_cycle_basin_states": int((labels == -2).sum()),
            "eventually_solved_states": int((labels == -1).sum()),
            "solvable_but_exceed_28_steps": int(((labels == -1) & ~reached).sum()),
            "exit_disadvantage_quantiles": np.quantile(gap, [0, .5, .9, 1]).tolist() if len(gap) else [],
            "two_cycle_basin_states_by_lower_depth": np.bincount(
                env.depth[low], weights=counts, minlength=env.diameter + 1).astype(int).tolist(),
            "oracle_edit_only_current_cycle_exits_diagnostic": {
                "edited_states": len(low), "eventually_solved_states": int((repaired_labels == -1).sum()),
                "still_in_two_cycle_basins": int((repaired_labels >= 0).sum()),
                "other_cycle_basin_states": int((repaired_labels == -2).sum())},
            "largest_traps": rows}


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
    return {"checkpoint": str(path), "epoch": saved.get("epoch"), "steps": saved.get("steps"),
            "source_commit": saved["source_commit"],
            "layers": layers,
            "axis_mean_distance_range_all_states": sum(frame_ranges) / env.size,
            "axis_sampled_states": len(sampled),
            "axis_states_correctness_changes": int((frame_good.any(-1) & ~frame_good.all(-1)).sum()),
            "common_left_mean_absolute_distance_change": float(np.abs(translated - height[source, None]).mean()),
            "baseline": summarize_policy(env, baseline),
            "cycle_diagnostics": cycle_diagnostics(env, height, baseline, oracle),
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
