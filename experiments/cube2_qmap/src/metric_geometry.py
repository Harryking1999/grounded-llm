"""Frozen validation diagnostics: coordinate collapse and common-left invariance."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .evaluate import encode
from .model import MapReader
from .run import make_environment


def audit(run, device="cpu"):
    config = json.loads((run / "config.json").read_text())
    env = make_environment(config["task"], run / "oracle_depth.npy")
    data = np.load(run / "data.npz", allow_pickle=False)
    model = MapReader(env, run / "best.pt", device).model
    pairs = data["pairs_validation"]
    ids = np.unique(pairs[:, :2])
    q = encode(model, env, ids, device).cpu().numpy()
    centered = q - q.mean(0)
    eigenvalues = np.linalg.eigvalsh(centered.T @ centered / len(q)).clip(0)
    total = eigenvalues.sum()
    effective_rank = float(total**2 / np.square(eigenvalues).sum()) if total else 0.
    a, b = np.searchsorted(ids, pairs[:, 0]), np.searchsorted(ids, pairs[:, 1])
    diff = np.abs(q[a] - q[b])
    winners = np.bincount(diff.argmax(-1), minlength=q.shape[1])
    histogram = np.sort(winners)[::-1]
    # Same labels under a fixed common-left transformation. This measures
    # the symmetry the data already teaches, without changing model selection.
    transformed = env.compose(np.random.default_rng(20261016).integers(env.size, size=len(pairs))[:, None],
                              pairs[:, :2])
    before = model.distance(torch.as_tensor(q[a], device=device), torch.as_tensor(q[b], device=device))
    after = model.distance(encode(model, env, transformed[:, 0], device),
                           encode(model, env, transformed[:, 1], device))
    gap = (before - after).abs().cpu().numpy()
    return {"metric": model.metric, "validation_only": True,
            "source_commit": torch.load(run / "best.pt", map_location="cpu", weights_only=False)["source_commit"],
            "state_count": len(ids), "coordinate_std_mean": float(q.std(0).mean()),
            "covariance_effective_rank": effective_rank,
            "largest_eigenvalue_fraction": float(eigenvalues[-1] / total) if total else 0.,
            "fraction_near_sigmoid_bounds": float(((q < .01 * env.diameter) |
                                                    (q > .99 * env.diameter)).mean()),
            "max_difference_winner_top1_fraction": float(histogram[0] / len(pairs)),
            "max_difference_winner_top8_fraction": float(histogram[:8].sum() / len(pairs)),
            "max_difference_winner_used_coordinates": int((winners > 0).sum()),
            "common_left_distance_gap_mean": float(gap.mean()),
            "common_left_distance_gap_max": float(gap.max())}


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    complete = json.loads((args.study / "complete.json").read_text())
    result = {name: audit(args.study / name) for name in complete["conditions"]}
    with args.out.open("x") as handle:
        handle.write(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
