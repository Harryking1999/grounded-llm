"""Read-only audit of relational supervision and the gradients it supplies."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from experiments.puzzle_qmap.src.model import MapReader, soft_rank_loss
from .environment import Cube2
from .structure_study import dense_upper_loss, load_contract, relation_values


def describe_gradient(gradient):
    return {"norm": float(gradient.norm()),
            "coordinates_with_gradient": int((gradient.abs().sum(tuple(range(gradient.ndim - 1))) > 1e-10).sum()),
            "state_coordinate_nonzero_fraction": float((gradient.abs() > 1e-10).float().mean())}


def cosine(a, b):
    denominator = a.norm() * b.norm()
    return float(torch.dot(a.ravel(), b.ravel()) / denominator) if denominator > 0 else None


def audit(model, features, data, temperature, rank_weight):
    model.eval()
    q = model.encode(features).reshape(1, len(data["states"]), -1)
    pair, scores, _, _ = relation_values(model, q, data, np.arange(len(data["pairs"])))
    pairs = torch.as_tensor(data["pairs"], device=q.device)
    a, b = pairs.T
    target = torch.as_tensor(data["distances"][data["pairs"][:, 0], data["pairs"][:, 1]], device=q.device).float()
    truth = torch.as_tensor(data["truth"], device=q.device)
    losses = {"pair": F.smooth_l1_loss(pair, target[None]),
              "rank": rank_weight * soft_rank_loss(scores[0], truth, torch.ones_like(truth, dtype=torch.bool), temperature),
              "dense_upper": dense_upper_loss(q, pairs, target)}
    for name, mask in (("near", target <= 4), ("far", target >= 5)):
        losses[name] = F.smooth_l1_loss(pair[:, mask], target[None, mask])
    gradients, result = {}, {}
    parameters = tuple(model.parameters())
    for name, loss in losses.items():
        values = torch.autograd.grad(loss, (q, *parameters), retain_graph=True)
        gradients[name] = (values[0], torch.cat([g.ravel() for g in values[1:]]))
        result[name] = {"loss": float(loss.detach()), "q_gradient": describe_gradient(values[0]),
                        "parameter_gradient_norm": float(gradients[name][1].norm())}
    result["cosines"] = {f"{first}_vs_{second}":
                         {"q": cosine(gradients[first][0], gradients[second][0]),
                          "parameters": cosine(gradients[first][1], gradients[second][1])}
                         for first, second in (("pair", "rank"), ("near", "far"), ("pair", "dense_upper"))}
    difference = (q[:, a] - q[:, b]).abs()
    violates = difference > target[None, :, None] + .01
    result["coordinate_bounds"] = {"overestimated_pairs": int(violates.any(-1).sum()),
                                    "violating_coordinate_constraints": int(violates.sum()),
                                    "mean_violations_per_overestimated_pair": float(violates.sum() / violates.any(-1).sum().clamp_min(1))}
    result["output_boundary_fraction"] = float(((q < .01 * model.cap) | (q > .99 * model.cap)).float().mean())
    return result


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--prepared-run", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    config = load_contract(args.contract)
    env = Cube2(args.prepared_run / "oracle_depth.npy", metric="QTM_URF_6")
    data = dict(np.load(args.study / "data.npz", allow_pickle=False))
    features = torch.as_tensor(env.features(data["poses_fit"])).float()
    checkpoint = args.study / "linf_joint_fixed" / "best.pt"
    model = MapReader(env, checkpoint).model
    result = {"source_commit": args.source_commit, "checkpoint": str(checkpoint),
              "scope": "Complete fixed finite map; gradients at one trained checkpoint, not a causal training comparison.",
              "states": len(data["states"]), "supervised_pairs": len(data["pairs"]),
              "minimum_pair_degree": int(np.bincount(data["pairs"].ravel(), minlength=len(data["states"])).min()),
              "audit": audit(model, features, data, config["training"]["temperature"], config["training"]["rank_weight"])}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as handle:
        handle.write(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
