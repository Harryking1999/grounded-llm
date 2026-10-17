"""Unrestricted greedy policy evaluation after the fixed overlap experiment."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .evaluate import rollout
from .model import MapReader
from .environment import Cube2


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--study", type=Path, required=True, help="Completed single-seed run")
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--source-commit", required=True, help="Evaluation source; training identity is read separately")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if not (args.study / "complete.json").exists():
        raise ValueError("Finish the fixed training comparison before policy evaluation")
    torch.set_num_threads(2)
    protocol = json.loads(args.contract.read_text())
    training = json.loads((args.study / "study_summary.json").read_text())
    env = Cube2(args.inputs / "oracle_depth.npy", metric="QTM_URF_6")
    base = dict(np.load(args.inputs / "base_data.npz", allow_pickle=False))
    data = {kind: dict(np.load(args.study / f"data_{kind}.npz", allow_pickle=False)) for kind in ("overlap", "disjoint")}
    cohorts = {}
    for kind, d in data.items():
        cohorts[kind] = {"old_training": d["rows"][0][base["initial"]],
                         "all_training": d["rows"][:, base["initial"]].reshape(-1, 2),
                         "novel_cross_goals": d["states"][d["tasks"]],
                         **{s: d[f"poses_{s}"][:, base["initial"]].reshape(-1, 2) for s in ("validation", "test")}}

    def evaluate(checkpoint, kind):
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if saved["source_commit"] != training["source_commit"]:
            raise ValueError("Checkpoint does not belong to this training study")
        model = MapReader(env, checkpoint, args.device).model
        # The harness independently computes Q, hamming and oracle policies;
        # only the Q result is retained. Oracle never selects a Q action.
        return {name: rollout(model, env, cohorts[kind][name], args.device, 2,
                              horizon=protocol["horizon"])["q"] for name in protocol["cohorts"]}

    result = {"evaluation_source_commit": args.source_commit, "training_source_commit": training["source_commit"],
              "seed": training["seed"], "protocol": protocol,
              "warmup": {kind: evaluate(args.study / "warmup/best.pt", kind) for kind in data}, "conditions": {}}
    for name, row in training["conditions"].items():
        kind = row["condition"]["data_kind"]
        result["conditions"][name] = {checkpoint: evaluate(args.study / name / f"{checkpoint}.pt", kind)
                                     for checkpoint in protocol["checkpoints"]}
    with (args.study / "unrestricted_policy.json").open("x") as handle:
        handle.write(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"seed": training["seed"], "last": {name: {k: v["solved"] for k, v in values["last"].items()}
                     for name, values in result["conditions"].items()}}))


if __name__ == "__main__":
    main()
