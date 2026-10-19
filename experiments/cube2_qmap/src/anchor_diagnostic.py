"""Oracle-assisted diagnostic of 128 fixed anchor coordinates, never a learned Q."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from grounded_llm.tasks.cube2.evaluate import evaluate
from .run import make_environment
from grounded_llm.tasks.cube2.environment import POWERS, SLOTS, permutation_rank


class OracleAnchors(torch.nn.Module):
    def __init__(self, env, anchors):
        super().__init__()
        self.env = env
        self.anchors = np.asarray(anchors, dtype=np.int64)
        self.cache = {}

    def encode(self, features):
        joint = features.detach().cpu().numpy().reshape(-1, 8, 24).argmax(-1)
        cp, co = joint // 3, joint % 3
        ids = permutation_rank(cp[:, SLOTS]) * 729 + co[:, SLOTS[:6]] @ POWERS
        missing = np.array([int(s) for s in np.unique(ids) if int(s) not in self.cache], dtype=np.int64)
        for begin in range(0, len(missing), 256):
            chunk = missing[begin:begin + 256]
            values = self.env.distance(chunk[:, None], self.anchors[None, :]).astype(np.float32)
            self.cache.update(zip(chunk.tolist(), values))
        return torch.as_tensor(np.stack([self.cache[int(s)] for s in ids]), device=features.device)

    @staticmethod
    def distance(source, goal):
        return (source - goal).abs().amax(-1)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--prepared-run", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    args = parser.parse_args()
    contract = json.loads(args.contract.read_text())
    torch.set_num_threads(2)
    root = args.prepared_run
    config = json.loads((root / "config.json").read_text())
    data = dict(np.load(root / "data.npz", allow_pickle=False))
    env = make_environment(config["task"], root / "oracle_depth.npy")
    available = np.setdiff1d(np.arange(env.size), data["protected_states"])
    anchors = np.random.default_rng(contract["seed"]).choice(available, contract["dimension"], replace=False)
    model = OracleAnchors(env, anchors)
    # Check that feature decoding and signed subtraction preserve the exact
    # anchor construction, including states with large IDs and anchor self-zero.
    probe = np.r_[anchors[:4], env.size - 1]
    q = model.encode(torch.as_tensor(env.features(probe))).numpy()
    np.testing.assert_array_equal(q, env.distance(probe[:, None], anchors[None, :]))
    assert not np.isin(anchors, data["protected_states"]).any()
    result = {"condition": "oracle_assisted_fixed_anchor_linf", "learned": False,
              "uses_oracle_at_inference": True, "source_commit": args.source_commit,
              "prepared_source_commit": json.loads((root / "summary.json").read_text())["source_commit"],
              "contract": contract, "anchors": anchors.tolist(),
              "metrics": evaluate(model, env, data, config, "cpu")}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as handle:
        handle.write(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"output": str(args.out), "test_rollout": result["metrics"]["test"]["rollout"]["q"]}))


if __name__ == "__main__":
    main()
