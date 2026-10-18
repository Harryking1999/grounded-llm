"""Check a necessary directed-sum identity against unchanged finite labels."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from experiments.gcml_counterexamples.src import blocks
from .oracle import DistanceOracle
from .tree_supervision_eval import QScorer


SIGNS = np.asarray([1, -1, 1, -1, -1, 1])


def lower_bounds(cap):
    # |sum(sign_i error_i)| <= sum |error_i|. Jensen bounds equal-weight
    # smooth-L1 over these six scores, without claiming training samples are
    # equally weighted or that solving the greedy task is impossible.
    mean_absolute = cap / 6
    huber = .5 * mean_absolute ** 2 if mean_absolute < 1 else mean_absolute - .5
    return {"six_pair_absolute_error_sum_at_least": cap,
            "six_pair_equal_weight_mean_huber_at_least": huber}


def run(roots_file, checkpoint, out):
    torch.set_num_threads(2)
    scorer = QScorer(checkpoint, "cpu")
    if scorer.metric != "directed_sum":
        raise ValueError("This identity audit applies to the original directed_sum metric")
    oracle = DistanceOracle()
    rows = []
    for board, root in enumerate(json.loads(roots_file.read_text())):
        a = int(root["initial"])
        b = blocks.apply(a, root["construction"][0])
        c = blocks.apply(b, root["construction"][1])
        pairs = [(a, b), (b, a), (b, c), (c, b), (a, c), (c, a)]
        truth = np.asarray([oracle.distance(s, g) for s, g in pairs])
        if not np.array_equal(truth, [1, -1, 1, -1, 2, -1]):
            raise AssertionError("Two legal removals must form the certified short chain")
        scores = np.asarray([float(scorer.scores([s], g)[0]) for s, g in pairs])
        rows.append({"root": board, "states": [str(a), str(b), str(c)],
                     "truth": truth.tolist(), "scores": scores.tolist(),
                     "predicted_identity_residual": float(scores @ SIGNS)})
    scores = np.asarray([r["scores"] for r in rows])
    targets = np.asarray([1, -1, 1, -1, 2, -1])
    result = {"checkpoint": str(checkpoint), "identity": scorer.identity, "metric": scorer.metric,
              "certified_training_root_chains": len(rows),
              "identity": "D(A,B)-D(B,A)+D(B,C)-D(C,B)-D(A,C)+D(C,A)=0",
              "derivation": "positive(x)-positive(-x)=x; D(s,g)-D(g,s) is a scalar potential difference for directed_sum, in every dimension and at every coordinate scale",
              "maximum_absolute_predicted_residual": max(abs(r["predicted_identity_residual"]) for r in rows),
              "caps": {}, "examples": rows[:5],
              "scope": "Analytic incompatibility of all six exact regression targets, present for every certified two-action chain. Not proof of a greedy success-rate ceiling; no model, labels, losses or execution changed by this read-only audit."}
    for cap in (13, 19):
        label = np.where(targets < 0, cap, targets)
        error = np.abs(scores - label)
        result["caps"][str(cap)] = {"target_identity_residual": int(label @ SIGNS), **lower_bounds(cap),
            "model_six_pair_mean_absolute_error": float(error.mean()),
            "model_forward_mean_absolute_error": float(error[:, [0, 2, 4]].mean()),
            "model_reverse_mean_absolute_error": float(error[:, [1, 3, 5]].mean())}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--roots", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    run(args.roots, args.checkpoint, args.out)
