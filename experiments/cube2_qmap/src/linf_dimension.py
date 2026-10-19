"""Exact finite witness for a lower bound on Linf embedding dimension."""
import argparse
import json
from pathlib import Path

import numpy as np

from grounded_llm.tasks.cube2.environment import Cube2


def construct(env, options, protected=()):
    rng = np.random.default_rng(options["seed"])
    antipodes = np.flatnonzero(env.depth == env.diameter)
    unavailable = set(map(int, protected))
    pairs = []
    for attempt in range(options["maximum_attempts"]):
        a = int(rng.integers(env.size))
        b = int(env.compose(a, int(rng.choice(antipodes))))
        if a in unavailable or b in unavailable:
            continue
        pair = np.array([a, b])
        if pairs:
            cross = env.distance(pair[:, None, None], np.asarray(pairs)[None])
            if cross.max() > options["cross_distance_cap"]:
                continue
        pairs.append(pair.tolist())
        unavailable.update(pair.tolist())
        if len(pairs) == options["pairs"]:
            break
    if len(pairs) != options["pairs"]:
        raise RuntimeError("Could not construct the contracted witness")
    pairs = np.asarray(pairs)
    diameters = env.distance(pairs[:, 0], pairs[:, 1])
    assert (diameters == env.diameter).all()
    matrix = env.distance(pairs.ravel()[:, None], pairs.ravel()[None]).reshape(len(pairs), 2, len(pairs), 2)
    # Each selected pair is diameter apart, but all four cross distances
    # between any two selected pairs are strictly smaller.
    mask = np.broadcast_to((~np.eye(len(pairs), dtype=bool))[:, None, :, None], matrix.shape)
    cross_maximum = int(matrix[mask].max())
    assert cross_maximum <= options["cross_distance_cap"]
    return {"pairs": pairs.tolist(), "attempts": attempt + 1, "diameter": env.diameter,
            "cross_distance_maximum": cross_maximum,
            "exact_linf_dimension_lower_bound": len(pairs),
            "uniform_error_lower_bound_below_this_dimension": (env.diameter - cross_maximum) / 2,
            "interpretation": "A coordinate cannot witness two of these pairs, in either orientation. This is a uniform all-pair error bound, not an average-error or action-ranking impossibility theorem."}


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--prepared-run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    args = parser.parse_args()
    options = json.loads(args.contract.read_text())
    env = Cube2(args.prepared_run / "oracle_depth.npy", metric=options["metric"])
    protected = np.load(args.prepared_run / "data.npz", allow_pickle=False)["protected_states"]
    result = {"source_commit": args.source_commit, "contract": options, **construct(env, options, protected)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as handle:
        handle.write(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "pairs"}))


if __name__ == "__main__":
    main()
