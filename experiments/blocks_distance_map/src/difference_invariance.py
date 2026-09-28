"""Test whether Q distance is stable when the removable difference stays fixed."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np

from .hardbranch_eval import load_values
from .multiboard_eval import pair_scores


def repeated_differences(data, split):
    states = [int(mask) for mask in data["states"]]
    rows = data[split]
    groups = defaultdict(list)
    for index, (source, goal, steps, _) in enumerate(rows):
        if steps >= 0 and states[int(goal)] != 0:
            s, g = states[int(source)], states[int(goal)]
            if s & g == g:
                groups[s ^ g].append(index)
    repeated = [indices for indices in groups.values()
                if len(indices) >= 2 and len({int(rows[i, 1]) for i in indices}) >= 2]
    for indices in repeated:
        if len({int(rows[i, 2]) for i in indices}) != 1:
            raise AssertionError("Exact-cover distance changed under shared background")
    return repeated


def analyze(data_path, checkpoints, out, device):
    data = dict(np.load(data_path, allow_pickle=False))
    result = {"checkpoints": {}, "splits": {}}
    for name, path in checkpoints.items():
        saved, _, values = load_values(path, data, device)
        result["checkpoints"][name] = {"source_commit": saved["source_commit"],
                                       "step": saved["step"],
                                       "finetune_source_commit": saved.get("finetune_source_commit"),
                                       "finetune_step": saved.get("finetune_step"),
                                       "path": str(path)}
        for split in ("test_ood_board", "test_unseen_goal"):
            groups = repeated_differences(data, split)
            scores = pair_scores(values, data[split], saved["metric"])
            ranges = [float(np.ptp(scores[indices])) for indices in groups]
            deviations = [float(np.std(scores[indices])) for indices in groups]
            result["splits"].setdefault(split, {"repeated_differences": len(groups),
                                                "rows": sum(len(indices) for indices in groups),
                                                "models": {}})["models"][name] = {
                "median_predicted_range": float(np.median(ranges)) if ranges else None,
                "mean_predicted_range": float(np.mean(ranges)) if ranges else None,
                "median_predicted_std": float(np.median(deviations)) if deviations else None,
                "fraction_range_above_one_step": float(np.mean(np.asarray(ranges) > 1))
                if ranges else None}
    if out:
        Path(out).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--current", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    analyze(args.data, {"base": args.base, "current": args.current}, args.out, args.device)


if __name__ == "__main__":
    main()
