"""Score locally plausible, equal-area sibling branches with a frozen Q."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .hardbranch_data import find_cases
from .model import BoardEncoder, board_bits, distance
from .multiboard_eval import encoded_values
from .oracle import DistanceOracle


@torch.no_grad()
def score_cases(model, cases, metric, device):
    masks = sorted({mask for parent, good, bad, goal, steps in cases
                    for mask in (good, bad, goal)})
    ids = {mask: i for i, mask in enumerate(masks)}
    values = torch.cat([model.encode(board_bits(masks[i:i + 4096]).to(device))
                        for i in range(0, len(masks), 4096)])
    good = torch.as_tensor([ids[case[1]] for case in cases], device=device)
    bad = torch.as_tensor([ids[case[2]] for case in cases], device=device)
    goal = torch.as_tensor([ids[case[3]] for case in cases], device=device)
    good_scores = distance(values[good], values[goal], metric).cpu().numpy()
    bad_scores = distance(values[bad], values[goal], metric).cpu().numpy()
    return {"cases": len(cases), "strict_correct": int((good_scores < bad_scores).sum()),
            "ties": int((good_scores == bad_scores).sum()),
            "mean_margin": float((bad_scores - good_scores).mean())}


@torch.no_grad()
def score_case_ids(values, cases, metric):
    if not len(cases):
        return {"cases": 0}
    rows = torch.as_tensor(cases, device=values.device, dtype=torch.long)
    good_scores = distance(values[rows[:, 1]], values[rows[:, 3]], metric)
    bad_scores = distance(values[rows[:, 2]], values[rows[:, 3]], metric)
    margin = (bad_scores - good_scores).cpu().numpy()
    return {"cases": len(cases), "strict_correct": int((margin > 0).sum()),
            "ties": int((margin == 0).sum()), "mean_margin": float(margin.mean())}


def evaluate(data_path, checkpoint, split, max_pairs, max_cases, seed, device):
    data = dict(np.load(data_path, allow_pickle=False))
    oracle = DistanceOracle(cache_limit=2_000_000, seconds=3600)
    cases, counts = find_cases(data, split, max_pairs, max_cases, seed, oracle)
    saved = torch.load(checkpoint, map_location=device, weights_only=True)
    config = saved["config"]
    model = BoardEncoder(config["model"]["state_dim"],
                         config["model"]["encoder_hidden_dim"], saved["cap"],
                         saved["metric"]).to(device)
    model.load_state_dict(saved["model"])
    model.eval()
    result = {"split": split, "source_commit": saved["source_commit"],
              "case_counts": counts, "scores": score_cases(model, cases, saved["metric"], device)}
    print(json.dumps(result), flush=True)


def evaluate_prepared(data_path, checkpoint, device):
    data = dict(np.load(data_path, allow_pickle=False))
    saved = torch.load(checkpoint, map_location=device, weights_only=True)
    config = saved["config"]
    model = BoardEncoder(config["model"]["state_dim"],
                         config["model"]["encoder_hidden_dim"], saved["cap"],
                         saved["metric"]).to(device)
    model.load_state_dict(saved["model"])
    values = encoded_values(model, board_bits(data["states"]).to(device))
    result = {"source_commit": saved["source_commit"], "step": saved["step"],
              "hardbranch": {name: score_case_ids(values, data[f"hard_{name}_cases"],
                                                       saved["metric"])
                             for name in ("train", "validation", "ood_board", "unseen_goal")}}
    print(json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split")
    parser.add_argument("--prepared", action="store_true")
    parser.add_argument("--max-pairs", type=int, default=10000)
    parser.add_argument("--max-cases", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260929)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.prepared:
        evaluate_prepared(args.data, args.checkpoint, args.device)
    else:
        if not args.split:
            parser.error("--split is required unless --prepared is set")
        evaluate(args.data, args.checkpoint, args.split, args.max_pairs,
                 args.max_cases, args.seed, args.device)


if __name__ == "__main__":
    main()
