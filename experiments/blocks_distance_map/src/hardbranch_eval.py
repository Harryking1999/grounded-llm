"""Score locally plausible, equal-area sibling branches with a frozen Q."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .hardbranch_data import find_cases
from .model import BoardEncoder, board_bits, distance
from .multiboard_eval import encoded_values, goal_rollouts
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
def case_margins(values, cases, metric):
    if not len(cases):
        return np.empty(0, dtype=np.float32)
    rows = torch.as_tensor(cases, device=values.device, dtype=torch.long)
    good_scores = distance(values[rows[:, 1]], values[rows[:, 3]], metric)
    bad_scores = distance(values[rows[:, 2]], values[rows[:, 3]], metric)
    return (bad_scores - good_scores).cpu().numpy()


def score_case_ids(values, cases, metric):
    margin = case_margins(values, cases, metric)
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


def load_values(checkpoint, data, device):
    saved = torch.load(checkpoint, map_location=device, weights_only=True)
    config = saved["config"]
    model = BoardEncoder(config["model"]["state_dim"],
                         config["model"]["encoder_hidden_dim"], saved["cap"],
                         saved["metric"]).to(device)
    model.load_state_dict(saved["model"])
    values = encoded_values(model, board_bits(data["states"]).to(device))
    return saved, model, values


def evaluate_prepared(data_path, checkpoint, device, compare_checkpoint=None, out=None,
                      analysis_commit=None, rollout_count=200):
    data = dict(np.load(data_path, allow_pickle=False))
    saved, model, values = load_values(checkpoint, data, device)
    result = {"source_commit": saved["source_commit"], "step": saved["step"],
              "finetune_source_commit": saved.get("finetune_source_commit"),
              "finetune_step": saved.get("finetune_step"),
              "analysis_commit": analysis_commit,
              "hardbranch": {name: score_case_ids(values, data[f"hard_{name}_cases"],
                                                       saved["metric"])
                             for name in ("train", "validation", "ood_board", "unseen_goal")}}
    if compare_checkpoint is not None:
        other, other_model, other_values = load_values(compare_checkpoint, data, device)
        comparison = {"source_commit": other["source_commit"], "step": other["step"],
                      "finetune_source_commit": other.get("finetune_source_commit"),
                      "finetune_step": other.get("finetune_step"),
                      "hardbranch": {}, "rollouts": {}}
        for name in ("train", "validation", "ood_board", "unseen_goal"):
            cases = data[f"hard_{name}_cases"]
            baseline = case_margins(other_values, cases, other["metric"]) > 0
            current = case_margins(values, cases, saved["metric"]) > 0
            comparison["hardbranch"][name] = {
                "baseline_correct": int(baseline.sum()),
                "current_correct": int(current.sum()),
                "rescued": int((~baseline & current).sum()),
                "regressed": int((baseline & ~current).sum())}
        for name in ("ood_board", "unseen_goal"):
            tasks = data[f"hard_{name}_cases"][:rollout_count][:, [0, 3]]
            comparison["rollouts"][name] = {
                "baseline": goal_rollouts(other_model, tasks, data["states"],
                                           other["metric"], device),
                "current": goal_rollouts(model, tasks, data["states"], saved["metric"], device)}
        result["comparison"] = comparison
    if out is not None:
        Path(out).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split")
    parser.add_argument("--prepared", action="store_true")
    parser.add_argument("--compare-checkpoint", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--analysis-commit")
    parser.add_argument("--rollout-count", type=int, default=200)
    parser.add_argument("--max-pairs", type=int, default=10000)
    parser.add_argument("--max-cases", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260929)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.prepared:
        evaluate_prepared(args.data, args.checkpoint, args.device,
                          args.compare_checkpoint, args.out, args.analysis_commit,
                          args.rollout_count)
    else:
        if not args.split:
            parser.error("--split is required unless --prepared is set")
        evaluate(args.data, args.checkpoint, args.split, args.max_pairs,
                 args.max_cases, args.seed, args.device)


if __name__ == "__main__":
    main()
