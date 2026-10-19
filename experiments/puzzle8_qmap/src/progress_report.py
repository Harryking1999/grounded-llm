"""Read completed training checkpoints and report fixed-cohort rollout progress."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from grounded_llm.core.qmap.metrics import rollout
from grounded_llm.core.qmap.sampling import task_indices
from grounded_llm.tasks.puzzle8.environment import Puzzle8
from grounded_llm.tasks.puzzle8.model import QMap, MODEL_TYPE


def progress_rows(path, after_step=0):
    rows = []
    for line in path.read_text().splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            # The running trainer may be appending the last line.
            continue
        if row["step"] <= after_step:
            continue
        policy, sampling = row["policy"], row["sampling"]
        rows.append({"step": row["step"], "seconds": row["seconds"],
                     "restoration_solved": policy["within_budget"],
                     "restoration_rate": policy["within_budget_rate"],
                     "two_cycles": policy["two_cycles"],
                     "previously_solved_lost": policy["previously_solved_lost"],
                     "previously_failed_gained": policy["previously_failed_gained"],
                     "same1024_solved": policy["same1024"]["solved"],
                     "validation_mae": row["validation_mae"],
                     "validation_shortest_action_rate": row["validation_shortest_action_rate"],
                     "broad_pair_passes": sampling["all_pairs_passes"],
                     "broad_decision_passes": sampling["all_decision_passes"],
                     "best_step": row["best_step"]})
    return rows


def intermediate_report(run, prepared_run, device):
    """Evaluate the latest published checkpoint; never feed results to training."""
    cache_path = run / "intermediate_progress.jsonl"
    cached = ([json.loads(line) for line in cache_path.read_text().splitlines() if line]
              if cache_path.exists() else [])
    latest = progress_rows(run / "progress.jsonl")[-1]["step"]
    previous = next((row for row in cached if row["step"] == latest), None)
    if previous is not None:
        return previous
    # Load once. Subsequent training may overwrite last.pt; this in-memory
    # snapshot and its explicit step remain fixed throughout the evaluation.
    saved = torch.load(run / "last.pt", map_location=device, weights_only=False)
    step = saved["continuation_step"]
    if step != latest:
        return {"status": "checkpoint_advancing_retry_later", "step": step,
                "latest_published_step": latest}
    if saved["model_type"] != MODEL_TYPE or saved["task"] != "puzzle8":
        raise ValueError("Expected the matching Puzzle8 checkpoint")
    config = saved["config"]
    env = Puzzle8()
    with np.load(prepared_run / "oracle.npz") as oracle:
        env.tables = oracle["distances"]
        env.goal_columns = {int(g): i for i, g in enumerate(oracle["goals"])}
    with np.load(prepared_run / "data.npz") as packed:
        data = {f"{kind}_{split}": packed[f"{kind}_{split}"]
                for split in ("validation", "test")
                for kind in ("tasks", "parent_distances")}
    model = QMap(**config["model"]).to(device)
    model.load_state_dict(saved["model"])
    model.eval().requires_grad_(False)
    result = {"step": step, "checkpoint_step": saved["step"],
              "checkpoint_source_commit": saved["source_commit"],
              "scope": "read-only fixed-cohort monitoring; no selection or training feedback"}
    for split in ("validation", "test"):
        indices = task_indices(env, data, split, config["evaluation"]["rollout_count"], config)
        results = rollout(model, env, data[f"tasks_{split}"][indices], device,
                          config["evaluation"]["budget_multiplier"])
        result[split] = {"q": results["q"], "oracle_solved": results["oracle"]["solved"]}
    with cache_path.open("a") as output:
        output.write(json.dumps(result) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--after-step", type=int, default=0)
    parser.add_argument("--prepared-run", type=Path)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    torch.set_num_threads(1)
    report = {"stage": "continuation_updates_not_puzzle_moves",
              "rows": progress_rows(args.run / "progress.jsonl", args.after_step)}
    if args.prepared_run is not None:
        report["intermediate"] = intermediate_report(args.run, args.prepared_run, args.device)
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
