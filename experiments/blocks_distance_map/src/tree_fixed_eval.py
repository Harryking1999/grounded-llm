"""Compare continuation checkpoints on fixed development rollouts."""
import argparse
import json
from pathlib import Path

from .hardbranch_failure import summarize, trace
from .oracle import DistanceOracle
from .tree_supervision_eval import QScorer


def evaluate(comparison_path, clear_path, checkpoint_path, output_path, device="cuda:0"):
    comparison = json.loads(Path(comparison_path).read_text())
    clear = json.loads(Path(clear_path).read_text())
    scorer = QScorer(checkpoint_path, device)
    oracle = DistanceOracle(cache_limit=2_000_000, seconds=3600)
    task_sets = {
        "nonempty": [(int(row["source"]), int(row["goal"]), int(row["true_steps"]))
                     for row in comparison["task_rows"]],
        "clear": [(int(row["source"]), 0, int(row["true_steps"]))
                  for row in clear["rows"]],
    }
    results = {}
    for name, tasks in task_sets.items():
        outcomes = [trace(scorer, source, goal, steps, oracle)
                    for source, goal, steps in tasks]
        results[name] = summarize(outcomes)
    result = {"scorer": scorer.identity, "results": results}
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({name: block["outcomes"] for name, block in results.items()}), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--clear", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    evaluate(args.comparison, args.clear, args.checkpoint, args.out, args.device)


if __name__ == "__main__":
    main()
