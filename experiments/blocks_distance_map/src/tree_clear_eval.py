"""Evaluate the frozen tree Q on fresh official boards with the empty goal."""
import argparse
import json
from pathlib import Path

from .rollout_eval import summarize, trace
from .multiboard_data import fresh_official_boards
from .oracle import DistanceOracle
from .tree_supervision_eval import QScorer


def evaluate(data_path, config_path, official_path, comparison_path,
             checkpoint_path, output_path, device="cuda:0"):
    fresh, comparison = fresh_official_boards(
        data_path, config_path, official_path, comparison_path)
    board_count = len(comparison["board_rows"])

    scorer = QScorer(checkpoint_path, device)
    oracle = DistanceOracle(cache_limit=2_000_000, seconds=3600)
    rows = []
    for board_row, source, _ in fresh:
        true_steps = oracle.distance(source, 0)
        if true_steps < 0:
            raise ValueError(f"Official board {board_row} cannot be cleared")
        outcome = trace(scorer, source, 0, true_steps, oracle)
        rows.append({"board_row": board_row, "source": str(source),
                     "true_steps": true_steps, **outcome})
    result = {"board_count": board_count, "goal": "empty",
              "scorer": scorer.identity, "summary": summarize(rows),
              "rows": rows}
    result["success_rate"] = result["summary"]["outcomes"].get("reached", 0) / board_count
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
    print(json.dumps({"board_count": board_count,
                      "outcomes": result["summary"]["outcomes"],
                      "success_rate": result["success_rate"]}, ensure_ascii=False))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--official", type=Path, required=True)
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    evaluate(args.data, args.config, args.official, args.comparison,
             args.checkpoint, args.out, args.device)


if __name__ == "__main__":
    main()
