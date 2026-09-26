"""Aggregate pass@k across the frozen independent graph suites."""

import argparse
from collections import Counter
import json
from pathlib import Path

from .evaluate_path256 import summarize


def inference_cost(records):
    """Keep inference costs for failed attempts as well as successful ones."""
    shortest = [r for r in records if r["shortest"]]
    return {
        "trials": len(records),
        "shortest_trials": len(shortest),
        "decision_count": sum(len(r["trace"]) for r in records),
        "truncated_decision_count": sum(
            t["response"]["finish_reason"] == "length"
            for r in records for t in r["trace"]),
        "total_output_tokens": sum(r["output_tokens"] for r in records),
        "total_input_tokens": sum(r["input_tokens"] for r in records),
        "total_model_seconds": sum(r["elapsed_seconds"] for r in records),
        "shortest_trial_output_tokens": sum(r["output_tokens"] for r in shortest),
        "shortest_trial_model_seconds": sum(r["elapsed_seconds"] for r in shortest),
    }


def combine_inference_cost(costs):
    total = {key: sum(cost[key] for cost in costs) for key in costs[0]}

    def ratio(numerator, denominator):
        return total[numerator] / total[denominator] if total[denominator] else None

    return {
        **total,
        "mean_output_tokens_per_trial": ratio("total_output_tokens", "trials"),
        "mean_input_tokens_per_trial": ratio("total_input_tokens", "trials"),
        "mean_model_seconds_per_trial": ratio("total_model_seconds", "trials"),
        "mean_output_tokens_per_decision": ratio("total_output_tokens", "decision_count"),
        "mean_input_tokens_per_decision": ratio("total_input_tokens", "decision_count"),
        "mean_model_seconds_per_decision": ratio("total_model_seconds", "decision_count"),
        "truncated_decision_rate": ratio("truncated_decision_count", "decision_count"),
        "mean_output_tokens_per_shortest_trial": ratio("shortest_trial_output_tokens", "shortest_trials"),
        "mean_model_seconds_per_shortest_trial": ratio("shortest_trial_model_seconds", "shortest_trials"),
        "total_output_tokens_per_shortest_success": ratio("total_output_tokens", "shortest_trials"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--suite-config", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    suite_config = json.loads(args.suite_config.read_text(encoding="utf-8"))
    graph_count = len(suite_config["graph_seeds"])
    questions = graph_count * config["pair_count"]
    expected_trials = config["pair_count"] * config["replicates"]
    output = {"questions": questions, "replicates": config["replicates"], "conditions": {}}
    for condition in config["conditions"]:
        graph_summaries = []
        graph_costs = []
        for graph_index in range(graph_count):
            folder = args.root / f"graph_{graph_index:02d}" / condition
            records = [json.loads(path.read_text(encoding="utf-8"))
                       for path in sorted((folder / "samples").glob("*.json"))]
            if len(records) != expected_trials:
                raise ValueError(f"Incomplete {condition} graph {graph_index}: {len(records)} trials")
            summary = summarize(records, config["pass_k"])
            if summary != json.loads((folder / "summary.json").read_text(encoding="utf-8")):
                raise ValueError(f"Summary mismatch: {folder}")
            graph_summaries.append(summary)
            graph_costs.append(inference_cost(records))
        failures = Counter()
        for summary in graph_summaries:
            failures.update(summary["failures"])
        output["conditions"][condition] = {
            "trials": sum(s["trials"] for s in graph_summaries),
            "reached": sum(s["reached"] for s in graph_summaries),
            "shortest": sum(s["shortest"] for s in graph_summaries),
            "pass_at": {str(k): sum(s[f"shortest_pass_at_{k}"] for s in graph_summaries) / questions
                        for k in config["pass_k"]},
            "failures": failures,
            "inference_cost": combine_inference_cost(graph_costs),
            "by_graph": [{"reached": s["reached"], "shortest": s["shortest"],
                          "pass_at": {str(k): s[f"shortest_pass_at_{k}"] / config["pair_count"]
                                      for k in config["pass_k"]},
                          "inference_cost": combine_inference_cost([cost])}
                         for s, cost in zip(graph_summaries, graph_costs)],
        }
    path = args.root / "summary.json"
    path.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
