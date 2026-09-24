"""Aggregate pass@k across the frozen independent graph suites."""

import argparse
from collections import Counter
import json
from pathlib import Path

from .evaluate_path256 import summarize


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
            "by_graph": [{"reached": s["reached"], "shortest": s["shortest"],
                          "pass_at": {str(k): s[f"shortest_pass_at_{k}"] / config["pair_count"]
                                      for k in config["pass_k"]}}
                         for s in graph_summaries],
        }
    path = args.root / "summary.json"
    path.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
