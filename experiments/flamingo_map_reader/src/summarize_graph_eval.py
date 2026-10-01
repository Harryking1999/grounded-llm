"""Combine disjoint graph-evaluation shards and check reserved-suite coverage."""

from argparse import ArgumentParser
from collections import Counter, defaultdict
import json
from pathlib import Path

import numpy as np

from .data import load_graph
from .graph import graph_step
from .relations import score_relationships


def score_rollout_relationships(row, environment, qmap, case, seed):
    """Replay only the frozen map and actual path; never infer labels from text."""
    rng = np.random.default_rng(seed + int(case["id"].split("_")[-1]))
    path = [int(case["start"])]
    scored = []
    for item in row["trace"]:
        step = graph_step(environment, qmap, path[-1], int(case["goal"]),
                          executed_path=path, rng=rng)
        if item["current"] != path[-1] or tuple(item["candidate_destinations"]) != step.candidate_destinations:
            raise ValueError(f"Trace/map mismatch: {row['graph_id']}, {row['case_id']}")
        if not step.done and step.candidate_actions:
            scored.append(score_relationships(step, item["answer"]))
        if "chosen_id" in item:
            _, destination = step.execute(environment, item["chosen_id"])
            if destination != item["after"]:
                raise ValueError("Trace action does not match the environment")
            path.append(destination)
    if path != row["path"]:
        raise ValueError("Trace does not reproduce the executed path")
    return scored


def add_relation(bucket, result):
    bucket.update(turns=1, valid_ranking=int(result["valid_ranking"]),
                  pairwise_correct=result["pairwise_correct"],
                  pairwise_total=result["pairwise_total"],
                  exact_ranking=int(result["exact_ranking"]),
                  closest_candidate_set_exact=int(result["closest_candidate_set_exact"]),
                  action_map_minimum=int(result["action_map_minimum"]),
                  action_follows_ranking=int(result["action_follows_ranking"]))


def relation_summary(counts):
    return {key: dict(value,
                      valid_ranking_rate=value["valid_ranking"] / value["turns"],
                      pairwise_accuracy=value["pairwise_correct"] / value["pairwise_total"],
                      exact_ranking_rate=value["exact_ranking"] / value["turns"],
                      closest_candidate_accuracy=value["closest_candidate_set_exact"] / value["turns"],
                      action_map_minimum_rate=value["action_map_minimum"] / value["turns"],
                      action_follows_ranking_rate=value["action_follows_ranking"] / value["turns"])
            for key, value in sorted(counts.items()) if value["turns"]}


def main():
    parser = ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--shard-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    expected = {}
    graphs = {}
    for graph_id in [*config["train_validation_graphs"], config["unseen_test_graph"]]:
        environment, qmap, suite = load_graph(args.source_root, graph_id)
        graphs[graph_id] = environment, qmap
        expected.update({(graph_id, case["id"]): case
                         for case in suite["cases"]})

    seen = set()
    counts = defaultdict(Counter)
    relationships = defaultdict(Counter)
    for path in sorted(args.shard_root.glob("shard_*/logs/rollouts.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            key = (row["graph_id"], row["case_id"])
            if key not in expected or key in seen:
                raise ValueError(f"Unexpected or duplicate reserved case: {key}")
            seen.add(key)
            case = expected[key]
            length = case["reference"]["length"]
            for group in ("overall", f"graph:{key[0]}", f"length:{length}",
                          f"graph_length:{key[0]}:{length}"):
                bucket = counts[group]
                bucket.update(attempts=1, reached=int(row["reached_goal"]),
                              success=int(row["success"]),
                              shortest=int(row["shortest_success"]))
                if row["failure"]:
                    bucket[row["failure"]] += 1
            for turn, result in enumerate(score_rollout_relationships(
                    row, *graphs[key[0]], case, config["seed"])):
                for group in ("overall", f"graph:{key[0]}", f"length:{length}",
                              f"graph_length:{key[0]}:{length}",
                              "first_turn" if turn == 0 else "later_turn"):
                    add_relation(relationships[group], result)
    if seen != expected.keys():
        raise ValueError(f"Reserved suite incomplete: {len(seen)}/{len(expected)}")
    summary = {key: dict(value,
                         goal_reach_rate=value["reached"] / value["attempts"],
                         shortest_success_rate=value["shortest"] / value["attempts"])
               for key, value in sorted(counts.items())}
    summary["relationships"] = relation_summary(relationships)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary["overall"]))


if __name__ == "__main__":
    main()
