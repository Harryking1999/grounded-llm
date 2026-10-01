"""Combine disjoint graph-evaluation shards and check reserved-suite coverage."""

from argparse import ArgumentParser
from collections import Counter, defaultdict
import json
from pathlib import Path

from .data import load_graph


def main():
    parser = ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--shard-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    expected = {}
    for graph_id in [*config["train_validation_graphs"], config["unseen_test_graph"]]:
        _, _, suite = load_graph(args.source_root, graph_id)
        expected.update({(graph_id, case["id"]): case["reference"]["length"]
                         for case in suite["cases"]})

    seen = set()
    counts = defaultdict(Counter)
    for path in sorted(args.shard_root.glob("shard_*/logs/rollouts.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            key = (row["graph_id"], row["case_id"])
            if key not in expected or key in seen:
                raise ValueError(f"Unexpected or duplicate reserved case: {key}")
            seen.add(key)
            length = expected[key]
            for group in ("overall", f"graph:{key[0]}", f"length:{length}",
                          f"graph_length:{key[0]}:{length}"):
                bucket = counts[group]
                bucket.update(attempts=1, reached=int(row["reached_goal"]),
                              success=int(row["success"]),
                              shortest=int(row["shortest_success"]))
                if row["failure"]:
                    bucket[row["failure"]] += 1
    if seen != expected.keys():
        raise ValueError(f"Reserved suite incomplete: {len(seen)}/{len(expected)}")
    summary = {key: dict(value,
                         goal_reach_rate=value["reached"] / value["attempts"],
                         shortest_success_rate=value["shortest"] / value["attempts"])
               for key, value in sorted(counts.items())}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary["overall"]))


if __name__ == "__main__":
    main()
