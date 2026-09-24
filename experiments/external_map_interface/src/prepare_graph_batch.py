"""Freeze several independent path suites with the existing graph generator."""

import argparse
from copy import deepcopy
import json
from pathlib import Path

from experiments.qwen_path_blocks.src.prepare_path_suite import prepare


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    seeds = config["graph_seeds"]
    if len(seeds) != 5 or len(set(seeds)) != 5:
        raise ValueError("Expected five distinct graph seeds")
    if args.out.exists():
        raise FileExistsError(args.out)
    suites = []
    for index, seed in enumerate(seeds):
        graph_config = deepcopy(config)
        graph_config.pop("graph_seeds")
        graph_config["seed"] = seed
        suite = prepare(graph_config)
        suites.append((index, seed, suite))
    args.out.mkdir(parents=True)
    for index, seed, suite in suites:
        path = args.out / f"graph_{index:02d}" / "suite.json"
        path.parent.mkdir()
        path.write_text(json.dumps(suite, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"graph": index, "seed": seed, "cases": len(suite["cases"]),
                          "replicates": suite["config"]["replicates"], "path": str(path)}), flush=True)


if __name__ == "__main__":
    main()
