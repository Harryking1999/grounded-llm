import argparse
import json
from pathlib import Path
"""Compatibility imports for grounded_llm.tasks.blocks.data."""
from grounded_llm.tasks.blocks.data import Counter, defaultdict, Path, DistanceOracle, successors, collect, pair_splits, comparisons, prepare


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    prepare(json.loads(args.config.read_text()), args.out)


if __name__ == "__main__":
    main()
