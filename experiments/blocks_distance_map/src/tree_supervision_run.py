"""Retrain unchanged shared Q on goal-relative sampled-DAG relations."""
import argparse
import json
from pathlib import Path

import torch

from .multiboard_run import train_prepared
from .tree_supervision_data import prepare_tree


def run(contract, out, source_commit, device, max_boards=None):
    base = Path(contract["base_run"])
    checkpoint = torch.load(base / "best.pt", map_location="cpu", weights_only=True)
    if checkpoint["source_commit"] != contract["base_source_commit"]:
        raise ValueError("Base data source differs from the formal contract")
    base_config = json.loads((base / "config.json").read_text())
    config = {**base_config, "tree_supervision": contract}
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    (out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    (out / "source_commit.txt").write_text(source_commit + "\n")
    prepare_tree(base / "data" / "data.npz", Path(contract["official_dataset"]),
                 out / "data", contract, max_boards=max_boards)
    if max_boards is None:
        train_prepared(config, out, source_commit, device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-boards", type=int,
                        help="Data-only smoke on the first N training boards")
    args = parser.parse_args()
    run(json.loads(args.config.read_text()), args.out,
        args.source_commit, args.device, args.max_boards)


if __name__ == "__main__":
    main()
