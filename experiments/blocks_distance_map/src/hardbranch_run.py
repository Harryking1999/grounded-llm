"""Retrain the unchanged Q/loss on base pairs plus hard sibling branches."""
import argparse
import json
from pathlib import Path

import torch

from .hardbranch_data import prepare_augmented
from .multiboard_run import train_prepared


def run(contract, out, source_commit, device):
    base = Path(contract["base_run"])
    checkpoint = torch.load(base / "best.pt", map_location="cpu", weights_only=True)
    if checkpoint["source_commit"] != contract["base_source_commit"]:
        raise ValueError("Base checkpoint does not match the hard-branch contract")
    base_config = json.loads((base / "config.json").read_text())
    config = {**base_config, "hardbranch": contract}
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    (out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    (out / "source_commit.txt").write_text(source_commit + "\n")
    prepare_augmented(base / "data" / "data.npz", out / "data", contract)
    train_prepared(config, out, source_commit, device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    run(json.loads(args.config.read_text()), args.out, args.source_commit, args.device)


if __name__ == "__main__":
    main()
