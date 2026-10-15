"""Run one bounded comparison using the shared pure-Q training pipeline."""
import argparse
import json
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--prepared-run", type=Path, required=True)
    parser.add_argument("--condition", choices=("control", "repair"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    contract = json.loads(args.contract.read_text())
    config = json.loads((args.prepared_run / "config.json").read_text())
    parent = json.loads((args.prepared_run / "summary.json").read_text())
    if parent["source_commit"] != contract["prepared_source_commit"]:
        raise ValueError("Expected the specified frozen starting run")
    config["seed"] = contract["seed"]
    config["training"].update(contract["training"])
    config["sampling"].update(contract["sampling"])
    config["sampling"].update(contract["conditions"][args.condition])
    config_path = args.out.with_suffix(".config.json")
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(config, indent=2) + "\n")
    subprocess.run([sys.executable, "-u", "-m", "experiments.puzzle_qmap.src.run",
                    "--config", str(config_path), "--prepared-run", str(args.prepared_run),
                    "--checkpoint", str(args.prepared_run / contract["checkpoint"]),
                    "--out", str(args.out), "--source-commit", args.source_commit,
                    "--device", args.device], check=True)


if __name__ == "__main__":
    main()
