"""Run one contracted matched continuation condition; no new model modules."""
import argparse
import json
from pathlib import Path
import subprocess
import sys


def materialize(contract, condition):
    config = json.loads(Path(contract["base_config"]).read_text())
    config["training"].update(contract["training"])
    config["training"]["distance_supervision"] = contract["conditions"][condition]
    config["sampling"].update(contract["sampling"])
    return config


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--condition", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    contract = json.loads(args.contract.read_text())
    config_path = args.out.with_suffix(".config.json")
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(materialize(contract, args.condition), indent=2) + "\n")
    subprocess.run([sys.executable, "-u", "-m", "experiments.puzzle_qmap.src.run", "--config", str(config_path),
                    "--out", str(args.out), "--source-commit", args.source_commit, "--device", args.device,
                    "--prepared-run", contract["prepared_run"], "--checkpoint", contract["checkpoint"]], check=True)
