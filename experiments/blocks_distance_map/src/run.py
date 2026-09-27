"""Sequential bounded pilot: one shared dataset, matched Q-only conditions."""
import argparse
import json
from pathlib import Path
import time

from .data import prepare
from .train import train


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--data", type=Path, help="Reuse a data.npz already prepared from exactly this config")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    (args.out / "source_commit.txt").write_text(args.source_commit + "\n")
    started = time.monotonic()
    if args.data:
        if json.loads((args.data.parent / "config.json").read_text()) != config:
            raise ValueError("Existing dataset config differs")
        dataset = json.loads((args.data.parent / "summary.json").read_text())
        data_path = args.data
    else:
        dataset = prepare(config, args.out / "data")
        data_path = args.out / "data" / "data.npz"
    results = []
    for metric in config["training"]["metrics"]:
        for seed in config["training"]["seeds"]:
            result = train(config, data_path, args.out / f"{metric}_{seed}", metric, seed,
                           args.device, args.source_commit)
            results.append(result)
    (args.out / "summary.json").write_text(json.dumps({"source_commit": args.source_commit,
        "data_path": str(data_path.resolve()), "data": dataset, "runs": results,
        "seconds": time.monotonic() - started}, indent=2) + "\n")
    print(json.dumps({"queue_complete": str(args.out), "runs": len(results)}), flush=True)


if __name__ == "__main__":
    main()
