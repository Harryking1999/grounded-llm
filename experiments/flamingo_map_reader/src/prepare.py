"""Create the frozen train/validation pair manifest from existing graph maps."""

from argparse import ArgumentParser
from pathlib import Path
import json

from .data import build_manifest


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with args.config.open(encoding="utf-8") as handle:
        config = json.load(handle)
    manifest = build_manifest(args.source_root, config)
    manifest["config"] = config
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")
    counts = {split: sum(record["split"] == split for record in manifest["records"])
              for split in ("train", "validation")}
    print(json.dumps({"manifest": str(args.output), "counts": counts,
                      "excluded": manifest["excluded"]}))


if __name__ == "__main__":
    main()
