"""Measure actual Qwen token lengths before a full-trajectory training run."""

from argparse import ArgumentParser
from pathlib import Path
import json

import numpy as np
from transformers import AutoTokenizer

from .data import demonstration_from_record, load_graph
from .transcript import encode_trajectory


def _summary(values: list[int]) -> dict:
    return {str(percentile): float(np.percentile(values, percentile))
            for percentile in (0, 50, 90, 95, 99, 100)}


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    args = parser.parse_args()
    with args.config.open(encoding="utf-8") as handle:
        config = json.load(handle)
    with args.manifest.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest["study"] != config["study"]:
        raise ValueError("manifest and config refer to different studies")
    tokenizer = AutoTokenizer.from_pretrained(config["model"],
                                                cache_dir=args.cache_dir)
    graphs = {}
    counts = {split: {"sequence": [], "answer": [], "over_limit": 0}
              for split in ("train", "validation")}
    for record in manifest["records"]:
        graph_id = record["graph_id"]
        if graph_id not in graphs:
            environment, qmap, _ = load_graph(manifest["source_root"], graph_id)
            graphs[graph_id] = environment, qmap
        demonstration = demonstration_from_record(*graphs[graph_id], record)
        if not demonstration.success or len(demonstration.executed_path) - 1 != record["action_count"]:
            raise ValueError("a manifest trajectory did not reproduce")
        encoded = encode_trajectory(demonstration, tokenizer, max_tokens=100000)
        row = counts[record["split"]]
        row["sequence"].append(len(encoded.input_ids))
        row["answer"].append(encoded.answer_tokens)
        row["over_limit"] += len(encoded.input_ids) > config["maximum_sequence_tokens"]
    result = {
        "model_tokenizer": config["model"],
        "sequence_limit": config["maximum_sequence_tokens"],
        "splits": {
            split: {
                "count": len(row["sequence"]),
                "sequence_tokens_percentiles": _summary(row["sequence"]),
                "answer_tokens_percentiles": _summary(row["answer"]),
                "answer_tokens_mean": float(np.mean(row["answer"])),
                "over_limit": row["over_limit"],
            }
            for split, row in counts.items()
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
