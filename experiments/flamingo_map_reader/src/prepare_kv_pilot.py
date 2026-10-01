"""Make paired small-data joint/KV manifests from frozen full task manifests."""

from argparse import ArgumentParser
from collections import defaultdict
from copy import deepcopy
import json
from pathlib import Path
import random


def select_records(task, source, options, seed):
    rng = random.Random(seed)
    if task == "graph":
        selected = []
        graph_ids = source["config"]["train_validation_graphs"]
        for split in ("train", "validation"):
            count = options[f"{split}_per_graph"]
            for graph_id in graph_ids:
                candidates = [r for r in source["records"]
                              if r["split"] == split and r["graph_id"] == graph_id]
                if len(candidates) < count:
                    raise ValueError(f"Not enough {split} records for {graph_id}")
                selected.extend(rng.sample(candidates, count))
        return selected
    by_board = defaultdict(list)
    for record in source["records"]:
        if record["split"] == "train":
            by_board[record["board_row"]].append(record)
    required = options["train_boards"] + options["validation_boards"]
    if len(by_board) < required:
        raise ValueError("Not enough distinct training boards for pilot")
    board_ids = rng.sample(sorted(by_board), required)
    selected = []
    for index, board_id in enumerate(board_ids):
        split = "train" if index < options["train_boards"] else "validation"
        count = options["pairs_per_board"]
        if len(by_board[board_id]) < count:
            raise ValueError(f"Board {board_id} has too few examples")
        for record in rng.sample(by_board[board_id], count):
            selected.append(dict(record, split=split))
    return selected


def main():
    parser = ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    args = parser.parse_args()
    contract = json.loads(args.contract.read_text(encoding="utf-8"))
    config_root = args.contract.parent
    if args.out_root.exists():
        raise FileExistsError(args.out_root)
    prepared = []
    for task in ("graph", "blocks"):
        source_path = args.data_root / contract[task]["source_manifest"]
        source = json.loads(source_path.read_text(encoding="utf-8"))
        old_name = "pilot_path256.json" if task == "graph" else "pilot_blocks.json"
        kv_name = old_name.replace(".json", "_addressed_kv.json")
        old_config = json.loads((config_root / old_name).read_text(encoding="utf-8"))
        kv_config = json.loads((config_root / kv_name).read_text(encoding="utf-8"))
        if source["config"] != old_config:
            raise ValueError(f"Frozen {task} source manifest differs from baseline contract")
        records = select_records(task, source, contract[task], contract["seed"])
        for mode, original in (("joint", old_config), ("addressed_kv", kv_config)):
            config = deepcopy(original)
            config["study"] += "_small_pilot"
            config["training"]["epochs"] = contract["epochs"]
            config["pilot"] = {"selection_contract": contract,
                               "source_manifest": str(source_path.resolve())}
            manifest = {"config": config, "records": records,
                        "pilot_source_manifest": str(source_path.resolve())}
            source_field = "source_root" if task == "graph" else "q_checkpoint"
            manifest[source_field] = source[source_field]
            prepared.append((task, mode, config, manifest))
    args.out_root.mkdir(parents=True)
    for task, mode, config, manifest in prepared:
        prefix = args.out_root / f"{task}_{mode}"
        prefix.with_suffix(".config.json").write_text(
            json.dumps(config, indent=2) + "\n", encoding="utf-8")
        prefix.with_suffix(".manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"task": task, "mode": mode,
                          "train": sum(r["split"] == "train" for r in manifest["records"]),
                          "validation": sum(r["split"] == "validation" for r in manifest["records"])}))


if __name__ == "__main__":
    main()
