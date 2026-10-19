"""Retain successful training-only greedy clearing paths as exact fork contexts."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .oracle import successors
from .tree_supervision_eval import QScorer


def prepare(config):
    torch.set_num_threads(2)
    records = json.loads((Path(config["graph_data"]) / "roots.json").read_text())
    excluded = {row["source"] for row in json.loads(Path(config["baseline_cases"]).read_text())["cases"]}
    scorer = QScorer(config["parent_checkpoint"], "cpu")
    selected = np.linspace(0, len(records) - 1, config["retention_training_roots"], dtype=int)
    rows = []
    for board in selected:
        record = records[int(board)]
        if record["initial"] in excluded:
            raise ValueError("Fixed clearing evaluation root entered retention preparation")
        state, path = int(record["initial"]), []
        for _ in range(state.bit_count() // 2):
            candidates = successors(state)
            if not state or not candidates:
                break
            path.append(str(state))
            state = candidates[int(scorer.scores([c for _, c in candidates], 0).argmin())][1]
        if state == 0:
            rows.append({"board": int(board), "initial": record["initial"], "states": path})
        if len(scorer.cache) > 50000:
            scorer.cache.clear()
    result = {"teacher": scorer.identity, "training_roots_examined": len(selected),
              "successful_training_roots": len(rows), "rows": rows,
              "scope": "Original greedy successful training paths only; graph distances revalidated by sampler. No teacher labels or extra losses. Fixed1000 roots excluded."}
    out = Path(config["training"]["clear_retention_pool"])
    out.write_text(json.dumps(result) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--config", type=Path, required=True)
    prepare(json.loads(parser.parse_args().config.read_text()))
