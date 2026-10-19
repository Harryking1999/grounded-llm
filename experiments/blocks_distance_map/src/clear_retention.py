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
    rows, graphs, completed = [], {}, 0
    decision_mode = config.get("prioritize_training_decisions", False)
    if decision_mode:
        from .failure_diagnosis import RootOracle
        from .full_graph_data import Root, load_component
    for board in selected:
        record = records[int(board)]
        if record["initial"] in excluded:
            raise ValueError("Fixed clearing evaluation root entered retention preparation")
        state, path, priorities = int(record["initial"]), [], []
        if decision_mode:
            for part in record["components"]:
                gid = part["graph"]
                if gid not in graphs:
                    graphs[gid] = load_component(Path(config["graph_data"]) / "graphs" / str(gid))
            oracle = RootOracle(Root(record, graphs))
        for _ in range(state.bit_count() // 2):
            candidates = successors(state)
            if not state or not candidates:
                break
            scores = scorer.scores([c for _, c in candidates], 0)
            chosen = int(scores.argmin())
            if decision_mode:
                truth = oracle.distances([c for _, c in candidates], 0)
                finite = truth >= 0
                if not finite.any():
                    raise ValueError("On-policy diagnostic parent must still be clearable")
                if 10 <= oracle.distance(state, 0) <= 12:
                    gap = float(scores[~finite].min() - scores[finite].min()) if (~finite).any() else 10
                    path.append(str(state))
                    # Sampling priority only. Every selected fork still keeps
                    # every legal child and its original exact distance label.
                    priorities.append(4.0 if truth[chosen] < 0 else 1.0 / (.25 + max(gap, 0)))
                if truth[chosen] < 0:
                    break
            else:
                path.append(str(state))
            state = candidates[chosen][1]
        completed += int(state == 0)
        if path and (state == 0 or decision_mode):
            rows.append({"board": int(board), "initial": record["initial"], "states": path,
                         **({"priorities": priorities} if decision_mode else {})})
        if len(scorer.cache) > 50000:
            scorer.cache.clear()
    for graph in graphs.values():
        graph.close()
    result = {"teacher": scorer.identity, "training_roots_examined": len(selected),
              "successful_training_roots": completed, "roots_with_contexts": len(rows), "rows": rows,
              "scope": "Training-only greedy path contexts. Decision mode includes first wrong clearable parent and weights fragile correct forks; exact labels revalidated by sampler. No teacher labels or losses. Fixed1000 roots excluded."}
    out = Path(config["training"]["clear_retention_pool"])
    out.write_text(json.dumps(result) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--config", type=Path, required=True)
    prepare(json.loads(parser.parse_args().config.read_text()))
