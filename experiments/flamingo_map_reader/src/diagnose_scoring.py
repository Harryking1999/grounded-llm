"""What computation does the readout label require of the candidate Q vectors?

The interface hands the language model a mixture of value vectors. Whether that is
enough depends entirely on how hard the target function is: if a linear score over
(candidate, goal) separates the pairs, a mixture could suffice; if the label needs a
quadratic form, the two vectors must reach the model separately. This measures both
ceilings on the same training pairs the run was given.
"""

from argparse import ArgumentParser
import json
from pathlib import Path

import numpy as np
import torch

from .data import first_turn_from_record, load_graph


def pair_vectors(records, source_root, config):
    graphs = {}
    rows = []
    for record in records:
        graph_id = record["graph_id"]
        if graph_id not in graphs:
            graphs[graph_id] = load_graph(source_root, graph_id)[:2]
        turn = first_turn_from_record(*graphs[graph_id], record)
        step = turn.step
        vectors = step.map_batch.vectors[0].float()
        left, right = record["candidate_pair"]
        distances = step.candidate_map_distances
        rows.append({"candidate_left": vectors[left + 1].clone(),
                     "candidate_right": vectors[right + 1].clone(),
                     "goal": vectors[1].clone(),
                     "left_is_closer": float(distances[left - 1] < distances[right - 1])})
    return rows


def features(rows, quadratic):
    """One row per (pair, side); a shared score must rank them, not memorise them."""
    blocks = []
    for row in rows:
        for side in ("left", "right"):
            candidate, goal = row[f"candidate_{side}"], row["goal"]
            block = [candidate, goal]
            if quadratic:
                block += [candidate * goal, candidate * candidate, goal * goal]
            blocks.append(torch.cat(block))
    return torch.stack(blocks).float()


def fit_ranking(rows, quadratic, split=0.5, steps=3000, seed=0):
    """Shared linear score trained with a pairwise logistic loss, as a probe ceiling."""
    torch.manual_seed(seed)
    x = features(rows, quadratic)
    first = torch.tensor([row["left_is_closer"] for row in rows]).repeat_interleave(2)
    order = torch.randperm(len(rows), generator=torch.Generator().manual_seed(seed))
    cut = int(len(rows) * split)
    train_pairs, test_pairs = order[:cut], order[cut:]
    # Interleave so each pair's two rows land adjacently; concatenating the two
    # sides would reshape lefts against lefts.
    def rows_of(pairs):
        return torch.stack([pairs * 2, pairs * 2 + 1], dim=1).reshape(-1)

    weight = torch.zeros(x.shape[1], requires_grad=True)
    optimizer = torch.optim.Adam([weight], lr=1e-2, weight_decay=1e-4)
    # Column order is (left, right); the class is 1 when the right candidate wins.
    class_of = lambda pairs: (first[pairs * 2] < 0.5).long()
    for _ in range(steps):
        scores = (x[rows_of(train_pairs)] @ weight).reshape(-1, 2)
        loss = torch.nn.functional.cross_entropy(scores, class_of(train_pairs))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        scores = (x[rows_of(test_pairs)] @ weight).reshape(-1, 2)
        correct = scores.argmax(dim=1) == class_of(test_pairs)
    return {"held_out_pairs": len(test_pairs), "accuracy": float(correct.float().mean())}


def closed_form_quadratic(rows):
    """Score -||candidate - goal||^2 directly; confirms the family contains a perfect ranker.

    -||c-g||^2 = -sum(c^2) + 2 sum(c*g) - sum(g^2), so the weight vector below is exact
    whatever the data looks like. Accuracy here is a ceiling, not a fit.
    """
    x = features(rows, quadratic=True)
    weight = torch.cat([torch.zeros(256), torch.full((128,), 2.0),
                        torch.full((128,), -1.0), torch.full((128,), -1.0)])
    scores = (x @ weight).reshape(-1, 2)
    truth = torch.tensor([row["left_is_closer"] for row in rows]).repeat_interleave(2)
    return {"pairs": len(rows),
            "accuracy": float((scores.argmax(dim=1) == (truth[::2] < 0.5).long())
                              .float().mean())}


def main():
    parser = ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if manifest["config"] != config:
        raise ValueError("Manifest and diagnostic configurations differ")
    records = [record for record in manifest["records"] if record["split"] == "train"]
    rows = pair_vectors(records, args.source_root, config)
    report = {"train_pairs": len(rows), "closed_form_quadratic": closed_form_quadratic(rows),
              "linear_candidate_goal": fit_ranking(rows, quadratic=False),
              "quadratic_candidate_goal": fit_ranking(rows, quadratic=True)}
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
