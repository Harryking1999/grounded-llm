"""Offline algebra and deterministic replay of completed Huber sampling."""
import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path

import numpy as np
import torch

from .broad_sampling import BroadData
from .failure_diagnosis import RootOracle
from .model import BoardEncoder, board_bits, distance
from experiments.gcml_counterexamples.src import blocks


def closed_chains(observed):
    """Six actually observed ordered relations, not merely queryable triples."""
    adjacent = defaultdict(list)
    for (a, b), (truth, _) in observed.items():
        if truth == 1 and observed.get((b, a), (None,))[0] == -1:
            adjacent[a].append(b)
    witnesses, involved = [], set()
    for a, middle in adjacent.items():
        for b in middle:
            for c in adjacent.get(b, ()):
                if (observed.get((a, c), (None,))[0] == 2 and
                        observed.get((c, a), (None,))[0] == -1):
                    witnesses.append((a, b, c))
                    involved.update(((a, b), (b, a), (b, c), (c, b), (a, c), (c, a)))
    return witnesses, involved


def algebra(checkpoint, states):
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = saved["config"]["model"]
    dimension, cap = model["state_dim"], saved["cap"]
    encoder = BoardEncoder(dimension, model["encoder_hidden_dim"], cap, saved["metric"])
    encoder.load_state_dict(saved["model"])
    with torch.no_grad():
        z = encoder.encode(board_bits(states)).double()
        scores = distance(z[:, None], z[None, :], "directed_sum")
        phi = z.sum(-1) / math.sqrt(dimension)
        error = (scores - scores.T - (phi[:, None] - phi[None, :])).abs().max().item()
    return {"metric": saved["metric"], "dimension": dimension, "coordinate_cap": cap,
        "numeric_identity_max_error_float64_on_actual_embeddings": error,
        "witness_states": list(map(str, states)), "witness_predictions": scores.tolist(),
        "exact_forward": [1, 1, 2], "exact_reverse_unreachable_labels": [19, 19, 19],
        "label_cycle_residual_absolute": 19,
        "six_equal_weight_smooth_l1_terms_sum_lower_bound": 16,
        "dynamic_two_step_example_reverse": [19, 19, 38],
        "potential_and_distance_max_span": cap * math.sqrt(dimension),
        "minimum_potential_span_for_12_edges_with_each_reverse_at_least19": 12 * 18,
        "dynamic_label_conclusion": "Potential consistency is necessary, not sufficient: triangle inequalities and bounded directed-L1 coordinates also apply. A 12-edge chain with forward1 and every reverse>=19 requires span216, exceeding this encoder bound. No formal labels changed.",
        "single_clear_target_embedding_existence": "For arbitrary labels y(S) in[0,19], z(S)=y(S)/sqrt(dimension) in every coordinate and z(empty)=0 represents D(S,empty)=y(S), since19/sqrt(dimension)<cap. This does not prove that the finite MLP learns/generalizes that assignment."}


def run(config, checkpoint, completed_updates, out):
    torch.set_num_threads(2)
    options = config["training"]
    if options.get("rank_disjoint_roots", False):
        raise ValueError("This replay contract audits the original shared-root control")
    data = BroadData(config["graph_data"], config["legacy_data"], options, 0, 4)
    observed, histograms = {}, defaultdict(Counter)
    witness = None
    for root in data.roots:
        tiles = [blocks.PLACEMENTS[blocks.BY_ACTION[(a["shape_id"], a["row"], a["col"]) ]][0]
                 for a in root.record["construction"]]
        large = [t for t in tiles if t.bit_count() >= 3]
        if len(large) >= 2:
            a = int(root.record["initial"]); b = a ^ large[0]; c = b ^ large[1]
            oracle = RootOracle(root)
            assert [oracle.distance(x, y) for x, y in ((a,b),(b,c),(a,c),(b,a),(c,b),(c,a))] == [1,1,2,-1,-1,-1]
            witness = (a,b,c)
            break
    assert witness is not None
    for rank in range(4):
        data.rank = rank
        state = torch.load(Path(options["previous_sampler_root"]) /
                          f"sampler_{options['previous_sampler_step']:06d}_rank{rank}.pt",
                          map_location="cpu", weights_only=False)
        data.load_state_dict(state)
        for step in range(completed_updates):
            batch = data.batch(options["batch_size_per_rank"], options["rank_contexts_per_rank"])
            s, g, truth = batch[:3]
            empty = (g == 0).all(-1)
            legacy = np.arange(len(truth)) < round(len(truth) * options["legacy_fraction"])
            for name, chosen in (("clear", empty), ("nonempty", ~empty),
                                 ("legacy", legacy), ("decision_clear", data.last_huber_decision_mask)):
                histograms[name].update(map(int, truth[chosen]))
            rows, counts = np.unique(np.column_stack((s, g, truth.astype(np.int64).view(np.uint64))),
                                     axis=0, return_counts=True)
            for row, count in zip(rows, counts):
                a = int(row[0]) | (int(row[1]) << 64)
                b = int(row[2]) | (int(row[3]) << 64)
                label = int(row[4].view(np.int64))
                key = (a,b)
                previous = observed.get(key, (label, 0))
                if previous[0] != label:
                    raise ValueError("Exact label disagreement in sampled relations")
                observed[key] = (label, previous[1] + int(count))
        print(json.dumps({"replayed_completed_rank": rank, "updates": completed_updates}), flush=True)
    witnesses, involved = closed_chains(observed)
    result = {"scope": "Deterministic replay from the actual889 four-rank sampler states and original shared16 contract. Only the specified prefix of already completed updates is reconstructed. Co-exposure is cumulative over this prefix, not simultaneous within one batch; no new gradient updates or test data.",
        "completed_updates_replayed_per_rank": completed_updates, "ranks": 4,
        "huber_draws": sum(n for _, n in observed.values()), "distinct_exact_huber_pairs": len(observed),
        "actual_distance_histograms": {k: dict(v) for k,v in histograms.items()},
        "fully_coexposed_six_relation_contradictory_triples": len(witnesses),
        "distinct_pairs_in_these_witnesses": len(involved),
        "huber_draws_in_these_witnesses": sum(observed[p][1] for p in involved),
        "observed_finite_pairs_also_having_observed_reverse19": sum(
            d >= 0 and observed.get((b,a), (None,))[0] == -1 for (a,b),(d,n) in observed.items()),
        "example_coexposed_witnesses": [[str(x) for x in row] for row in witnesses[:10]],
        "algebra": algebra(checkpoint, witness)}
    Path(out).write_text(json.dumps(result, indent=2) + "\n")
    data.close()
    print(json.dumps({k:v for k,v in result.items() if k not in ("actual_distance_histograms", "algebra")}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--completed-updates", type=int, default=32)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    run(json.loads(Path(args.config).read_text()), args.checkpoint, args.completed_updates, args.out)
