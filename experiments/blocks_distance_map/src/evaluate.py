"""Distance and action-selection diagnostics, with explicit oracle information."""
import numpy as np
import torch

from .model import strata


def ranks(values):
    _, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    midpoints = np.cumsum(counts) - (counts - 1) / 2
    return midpoints[inverse]


def auc(labels, scores):
    positive = np.asarray(labels, dtype=bool)
    npos, nneg = int(positive.sum()), int((~positive).sum())
    if not npos or not nneg:
        return None
    return float((ranks(scores)[positive].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def mean_or_none(values):
    return float(np.mean(values)) if len(values) else None


@torch.no_grad()
def predict_pairs(model, pairs):
    device = model.q.weight.device
    return np.concatenate([model(torch.as_tensor(chunk[:, :2], device=device, dtype=torch.long))
                           .cpu().numpy() for chunk in np.array_split(pairs, max(1, (len(pairs) + 8191) // 8192))])


def pair_metrics(pairs, scores, cap, outgoing, incoming, margin):
    finite = pairs[:, 2] >= 0
    labels = np.where(finite, pairs[:, 2], cap)
    error = np.abs(scores - labels)
    huber = np.where(error <= 1, .5 * error ** 2, error - .5)
    groups = strata(torch.as_tensor(pairs))
    calibrated = sum(float(huber[ids.numpy()].mean()) * weight for ids, weight in groups)
    ranking, stats = [], {}
    for name, rows in (("outgoing", outgoing), ("incoming", incoming)):
        if len(rows):
            gap = scores[rows[:, 1]] - scores[rows[:, 0]]
            loss = float(np.maximum(0, margin - gap).mean())
            ranking.append(loss)
            stats[name] = {"comparisons": len(rows), "accuracy": float((gap > 0).mean()),
                           "ties": float((gap == 0).mean()), "hinge_loss": loss}
        else:
            stats[name] = {"comparisons": 0, "accuracy": None, "ties": None, "hinge_loss": None}
    predicted_dead = scores >= cap - .5
    finite_scores, finite_labels = scores[finite], labels[finite]
    correlation = None
    if len(finite_scores) > 1 and np.std(finite_scores) > 0 and np.std(finite_labels) > 0:
        correlation = float(np.corrcoef(ranks(finite_scores), ranks(finite_labels))[0, 1])
    stats.update({"pairs": len(pairs), "finite_pairs": int(finite.sum()),
                  "finite_mae": mean_or_none(error[finite]),
                  "finite_rounded_accuracy": mean_or_none(np.rint(scores[finite]) == labels[finite]),
                  "finite_spearman_pooled": correlation,
                  "unreachable_auc": auc(~finite, scores),
                  "unreachable_recall_at_cap_minus_half": mean_or_none(predicted_dead[~finite]),
                  "finite_false_unreachable_rate": mean_or_none(predicted_dead[finite]),
                  "hard_unreachable_recall": mean_or_none(predicted_dead[pairs[:, 3] == 2]),
                  "distance_loss": calibrated,
                  "ranking_loss": mean_or_none(ranking) or 0.})
    return stats


def evaluate_split(data, split_name, all_scores, margin):
    sid = ("train", "validation", "test").index(split_name)
    selected = np.flatnonzero(data["split"] == sid)
    local = np.full(len(data["pairs"]), -1, dtype=np.int64)
    local[selected] = np.arange(len(selected))
    comparisons = [local[data[f"{split_name}_{axis}"]] for axis in ("outgoing", "incoming")]
    if any(np.any(rows < 0) for rows in comparisons):
        raise ValueError("Comparison uses labels from another split")
    return pair_metrics(data["pairs"][selected], all_scores[selected], int(data["cap"]),
                        *comparisons, margin)


def decisions_metrics(data, goal_scores):
    rows = data["decisions"]
    state_cells = np.asarray([int(s).bit_count() for s in data["states"]])
    good_picks, optimal_picks, bad_picks = [], [], []
    tie_sizes, hard_pair_correct, same_area_correct, deep_pair_correct = [], [], [], []
    for parent in np.unique(rows[:, 0]):
        candidates = rows[rows[:, 0] == parent]
        d = candidates[:, 3]
        scores = goal_scores[candidates[:, 2]]
        good = d >= 0
        if not good.any():
            continue
        # Expected outcome under uniform tie-breaking, not favorable action IDs.
        tied = np.isclose(scores, scores.min(), atol=1e-7, rtol=0)
        good_picks.append(float(good[tied].mean()))
        bad_picks.append(float((~good[tied]).mean()))
        optimal_picks.append(float((d[tied] == d[good].min()).mean()))
        tie_sizes.append(int(tied.sum()))
        for gi in np.flatnonzero(good):
            for bi in np.flatnonzero(~good):
                correct = float(scores[gi] < scores[bi])
                hard_pair_correct.append(correct)
                if state_cells[candidates[gi, 2]] == state_cells[candidates[bi, 2]]:
                    same_area_correct.append(correct)
                if not data["immediate_dead"][candidates[bi, 2]]:
                    deep_pair_correct.append(correct)
    return {"parents": len(good_picks), "solvable_action_rate": mean_or_none(good_picks),
            "optimal_action_rate": mean_or_none(optimal_picks),
            "dead_entry_rate": mean_or_none(bad_picks), "mean_tied_actions": mean_or_none(tie_sizes),
            "good_dead_pairs": len(hard_pair_correct), "good_dead_accuracy": mean_or_none(hard_pair_correct),
            "same_cells_good_dead_pairs": len(same_area_correct),
            "same_cells_good_dead_accuracy": mean_or_none(same_area_correct),
            "deep_dead_pairs": len(deep_pair_correct), "deep_dead_accuracy": mean_or_none(deep_pair_correct)}


@torch.no_grad()
def final_metrics(model, data, margin):
    scores = predict_pairs(model, data["pairs"])
    goal = int(data["goal_id"])
    goal_pairs = np.column_stack((np.arange(len(data["states"])), np.full(len(data["states"]), goal)))
    goal_scores = predict_pairs(model, goal_pairs)
    result = {name: evaluate_split(data, name, scores, margin)
              for name in ("train", "validation", "test")}
    result["heldout_goal_decisions"] = decisions_metrics(data, goal_scores)
    # Exact distances are an evaluation-only upper bound, never model inputs.
    true_goal = np.where(data["goal_labels"] >= 0, data["goal_labels"], int(data["cap"]))
    result["oracle_decisions"] = decisions_metrics(data, true_goal)
    area = np.asarray([int(s).bit_count() for s in data["states"]], dtype=np.float32)
    result["remaining_cells_decisions"] = decisions_metrics(data, area)
    result["q_centered_rms"] = float((model.q.weight - model.q.weight.mean(0)).square().mean().sqrt())
    result["q_min"] = float(model.q.weight.min())
    result["q_max"] = float(model.q.weight.max())
    return result
