"""Frozen-Q diagnostics for directed multi-goal relations and goal swaps."""
from collections import defaultdict

import numpy as np
import torch

from .evaluate import auc
from .model import distance, strata
from .oracle import successors


def listwise_groups(pairs, seed, width=4, per_anchor=8):
    rng = np.random.default_rng(seed)
    groups = []
    for axis in (0, 1):
        by_anchor = defaultdict(list)
        for index, row in enumerate(pairs):
            by_anchor[int(row[axis])].append(index)
        for indices in by_anchor.values():
            levels = defaultdict(list)
            for index in indices:
                levels[int(pairs[index, 2]) if pairs[index, 2] >= 0 else 1000].append(index)
            if len(levels) < 2:
                continue
            keys = sorted(levels)
            for _ in range(per_anchor):
                selected = [int(rng.choice(levels[keys[0]])), int(rng.choice(levels[keys[-1]]))]
                selected.extend(int(rng.choice(indices)) for _ in range(width - 2))
                rng.shuffle(selected)
                groups.append(selected)
    return np.asarray(groups, dtype=np.int32).reshape(-1, width)


@torch.no_grad()
def encoded_values(model, bits):
    model.eval()
    return torch.cat([model.encode(chunk) for chunk in bits.split(4096)], dim=0)


@torch.no_grad()
def pair_scores(values, pairs, metric):
    scores = []
    for chunk in np.array_split(pairs, max(1, (len(pairs) + 8191) // 8192)):
        if len(chunk):
            rows = torch.as_tensor(chunk[:, :2], dtype=torch.long, device=values.device)
            scores.append(distance(values[rows[:, 0]], values[rows[:, 1]], metric).cpu().numpy())
    return np.concatenate(scores) if scores else np.empty(0, dtype=np.float32)


def split_metrics(pairs, scores, groups, cap):
    if not len(pairs):
        return {"pairs": 0}
    finite = pairs[:, 2] >= 0
    labels = np.where(finite, pairs[:, 2], cap)
    error = np.abs(scores - labels)
    huber = np.where(error < 1, error * error / 2, error - .5)
    neg = ~finite
    hard = pairs[:, 3] == 2
    weighted = sum(float(huber[ids.numpy()].mean()) * weight
                   for ids, weight in strata(torch.as_tensor(pairs)))
    result = {"pairs": len(pairs), "finite_pairs": int(finite.sum()),
              "unreachable_pairs": int(neg.sum()), "hard_unreachable_pairs": int(hard.sum()),
              "distance_loss": weighted,
              "finite_mae": float(error[finite].mean()) if finite.any() else None,
              "unreachable_auc": auc(neg, scores),
              "unreachable_recall_at_cap_minus_half": float((scores[neg] >= cap - .5).mean()) if neg.any() else None,
              "hard_unreachable_recall": float((scores[hard] >= cap - .5).mean()) if hard.any() else None,
              "finite_false_unreachable_rate": float((scores[finite] >= cap - .5).mean()) if finite.any() else None}
    if len(groups):
        true = labels[groups]
        predicted = scores[groups]
        result["listwise_groups"] = len(groups)
        result["listwise_best_accuracy"] = float((true[np.arange(len(groups)), predicted.argmin(1)] ==
                                                   true.min(1)).mean())
        # The soft target uses the full ordering rather than a single positive.
        target = torch.softmax(torch.as_tensor(-true, dtype=torch.float32), dim=1)
        log_prob = torch.log_softmax(torch.as_tensor(-predicted, dtype=torch.float32), dim=1)
        result["listwise_loss"] = float(-(target * log_prob).sum(1).mean())
    else:
        result.update(listwise_groups=0, listwise_best_accuracy=None, listwise_loss=0.0)
    return result


@torch.no_grad()
def contrast_metrics(values, cases, metric):
    if not len(cases):
        return {"cases": 0}
    rows = torch.as_tensor(cases, dtype=torch.long, device=values.device)
    _, a, b, ga, gb = rows.T
    aa = distance(values[a], values[ga], metric).cpu().numpy()
    ba = distance(values[b], values[ga], metric).cpu().numpy()
    ab = distance(values[a], values[gb], metric).cpu().numpy()
    bb = distance(values[b], values[gb], metric).cpu().numpy()
    first, second = aa < ba, bb < ab
    return {"cases": len(cases), "goal_a_correct": float(first.mean()),
            "goal_b_correct": float(second.mean()),
            "both_goals_reverse_correct": float((first & second).mean()),
            "mean_correct_margin": float(np.mean(np.r_[ba - aa, ab - bb]))}


@torch.no_grad()
def goal_rollouts(model, tasks, states, metric, device):
    """Greedy legal-action rollout from certified reachable starts to goals."""
    if not len(tasks):
        return {"attempts": 0}
    cache = {}

    def encode(masks):
        missing = [mask for mask in dict.fromkeys(masks) if mask not in cache]
        if missing:
            from .model import board_bits
            for start in range(0, len(missing), 2048):
                chunk = missing[start:start + 2048]
                cache.update(zip(chunk, model.encode(board_bits(chunk).to(device)).split(1)))
        return torch.cat([cache[mask] for mask in masks], dim=0)

    solved = 0
    attempts = 0
    for source_id, goal_id in tasks:
        goal = int(states[goal_id])
        state = int(states[source_id])
        attempts += 1
        for _ in range(13):
            if state == goal:
                solved += 1
                break
            if _ == 12:
                break
            candidates = successors(state)
            if not candidates:
                break
            masks = [mask for _, mask in candidates]
            q = encode(masks)
            target = encode([goal]).expand_as(q)
            scores = distance(q, target, metric)
            state = masks[int(torch.argmin(scores))]
    return {"attempts": attempts, "reached_goal": solved,
            "goal_reach_rate": solved / attempts}
