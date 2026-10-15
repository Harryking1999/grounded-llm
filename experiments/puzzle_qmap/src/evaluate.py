"""Matched distance, full candidate ranking and rollout; no transition model."""
import numpy as np
import torch

from .model import distance, soft_rank_loss, edge_difference_loss
from .sampling import task_indices


def average_ranks(values):
    _, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    return (np.cumsum(counts) - (counts - 1) / 2)[inverse]


def correlation(x, y):
    x, y = average_ranks(x), average_ranks(y)
    return float(np.corrcoef(x, y)[0, 1]) if np.std(x) and np.std(y) else 0.0


@torch.no_grad()
def encode(model, env, ids, device):
    shape = np.asarray(ids).shape
    flat = np.asarray(ids).ravel()
    q = torch.cat([model.encode(torch.as_tensor(env.features(flat[i:i + 8192]), device=device))
                   for i in range(0, len(flat), 8192)])
    return q.reshape(*shape, -1)


@torch.no_grad()
def pair_metrics(model, env, rows, device):
    predictions = []
    for chunk in np.array_split(rows, max(1, (len(rows) + 4095) // 4096)):
        a, b = encode(model, env, chunk[:, 0], device), encode(model, env, chunk[:, 1], device)
        predictions.extend(distance(a, b).cpu().tolist())
    predictions, truth = np.asarray(predictions), rows[:, 2]
    error = np.abs(predictions - truth)
    return {"count": len(rows), "huber": float(np.where(error < 1, .5 * error**2, error - .5).mean()),
            "mae": float(error.mean()),
            "spearman": correlation(predictions, truth),
            "by_distance": {str(d): {"count": int((truth == d).sum()), "predicted_mean": float(predictions[truth == d].mean()),
                                     "mae": float(np.abs(predictions[truth == d] - d).mean())} for d in np.unique(truth)}}


@torch.no_grad()
def decisions(model, env, tasks, device):
    children = env.successors(tasks[:, 0])
    valid = children >= 0
    child_q = encode(model, env, np.maximum(children, 0), device)
    goal_q = encode(model, env, tasks[:, 1], device)
    q_scores = distance(child_q, goal_q[:, None]).cpu().numpy()
    baseline = env.baseline(np.maximum(children, 0), tasks[:, 1, None]).astype(float)
    truth = env.distance(np.maximum(children, 0), tasks[:, 1, None])
    parent_distance = env.distance(tasks[:, 0], tasks[:, 1])
    results = {}
    for name, scores in (("q", q_scores), ("hamming", baseline), ("oracle", truth.astype(float))):
        scores = np.where(valid, scores, np.inf)
        chosen = scores.argmin(axis=1)
        optimal = truth == parent_distance[:, None] - 1
        minimum = scores.min(axis=1, keepdims=True)
        tied = np.isclose(scores, minimum, rtol=0, atol=1e-7) & valid
        near = truth[:, :, None] < truth[:, None, :]
        ordered = near & valid[:, :, None] & valid[:, None, :]
        correct = scores[:, :, None] < scores[:, None, :] - 1e-7
        results[name] = {"count": len(tasks), "shortest_action_rate": float(optimal[np.arange(len(tasks)), chosen].mean()),
                         "uniform_tie_shortest_rate": float(((tied & optimal).sum(axis=1) / tied.sum(axis=1)).mean()),
                         "strict_candidate_order_accuracy": float(correct[ordered].mean()) if ordered.any() else None}
    return results


@torch.no_grad()
def ranking_loss(model, env, tasks, truth, device, temperature):
    scores, targets, masks = [], [], []
    for begin in range(0, len(tasks), 1024):
        chunk = tasks[begin:begin + 1024]
        children = env.successors(chunk[:, 0])
        scores.append(distance(encode(model, env, np.maximum(children, 0), device),
                               encode(model, env, chunk[:, 1], device)[:, None]))
        targets.append(torch.as_tensor(truth[begin:begin + len(chunk)], device=device))
        masks.append(torch.as_tensor(children >= 0, device=device))
    return float(soft_rank_loss(torch.cat(scores), torch.cat(targets), torch.cat(masks), temperature))


@torch.no_grad()
def progress_loss(model, env, tasks, truth, device):
    losses, counts = [], []
    for begin in range(0, len(tasks), 1024):
        chunk = tasks[begin:begin + 1024]
        children = env.successors(chunk[:, 0])
        goal = encode(model, env, chunk[:, 1], device)
        child_scores = distance(encode(model, env, np.maximum(children, 0), device), goal[:, None])
        parent_scores = distance(encode(model, env, chunk[:, 0], device), goal)
        valid = torch.as_tensor(children >= 0, device=device)
        loss = edge_difference_loss(parent_scores, child_scores,
                                    torch.as_tensor(env.distance(chunk[:, 0], chunk[:, 1]), device=device),
                                    torch.as_tensor(truth[begin:begin + len(chunk)], device=device), valid)
        counts.append(int(valid.sum()))
        losses.append(float(loss))
    return float(np.average(losses, weights=counts))


@torch.no_grad()
def rollout(model, env, tasks, device, budget_multiplier, horizon=None, return_tasks=False):
    results = {}
    exact = env.distance(tasks[:, 0], tasks[:, 1])
    # Same distance-independent horizon for all policies; no oracle-fed budget.
    budget = int(env.diameter * budget_multiplier) if horizon is None else int(horizon)
    for policy in ("q", "hamming", "oracle"):
        current, goal = tasks[:, 0].copy(), tasks[:, 1]
        steps = np.zeros(len(tasks), dtype=int)
        active = current != goal
        cycles = np.zeros(len(tasks), dtype=bool)
        seen = [{int(s)} for s in current]
        for _ in range(budget):
            ids = np.flatnonzero(active)
            if not len(ids):
                break
            children = env.successors(current[ids])
            valid = children >= 0
            safe = np.maximum(children, 0)
            if policy == "q":
                scores = distance(encode(model, env, safe, device), encode(model, env, goal[ids], device)[:, None]).cpu().numpy()
            elif policy == "hamming":
                scores = env.baseline(safe, goal[ids, None]).astype(float)
            else:
                scores = env.distance(safe, goal[ids, None]).astype(float)
            scores[~valid] = np.inf
            choice = scores.argmin(axis=1)
            current[ids] = children[np.arange(len(ids)), choice]
            steps[ids] += 1
            for j in ids:
                if current[j] == goal[j]:
                    active[j] = False
                elif int(current[j]) in seen[j]:
                    cycles[j], active[j] = True, False
                else:
                    seen[j].add(int(current[j]))
        solved = current == goal
        results[policy] = {"count": len(tasks), "solved": int(solved.sum()), "success_rate": float(solved.mean()),
                           "cycle_rate": float(cycles.mean()), "budget_exhausted": int(active.sum()), "horizon": budget,
                           "mean_extra_steps_on_solved": float((steps[solved] - exact[solved]).mean()) if solved.any() else None,
                           "illegal_action_rate": 0.0,
                           "by_start_distance": {str(d): {"count": int((exact == d).sum()), "success_rate": float(solved[exact == d].mean())}
                                                 for d in np.unique(exact)}}
        if return_tasks:
            results[policy]["per_task"] = {"solved": solved.tolist(), "steps": steps.tolist(), "cycles": cycles.tolist()}
    return results


def evaluate(model, env, data, config, device):
    model.eval()
    metrics = {}
    splits = ["validation", "test"] + (["reference_test"] if "tasks_reference_test" in data else [])
    for split in splits:
        idx = task_indices(env, data, split, config["evaluation"]["decision_count"], config)
        tasks = data[f"tasks_{split}"][idx]
        roll_idx = task_indices(env, data, split, config["evaluation"]["rollout_count"], config)
        metrics[split] = {"pairs": pair_metrics(model, env, data[f"pairs_{split}"], device),
                          "decisions": decisions(model, env, tasks, device),
                          "soft_rank_loss": ranking_loss(model, env, tasks, data[f"child_distances_{split}"][idx],
                                                         device, config["training"]["rank_temperature"]),
                          "rollout": rollout(model, env, data[f"tasks_{split}"][roll_idx], device,
                                             config["evaluation"]["budget_multiplier"])}
    return metrics
