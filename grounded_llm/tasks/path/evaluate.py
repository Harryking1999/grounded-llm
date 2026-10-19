"""Frozen direct-Q greedy routing; all neighbors remain legal."""
import numpy as np
import torch

from grounded_llm.core.qmap.model import soft_rank_loss


@torch.no_grad()
def distance_table(model, features):
    model.eval()
    q = model.encode(features)
    return model.distance(q[:, None], q[None]).cpu().numpy()


def relation_metrics(predicted, pairs):
    errors = np.abs(predicted[pairs[:, 0], pairs[:, 1]] - pairs[:, 2])
    return {"count": len(pairs), "mae": float(errors.mean()),
            "max_error": float(errors.max()),
            "huber": float(np.where(errors < 1, .5 * errors ** 2, errors - .5).mean())}


def candidate_scores(predicted, children, tasks):
    candidates = children[tasks[:, 0]]
    valid = candidates >= 0
    return np.where(valid, predicted[np.maximum(candidates, 0), tasks[:, 1, None]], np.inf), valid


def rank_objective(predicted, data, tasks, temperature):
    scores, valid = candidate_scores(predicted, data["children"], tasks)
    truth, _ = candidate_scores(data["distances"], data["children"], tasks)
    return float(soft_rank_loss(torch.from_numpy(np.where(valid, scores, 0)),
                               torch.from_numpy(np.where(valid, truth, 0)),
                               torch.from_numpy(valid), temperature))


def route_metrics(predicted, data, tasks, horizon):
    children, truth = data["children"], data["distances"]
    scores, _ = candidate_scores(predicted, children, tasks)
    chosen = children[tasks[:, 0], scores.argmin(1)]
    depth = truth[tasks[:, 0], tasks[:, 1]]
    optimal = truth[chosen, tasks[:, 1]] == depth - 1
    current, goals = tasks[:, 0].copy(), tasks[:, 1]
    active = current != goals
    cycles = np.zeros(len(tasks), dtype=bool)
    steps = np.zeros(len(tasks), dtype=np.int64)
    seen = np.zeros((len(tasks), len(children)), dtype=bool)
    seen[np.arange(len(tasks)), current] = True
    for _ in range(horizon):
        ids = np.flatnonzero(active)
        if not len(ids):
            break
        scores, _ = candidate_scores(predicted, children, np.column_stack((current[ids], goals[ids])))
        current[ids] = children[current[ids], scores.argmin(1)]
        steps[ids] += 1
        arrived = current[ids] == goals[ids]
        repeated = seen[ids, current[ids]] & ~arrived
        cycles[ids] = repeated
        active[ids[arrived | repeated]] = False
        seen[ids, current[ids]] = True
    solved = current == goals
    return {"count": len(tasks), "solved": int(solved.sum()), "success_rate": float(solved.mean()),
            "shortest_action_rate": float(optimal.mean()), "cycles": int(cycles.sum()),
            "budget_exhausted": int(active.sum()), "horizon": horizon,
            "mean_extra_steps_on_solved": float((steps[solved] - depth[solved]).mean()) if solved.any() else None,
            "by_distance": {str(d): {"count": int((depth == d).sum()),
                                     "success_rate": float(solved[depth == d].mean()),
                                     "shortest_action_rate": float(optimal[depth == d].mean())}
                            for d in np.unique(depth)}}


def final_metrics(predicted, data, config):
    horizon = int(data["distances"].max()) * config["evaluation"]["budget_multiplier"]
    source, goal = np.where(~np.eye(len(predicted), dtype=bool))
    result = {"all_routes": route_metrics(predicted, data, np.column_stack((source, goal)), horizon),
              "mean_direction_gap": float(np.abs(predicted - predicted.T).mean())}
    for name in ("train", "validation", "test"):
        tasks = data["train_monitor_tasks" if name == "train" else f"{name}_tasks"]
        result[name] = {"relations": relation_metrics(predicted, data[f"{name}_pairs"]),
                        "routes": route_metrics(predicted, data, tasks, horizon)}
    return result
