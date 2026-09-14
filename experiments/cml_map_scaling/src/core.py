"""CML local Q/V learning and fixed-graph diagnostics; no LLM or planner training.

Q and V are stored transposed (node/action x dimension) for contiguous updates.
Local updates implement equations 2-3 of Stoeckl et al. (2024), not full MSE
backpropagation. The random graph procedure follows the MIT-licensed GCML code.
"""
from collections import deque

import numpy as np


def random_graph(n, seed, min_edges=2, max_edges=5):
    rng = np.random.RandomState(seed)
    scores = rng.uniform(size=(n, n))
    scores += np.roll(np.eye(n), 1, axis=1)
    scores -= np.eye(n)
    low = int((min_edges - 1) / 2 + 0.5)
    high = int(max_edges / 2 + 0.5)
    ranks = rng.randint(low, high, n)
    thresholds = np.sort(scores, axis=1)[:, ::-1][np.arange(n), ranks]
    directed = scores > thresholds[:, None]
    return directed | directed.T


def action_catalog(adj):
    if not np.array_equal(adj, adj.T) or np.any(np.diag(adj)):
        raise ValueError("Expected a simple undirected graph")
    edges = [(int(i), int(j)) for i, j in zip(*np.nonzero(np.triu(adj, 1)))]
    actions = np.array([edge for i, j in edges for edge in [(i, j), (j, i)]])
    outgoing = [np.flatnonzero(actions[:, 0] == i) for i in range(len(adj))]
    if any(len(x) == 0 for x in outgoing):
        raise ValueError("Isolated node")
    return actions, outgoing


def shortest_distances(adj):
    n = len(adj)
    distances = np.full((n, n), -1, dtype=np.int32)
    neighbors = [np.flatnonzero(row) for row in adj]
    for start in range(n):
        distances[start, start] = 0
        queue = deque([start])
        while queue:
            node = queue.popleft()
            for other in neighbors[node]:
                if distances[start, other] < 0:
                    distances[start, other] = distances[start, node] + 1
                    queue.append(other)
    if np.any(distances < 0):
        raise ValueError("Disconnected graph")
    return distances


def sample_walks(actions, outgoing, count, steps, seed):
    rng = np.random.default_rng(seed)
    walks = np.empty((count, steps, 3), dtype=np.int32)
    for i in range(count):
        current = int(rng.integers(len(outgoing)))
        for t in range(steps):
            action = int(rng.choice(outgoing[current]))
            successor = int(actions[action, 1])
            walks[i, t] = current, action, successor
            current = successor
    return walks


def local_update(q, v, transition, eta_q, eta_v):
    source, action, destination = transition
    error = q[destination] - q[source] - v[action]
    q[destination] -= eta_q * error
    v[action] += eta_v * error


def train_epoch(q, v, walks, order, eta_q, eta_v):
    for i in order:
        for transition in walks[i]:
            local_update(q, v, transition, eta_q, eta_v)


def rankdata(values):
    """Average ranks, including tied shortest-path distances."""
    _, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    starts = np.cumsum(counts) - counts
    return (starts + (counts - 1) / 2.0)[inverse]


def spearman(x, y):
    rx, ry = rankdata(x), rankdata(y)
    rx -= rx.mean()
    ry -= ry.mean()
    denominator = np.linalg.norm(rx) * np.linalg.norm(ry)
    return float(rx @ ry / denominator) if denominator else None


def latent_distances(q):
    # Float64 arithmetic avoids cancellation in distances as the map contracts.
    q64 = q.astype(np.float64)
    norm = np.sum(q64 * q64, axis=1)
    squared = norm[:, None] + norm[None, :] - 2 * q64 @ q64.T
    return np.sqrt(np.maximum(0.0, squared))


def geometry(q, v, actions, graph_distances, visits):
    latent = latent_distances(q)
    pair_i, pair_j = np.triu_indices(len(q), 1)
    dg, ds = graph_distances[pair_i, pair_j], latent[pair_i, pair_j]
    root = np.sqrt(dg)
    c = float(root @ ds / (root @ root))
    fit_error = ds - c * root
    total_variance = float(np.sum((ds - ds.mean()) ** 2))
    error = q[actions[:, 1]] - q[actions[:, 0]] - v
    edge_error = np.mean(error.astype(np.float64) ** 2, axis=1)
    v_norm = np.linalg.norm(v, axis=1)
    buckets = []
    for distance in np.unique(dg):
        selected = ds[dg == distance]
        buckets.append({"graph_distance": int(distance), "pairs": len(selected),
                        "latent_mean": float(selected.mean()),
                        "latent_sd": float(selected.std(ddof=0))})
    return {
        "pairs": len(dg), "spearman": spearman(dg, ds),
        "latent_pair_mean": float(ds.mean()),
        "sqrt_fit_scale": c,
        "sqrt_fit_rmse": float(np.sqrt(np.mean(fit_error ** 2))),
        "sqrt_fit_r2": 1 - float(fit_error @ fit_error) / total_variance if total_variance else None,
        "transition_mse_all_actions": float(edge_error.mean()),
        "transition_mse_seen_actions": float(edge_error[visits > 0].mean()),
        "transition_mse_unseen_actions": float(edge_error[visits == 0].mean()) if np.any(visits == 0) else None,
        "v_norm_mean": float(v_norm.mean()),
        "v_norm_cv": float(v_norm.std() / v_norm.mean()),
        "distance_buckets": buckets,
    }, (pair_i, pair_j, dg, ds)


def cosine_policy(q, v, actions, outgoing):
    """Uses only learned vectors and legal action IDs, never graph distances."""
    n = len(q)
    policy = np.empty((n, n), dtype=np.int32)
    for source in range(n):
        candidates = outgoing[source]
        directions = v[candidates].astype(np.float64)
        norms = np.linalg.norm(directions, axis=1)
        directions /= np.maximum(norms[:, None], 1e-30)
        # Goal-difference norm is common to all candidates for a given goal.
        utilities = (q.astype(np.float64) - q[source]) @ directions.T
        policy[source] = actions[candidates[utilities.argmax(axis=1)], 1]
    return policy


def evaluate_policy(policy, graph_distances):
    """All ordered nontrivial pairs; an unreached deterministic walk has cycled."""
    n = len(policy)
    start, goal = np.where(~np.eye(n, dtype=bool))
    current = start.copy()
    steps = np.zeros(len(start), dtype=np.int32)
    active = np.ones(len(start), dtype=bool)
    for _ in range(n):
        ids = np.flatnonzero(active)
        if not len(ids):
            break
        current[ids] = policy[current[ids], goal[ids]]
        steps[ids] += 1
        active[ids] = current[ids] != goal[ids]
    reached = ~active
    shortest = graph_distances[start, goal]
    progress = graph_distances[policy[start, goal], goal] < shortest
    return {
        "pairs": len(start), "reached": int(reached.sum()),
        "reach_rate": float(reached.mean()),
        "cycle_or_step_cap": int(active.sum()),
        "shortest": int(np.sum(reached & (steps == shortest))),
        "shortest_rate_all_pairs": float(np.mean(reached & (steps == shortest))),
        "first_action_shortens_distance_rate": float(progress.mean()),
        "mean_steps_success_only": float(steps[reached].mean()) if reached.any() else None,
        "mean_stretch_success_only": float(np.mean(steps[reached] / shortest[reached])) if reached.any() else None,
    }, {"start": start, "goal": goal, "steps": steps, "reached": reached,
        "shortest": shortest, "policy": policy}
