"""Evaluation only: legal rollouts, latent prediction error, and Q geometry."""

import numpy as np
import torch


@torch.no_grad()
def transition_metrics(model, rows):
    if not len(rows):
        return {"count": 0, "mse": None, "relative_to_no_move_mse": None}
    predicted = model(rows[:, 0], rows[:, 1])
    target = model.q(rows[:, 2])
    mse = (predicted - target).square().mean().item()
    no_move = (model.q(rows[:, 0]) - target).square().mean().item()
    return {"count": len(rows), "mse": mse, "no_move_mse": no_move,
            "relative_to_no_move_mse": mse / no_move if no_move else None}


@torch.no_grad()
def q_scale(model):
    q = model.q.weight
    return {"centered_rms": (q - q.mean(dim=0)).square().mean().sqrt().item(),
            "rms": q.square().mean().sqrt().item()}


@torch.no_grad()
def rollout_diagnostic(model, task, case, states, train_rows, episodes, seed):
    """Actual boards choose legal actions; predicted Q is never reset to the table.

    References exist only for sampled boards. Unknown-board predictions have no
    target Q and cannot be reported as validated unseen-state representations.
    """
    rng = np.random.default_rng(seed)
    index = {state: i for i, state in enumerate(states)}
    device = model.q.weight.device
    initial = task.from_grid(case["grid"])
    trained_actions = set(map(int, train_rows[:, 1]))
    depth_metrics = {}
    for _ in range(episodes):
        actual = initial
        predicted = model.q.weight[index[actual]].clone()
        depth = 0
        while actual:
            legal = [(i, action) for i, (tile, action) in enumerate(task.placements)
                     if tile & actual == tile]
            if not legal:
                break
            action_id, action = legal[int(rng.integers(len(legal)))]
            predicted = model.predict(predicted, torch.tensor(action_id, device=device))
            actual = task.apply(actual, action, initial)
            depth += 1
            bucket = depth_metrics.setdefault(depth, {"steps": 0, "with_reference_q": 0,
                "untrained_action_steps": 0, "squared_error_sum": 0., "distance_error_sum": 0.})
            bucket["steps"] += 1
            bucket["untrained_action_steps"] += action_id not in trained_actions
            if actual in index:
                target = model.q.weight[index[actual]]
                goal = model.q.weight[index[0]]
                bucket["with_reference_q"] += 1
                bucket["squared_error_sum"] += (predicted - target).square().mean().item()
                bucket["distance_error_sum"] += abs(
                    torch.linalg.vector_norm(predicted - goal).item()
                    - torch.linalg.vector_norm(target - goal).item())
    for bucket in depth_metrics.values():
        count = bucket["with_reference_q"]
        bucket["mse_to_table_q"] = bucket.pop("squared_error_sum") / count if count else None
        bucket["goal_distance_mae_to_table"] = bucket.pop("distance_error_sum") / count if count else None
    return depth_metrics


@torch.no_grad()
def candidate_diagnostic(model, task, states, transitions):
    """Compare sampled siblings: certified immediate dead vs a witnessed goal path.

    A path in the sampled graph certifies solvability; absence of a sampled path
    does not certify a dead state. Labels here never enter the training loss.
    """
    goal = states.index(0)
    outgoing = {}
    for source, action, dest in transitions:
        outgoing.setdefault(int(source), []).append((int(action), int(dest)))
    solvable = {goal}
    for source in sorted(outgoing, key=lambda i: states[i].bit_count()):
        if any(dest in solvable for _, dest in outgoing[source]):
            solvable.add(source)
    dead = {i for i, state in enumerate(states) if state and
            not any(tile & state == tile for tile, _ in task.placements)}
    pairs = violations = 0
    distance_errors = []
    device = model.q.weight.device
    for source, edges in outgoing.items():
        actions = torch.tensor([a for a, _ in edges], device=device)
        scores = model.candidate_distances(model.q.weight[source], actions, model.q.weight[goal])
        scores = scores.cpu().numpy()
        target = model.q.weight[[d for _, d in edges]]
        reference = torch.linalg.vector_norm(target - model.q.weight[goal], dim=-1).cpu().numpy()
        distance_errors.extend(np.abs(scores - reference).tolist())
        for bad, (_, dest) in enumerate(edges):
            if dest in dead:
                for good, (_, other) in enumerate(edges):
                    if other in solvable:
                        pairs += 1
                        violations += int(scores[bad] <= scores[good])
    return {"sampled_states_with_witnessed_goal_path": len(solvable),
            "sampled_transitions_to_immediate_dead": sum(int(d) in dead for _, _, d in transitions),
            "same_source_dead_vs_solvable_candidate_pairs": pairs,
            "pairs_where_predicted_dead_is_no_farther": violations,
            "predicted_distance_mae_to_successor_table": float(np.mean(distance_errors))}
