"""hanoi evaluation cohorts and rollout orchestration."""
from grounded_llm.qmap_metrics import average_ranks, correlation, encode, pair_metrics, decisions, ranking_loss, rollout
from grounded_llm.qmap_sampling import task_indices

def tower_policy(model, env, device, budget_multiplier=2):
    """Audit the unchanged greedy policy from every state to every full tower."""
    import numpy as np
    q = encode(model, env, np.arange(env.size), device)
    towers = [0, (env.size - 1) // 2, env.size - 1]
    horizon = budget_multiplier * env.diameter
    reports, full = {}, 0
    for goal in towers:
        height = model.distance(q, q[goal]).detach().cpu().numpy()
        scores = np.where(env.next >= 0, height[np.maximum(env.next, 0)], np.inf)
        pointer = env.next[np.arange(env.size), scores.argmin(-1)].copy()
        pointer[goal] = goal
        reach, jump, remaining = np.arange(env.size), pointer.copy(), horizon
        while remaining:
            if remaining & 1:
                reach = jump[reach]
            jump, remaining = jump[jump], remaining >> 1
        solved = reach == goal
        solved[goal] = False
        count = int(solved.sum())
        full += count
        nonterminal = np.arange(env.size) != goal
        optimal = env.distance(pointer, goal) == env.distance(np.arange(env.size), goal) - 1
        cycles = (pointer[pointer] == np.arange(env.size)) & (np.arange(env.size) < pointer)
        reports[str(goal)] = {"within_budget": count, "rate": count / (env.size - 1),
                              "shortest_action_rate": float(optimal[nonterminal].mean()),
                              "two_cycles": int(cycles.sum()),
                              "tower_starts_solved": {str(s): bool(solved[s]) for s in towers if s != goal}}
    return {"nonterminal_tasks": 3 * (env.size - 1), "within_budget": full,
            "rate": full / (3 * (env.size - 1)), "horizon": horizon, "by_goal": reports}


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
    # Always expose the long standard tower tasks; random relation rollouts can
    # hide the critical bridge errors already observed in the six-disk map.
    import numpy as np
    towers = np.array([0, (env.size - 1) // 2, env.size - 1])
    tower_tasks = np.array([(s, g) for s in towers for g in towers if s != g])
    metrics["standard_towers"] = rollout(model, env, tower_tasks, device,
                                          config["evaluation"]["budget_multiplier"], return_tasks=True)
    if config["evaluation"].get("full_tower_policy", False):
        metrics["all_state_towers"] = tower_policy(model, env, device, config["evaluation"]["budget_multiplier"])
    return metrics
