"""hanoi evaluation cohorts and rollout orchestration."""
from grounded_llm.qmap_metrics import average_ranks, correlation, encode, pair_metrics, decisions, ranking_loss, rollout
from grounded_llm.qmap_sampling import task_indices

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
    return metrics
