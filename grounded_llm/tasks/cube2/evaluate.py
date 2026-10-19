"""cube2 evaluation cohorts and rollout orchestration."""
from grounded_llm.core.qmap.metrics import average_ranks, correlation, encode, pair_metrics, decisions, ranking_loss, rollout
from grounded_llm.core.qmap.sampling import task_indices

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
