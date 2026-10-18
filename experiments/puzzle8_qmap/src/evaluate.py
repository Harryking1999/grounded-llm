"""Restoration is primary; heldout intermediate goals measure relation completion."""
from grounded_llm.qmap_metrics import pair_metrics, decisions, ranking_loss, rollout
from grounded_llm.qmap_sampling import task_indices


def evaluate(model, env, data, config, device):
    result = {}
    for split in ("validation", "test"):
        idx = task_indices(env, data, split, config["evaluation"]["decision_count"], config)
        tasks = data[f"tasks_{split}"][idx]
        ridx = task_indices(env, data, split, config["evaluation"]["rollout_count"], config)
        result[split] = {"pairs": pair_metrics(model, env, data[f"pairs_{split}"], device),
                         "decisions": decisions(model, env, tasks, device),
                         "rollout": rollout(model, env, data[f"tasks_{split}"][ridx], device,
                                            config["evaluation"]["budget_multiplier"])}
    result["restoration_seen_states"] = rollout(model, env, data["tasks_restoration"], device,
                                               config["evaluation"]["budget_multiplier"])
    return result
