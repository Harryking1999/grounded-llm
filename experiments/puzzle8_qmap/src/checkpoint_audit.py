"""Read-only matched audit of a Puzzle8 checkpoint and its preserved baseline.

This never selects actions with oracle distances and never writes a checkpoint.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from grounded_llm.core.qmap.metrics import decisions, pair_metrics, rollout
from grounded_llm.core.qmap.sampling import task_indices
from grounded_llm.tasks.puzzle8.environment import Puzzle8
from grounded_llm.tasks.puzzle8.model import MODEL_TYPE, QMap
from grounded_llm.tasks.puzzle8.policy_audit import audit


def same_tasks(before, after):
    before, after = np.asarray(before, dtype=bool), np.asarray(after, dtype=bool)
    if before.shape != after.shape:
        raise ValueError("Matched outcomes must have the same shape")
    lost, gained = int((before & ~after).sum()), int((~before & after).sum())
    return {"count": int(before.size), "baseline_solved": int(before.sum()),
            "candidate_solved": int(after.sum()), "previously_solved_lost": lost,
            "previously_failed_gained": gained, "net_gain": gained - lost}


def cycle_ranking(env, state):
    """All cycle stars and exact basin sizes, including errors outside short cycles."""
    pointer, height = state["pointer"], state["height"]
    from grounded_llm.tasks.puzzle8.policy_audit import cycle_nodes
    cyclic = cycle_nodes(pointer)
    representatives = np.full(env.size, -1, dtype=np.int64)
    for start in np.flatnonzero(cyclic):
        if representatives[start] >= 0:
            continue
        members, current = [], int(start)
        while not members or current != start:
            members.append(current)
            current = int(pointer[current])
        representatives[members] = min(members)
    destinations = np.arange(env.size, dtype=np.int64)
    jump = pointer.copy()
    for _ in range(env.size.bit_length()):
        destinations = jump[destinations]
        jump = jump[jump]
    failed = ~state["within"]
    reached = representatives[destinations]
    valid = failed & (reached >= 0)
    counts = np.bincount(reached[valid], minlength=env.size)
    depth = env.distance(np.arange(env.size), 0)
    rows = []
    for representative in np.flatnonzero(counts):
        members = np.flatnonzero(representatives == representative)
        stars = []
        for parent in members:
            choices = np.flatnonzero(env.next[parent] >= 0)
            children = env.next[parent, choices]
            stars.append({"parent": int(parent), "true_distance": int(depth[parent]),
                          "q_distance": float(height[parent]), "chosen": int(pointer[parent]),
                          "candidates": [{"action": env.actions[a], "state": int(child),
                                          "true_distance": int(depth[child]),
                                          "q_distance": float(height[child])}
                                         for a, child in zip(choices, children)]})
        rows.append({"basin_states": int(counts[representative]), "stars": stars})
    return sorted(rows, key=lambda row: row["basin_states"], reverse=True)


def exposure(coverage, data_summary):
    result = dict(coverage)
    for prefix, draws, pool in (("pair", "all_pairs_draws", "available_joint_pairs"),
                                ("decision", "all_decision_draws", "available_joint_decisions")):
        if draws in coverage and pool in data_summary:
            size = data_summary[pool]
            # A permutation stream visits every ID once before repeating. This
            # says nothing about overlap with other components or model mastery.
            distinct = min(int(coverage[draws]), int(size))
            result[f"broad_{prefix}_distinct"] = distinct
            result[f"broad_{prefix}_distinct_fraction"] = distinct / size
    result["scope"] = "Distinct IDs in each broad permutation stream; excludes replay overlap and does not imply learned relations"
    return result


def load_model(saved, device):
    if saved.get("task") != "puzzle8" or saved.get("model_type") != MODEL_TYPE or "repair_config" in saved:
        raise ValueError("Expected a Puzzle8 checkpoint trained with the original objective")
    model = QMap(**saved["config"]["model"]).to(device)
    model.load_state_dict(saved["model"])
    return model.eval().requires_grad_(False)


def compare(checkpoint, baseline_checkpoint, prepared_run, device="cpu"):
    saved = torch.load(checkpoint, map_location=device, weights_only=False)
    baseline = torch.load(baseline_checkpoint, map_location=device, weights_only=False)
    if any(saved["config"][key] != baseline["config"][key] for key in ("task", "rules", "data", "model")):
        raise ValueError("Baseline and candidate must share the state, rule, model and fixed-cohort contract")
    config = saved["config"]
    env = Puzzle8()
    with np.load(prepared_run / "oracle.npz") as oracle:
        env.tables = oracle["distances"]
        env.goal_columns = {int(g): i for i, g in enumerate(oracle["goals"])}
    with np.load(prepared_run / "data.npz") as packed:
        data = {key: packed[key] for key in packed.files if not key.endswith("_train")}
    model, old_model = load_model(saved, device), load_model(baseline, device)
    cohort = data["tasks_restoration"][:, 0]
    old_policy, old_state = audit(old_model, env, device, cohort=cohort)
    policy, state = audit(model, env, device, reference=old_state["within"], cohort=cohort)
    result = {"checkpoint": str(checkpoint), "baseline_checkpoint": str(baseline_checkpoint),
              "continuation_step": saved.get("continuation_step"), "checkpoint_step": saved["step"],
              "source_commit": saved["source_commit"],
              "scope": "Read-only same-task audit; exact oracle supplies labels only; unchanged greedy Q policy",
              "baseline_restoration": old_policy, "candidate_restoration": policy,
              "restoration_same_tasks": same_tasks(old_state["within"][1:], state["within"][1:]),
              "cycle_rankings": cycle_ranking(env, state)}
    for split in ("validation", "test"):
        indices = task_indices(env, data, split, config["evaluation"]["rollout_count"], config)
        tasks = data[f"tasks_{split}"][indices]
        old_rollout = rollout(old_model, env, tasks, device, config["evaluation"]["budget_multiplier"], return_tasks=True)
        new_rollout = rollout(model, env, tasks, device, config["evaluation"]["budget_multiplier"], return_tasks=True)
        result[split] = {"tasks": tasks.tolist(),
                         "same_tasks": same_tasks(old_rollout["q"]["per_task"]["solved"], new_rollout["q"]["per_task"]["solved"]),
                         "baseline": old_rollout["q"], "candidate": new_rollout["q"],
                         "oracle_solved": new_rollout["oracle"]["solved"],
                         "pairs": pair_metrics(model, env, data[f"pairs_{split}"], device)}
        decision_indices = task_indices(env, data, split, config["evaluation"]["decision_count"], config)
        decision_tasks = data[f"tasks_{split}"][decision_indices]
        truth = env.distance(decision_tasks[:, 0], decision_tasks[:, 1])
        result[split]["decisions_by_distance"] = {str(int(distance)): decisions(model, env, decision_tasks[truth == distance], device)["q"] for distance in np.unique(truth)}
    graph_summary = checkpoint.parent / "graph_data_summary.json"
    result["exposure"] = exposure(saved.get("coverage", {}), json.loads(graph_summary.read_text()) if graph_summary.exists() else {})
    return result


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--baseline-checkpoint", type=Path, required=True)
    parser.add_argument("--prepared-run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    torch.set_num_threads(1)
    result = compare(args.checkpoint, args.baseline_checkpoint, args.prepared_run, args.device)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(args.out), "step": result["continuation_step"],
                      "restoration": result["restoration_same_tasks"],
                      "intermediate": result["test"]["same_tasks"]}))


if __name__ == "__main__":
    main()
