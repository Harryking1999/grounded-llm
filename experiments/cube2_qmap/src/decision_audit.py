"""Separate one-step error, reference-path accumulation and greedy traps."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from experiments.puzzle_qmap.src.evaluate import encode
from experiments.puzzle_qmap.src.model import MapReader, distance
from experiments.puzzle_qmap.src.sampling import task_indices
from experiments.puzzle_qmap.src.tasks import make_environment


@torch.no_grad()
def audit(run, device):
    config = json.loads((run / "config.json").read_text())
    data = dict(np.load(run / "data.npz", allow_pickle=False))
    env = make_environment(config["task"], run / "oracle_depth.npy")
    if env.name != "cube2_qtm":
        raise ValueError("This audit uses the reversible six-action distance")
    reader = MapReader(env, run / "best.pt", device)

    def scores(states, goals):
        children = env.successors(states)
        values = distance(encode(reader.model, env, children, device),
                          encode(reader.model, env, goals, device)[:, None]).cpu().numpy()
        return children, values

    idx = task_indices(env, data, "test", config["evaluation"]["decision_count"], config)
    tasks = data["tasks_test"][idx]
    children, values = scores(tasks[:, 0], tasks[:, 1])
    depth = env.distance(tasks[:, 0], tasks[:, 1])
    truth = env.distance(children, tasks[:, 1, None])
    optimal = truth == depth[:, None] - 1
    correct = optimal[np.arange(len(tasks)), values.argmin(1)]
    margins = np.where(~optimal, values, np.inf).min(1) - np.where(optimal, values, np.inf).min(1)
    by_depth = {}
    for d in np.unique(depth):
        mask = depth == d
        finite = mask & np.isfinite(margins)
        by_depth[str(d)] = {"count": int(mask.sum()), "shortest_action_rate": float(correct[mask].mean()),
                            "uniform_random_shortest_rate": float(optimal[mask].mean()),
                            "mean_number_of_shortest_actions": float(optimal[mask].sum(1).mean()),
                            "all_actions_shortest_count": int(optimal[mask].all(1).sum()),
                            "mean_best_bad_minus_best_good_q_distance": float(margins[finite].mean()) if finite.any() else None}

    roll_idx = task_indices(env, data, "test", config["evaluation"]["rollout_count"], config)
    roll_tasks = data["tasks_test"][roll_idx]
    start_depth = env.distance(roll_tasks[:, 0], roll_tasks[:, 1])
    current, goals = roll_tasks[:, 0].copy(), roll_tasks[:, 1]
    all_gold_correct = np.ones(len(current), dtype=bool)
    gold_correct_count = np.zeros(len(current), dtype=int)
    gold_by_depth = {}
    # Use exact labels only to specify a fixed reference trajectory. Q is scored
    # independently at every reference state, even after it would have erred.
    for _ in range(env.diameter):
        ids = np.flatnonzero(current != goals)
        if not len(ids):
            break
        child, value = scores(current[ids], goals[ids])
        d = env.distance(current[ids], goals[ids])
        good = env.distance(child, goals[ids, None]) == d[:, None] - 1
        chosen_good = good[np.arange(len(ids)), value.argmin(1)]
        all_gold_correct[ids] &= chosen_good
        gold_correct_count[ids] += chosen_good
        for k in np.unique(d):
            row = gold_by_depth.setdefault(str(k), {"count": 0, "correct": 0})
            row["count"] += int((d == k).sum())
            row["correct"] += int(chosen_good[d == k].sum())
        current[ids] = child[np.arange(len(ids)), good.argmax(1)]
    for row in gold_by_depth.values():
        row["shortest_action_rate"] = row["correct"] / row["count"]

    current = roll_tasks[:, 0].copy()
    active = current != goals
    seen = [{int(s): 0} for s in current]
    first_error_step = np.zeros(len(current), dtype=int)
    error_source = np.full(len(current), -1, dtype=np.int64)
    reversed_next_step = np.zeros(len(current), dtype=bool)
    first_error_local_minimum = np.zeros(len(current), dtype=bool)
    cycle_lengths = np.zeros(len(current), dtype=int)
    for step in range(1, env.diameter * config["evaluation"]["budget_multiplier"] + 1):
        ids = np.flatnonzero(active)
        if not len(ids):
            break
        child, value = scores(current[ids], goals[ids])
        choice = value.argmin(1)
        chosen = child[np.arange(len(ids)), choice]
        before = env.distance(current[ids], goals[ids])
        after = env.distance(chosen, goals[ids])
        newly_wrong = (after != before - 1) & (first_error_step[ids] == 0)
        wrong_ids = ids[newly_wrong]
        first_error_step[wrong_ids] = step
        error_source[wrong_ids] = current[wrong_ids]
        parent_scores = reader.scores(current[ids], goals[ids])
        first_error_local_minimum[wrong_ids] = parent_scores[newly_wrong] <= value.min(1)[newly_wrong] + 1e-7
        immediately_after = first_error_step[ids] == step - 1
        reversed_next_step[ids[immediately_after]] = chosen[immediately_after] == error_source[ids[immediately_after]]
        for j, state in zip(ids, chosen):
            if state == goals[j]:
                active[j] = False
            elif int(state) in seen[j]:
                cycle_lengths[j] = step - seen[j][int(state)]
                active[j] = False
            else:
                seen[j][int(state)] = step
        current[ids] = chosen
    solved = current == goals
    wrong = first_error_step > 0
    source_counts = {str(k): int((first_error_step == k).sum()) for k in np.unique(first_error_step[wrong])}
    result = {"source_commit": torch.load(run / "best.pt", map_location="cpu", weights_only=False)["source_commit"],
              "parameter_updates": 0,
              "definitions": {"shortest_action": "d(next,goal)=d(current,goal)-1",
                              "one_step_goal": "next equals goal; only depth-one tasks admit it",
                              "reachable_after_action": "all legal successors remain reachable; this is a rule property",
                              "gold_reference": "one deterministic oracle shortest path, with Q choices scored but not executed"},
              "independent_test_decisions": {"count": len(tasks), "shortest_action_rate": float(correct.mean()),
                                             "uniform_random_shortest_rate": float(optimal.mean()),
                                             "by_distance": by_depth},
              "oracle_path_states": {"count": int(start_depth.sum()), "shortest_action_rate": float(gold_correct_count.sum() / start_depth.sum()),
                                     "paths_all_reference_choices_shortest": int(all_gold_correct.sum()),
                                     "by_remaining_distance": gold_by_depth,
                                     "by_start_distance": {str(d): {"count": int((start_depth == d).sum()),
                                         "mean_fraction_correct_on_reference_path": float((gold_correct_count[start_depth == d] / d).mean()),
                                         "paths_all_reference_choices_shortest": int(all_gold_correct[start_depth == d].sum())}
                                         for d in np.unique(start_depth)}},
              "own_rollout": {"count": len(roll_tasks), "solved": int(solved.sum()),
                              "all_steps_shortest": int((~wrong).sum()), "first_error_step": source_counts,
                              "first_errors": int(wrong.sum()),
                              "first_error_at_predicted_local_minimum": int(first_error_local_minimum.sum()),
                              "first_error_reversed_immediately": int(reversed_next_step[wrong].sum()),
                              "solved_despite_first_error": int((solved & wrong).sum()),
                              "cycle_length_histogram": {str(k): int((cycle_lengths == k).sum()) for k in np.unique(cycle_lengths[cycle_lengths > 0])}}}
    (run / "decision_audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    torch.set_num_threads(4)
    audit(args.run, args.device)
