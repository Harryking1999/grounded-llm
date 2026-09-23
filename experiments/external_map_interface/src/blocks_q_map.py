"""Per-initial-board Q/V pilot using the existing Blocks action and CML update."""

import argparse
import json
from pathlib import Path

import numpy as np

from experiments.cml_map_scaling.src.core import local_update
from experiments.sol_dag_blocks.src.tasks import TASKS


def action_catalog(task):
    return [tuple(action[k] for k in ("shape_id", "row", "col"))
            for _, action in task.placements]


def collect_transitions(task, case, episodes: int, seed: int):
    """Random legal rollouts plus the case's construction witness to include goal."""
    initial = task.from_grid(case["grid"])
    actions = action_catalog(task)
    index = {action: i for i, action in enumerate(actions)}
    data = set()
    current = initial
    for action in case["construction_reference"]:
        following = task.apply(current, action, initial)
        data.add((current, index[action["shape_id"], action["row"], action["col"]], following))
        current = following
    if current != 0:
        raise ValueError("Construction reference does not reach empty board")
    rng = np.random.default_rng(seed)
    for _ in range(episodes):
        current = initial
        while current:
            legal = [(tile, i) for i, (tile, _) in enumerate(task.placements)
                     if tile & current == tile]
            if not legal:
                break
            tile, action_id = legal[int(rng.integers(len(legal)))]
            following = current ^ tile
            data.add((current, action_id, following))
            current = following
    states = sorted({initial, 0} | {s for s, _, _ in data} | {s for _, _, s in data})
    state_index = {state: i for i, state in enumerate(states)}
    rows = np.array(sorted((state_index[s], a, state_index[t]) for s, a, t in data), dtype=np.int32)
    return states, np.array(actions, dtype=np.int16), rows


def train_map(states, actions, transitions, config):
    """Same Q_destination + V_action local update as graph Step 1."""
    rng = np.random.default_rng(config["seed"])
    dim = config["state_dim"]
    q = rng.normal(0, config["q_init_std"], (len(states), dim)).astype(np.float32)
    v = rng.normal(0, config["v_init_std"], (len(actions), dim)).astype(np.float32)
    for _ in range(config["epochs"]):
        for i in rng.permutation(len(transitions)):
            local_update(q, v, transitions[i], config["eta_q"], config["eta_v"])
    residual = q[transitions[:, 2]] - q[transitions[:, 0]] - v[transitions[:, 1]]
    return q, v, float(np.mean(residual.astype(np.float64) ** 2))


def immediate_dead_diagnostic(task, case, states, q):
    """Known dead ends have no legal removal and are nonempty; no solver oracle."""
    goal = states.index(0)
    dead = [i for i, state in enumerate(states) if state and
            not any(tile & state == tile for tile, _ in task.placements)]
    known = []
    current = task.from_grid(case["grid"])
    known.append(current)
    for action in case["construction_reference"]:
        current = task.apply(current, action, known[0])
        known.append(current)
    state_index = {state: i for i, state in enumerate(states)}
    solvable = [{"remaining_cells": state.bit_count(),
                 "learned_distance": float(np.linalg.norm(q[state_index[state]] - q[goal]))}
                for state in known if state]
    violations = sum(
        float(np.linalg.norm(q[i] - q[goal])) <= item["learned_distance"]
        for i in dead for item in solvable if states[i].bit_count() < item["remaining_cells"]
    )
    comparable = sum(states[i].bit_count() < item["remaining_cells"]
                     for i in dead for item in solvable)
    return {"immediate_dead_count": len(dead),
            "immediate_dead": [{"remaining_cells": states[i].bit_count(),
                                "learned_distance": float(np.linalg.norm(q[i] - q[goal]))}
                               for i in dead],
            "known_solvable": solvable,
            "fewer_cells_dead_vs_more_cells_solvable_pairs": comparable,
            "pairs_where_dead_is_no_farther": violations}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    suite = json.loads(args.suite.read_text(encoding="utf-8"))
    case, = [c for c in suite["cases"] if c["id"] == args.case_id]
    if case["condition"] not in ("blocks8", "blocks12"):
        raise ValueError("Expected a Blocks case")
    task = TASKS[case["condition"]]
    states, actions, transitions = collect_transitions(task, case, config["episodes"], config["seed"])
    q, v, mse = train_map(states, actions, transitions, config)
    diagnostic = immediate_dead_diagnostic(task, case, states, q)
    args.out.mkdir(parents=True)
    np.savez_compressed(args.out / "map.npz", q=q, v=v, states=np.array([str(s) for s in states]),
                        actions=actions, transitions=transitions)
    (args.out / "summary.json").write_text(json.dumps({
        "case_id": args.case_id, "states": len(states), "shared_action_ids": len(actions),
        "sampled_unique_transitions": len(transitions), "transition_mse": mse,
        "diagnostic": diagnostic,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
