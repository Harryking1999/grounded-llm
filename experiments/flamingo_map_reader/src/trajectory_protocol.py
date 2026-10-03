"""Full-trajectory targets and consistent action renumbering, independent of training."""

from dataclasses import replace
from itertools import permutations

import numpy as np

from .memory import MapBatch
from .sft import Demonstration, decision_text


def numbered_step(step, order):
    """Move action, successor, vector and distance together; never swap Q alone."""
    count = len(step.candidate_actions)
    if sorted(order) != list(range(count)):
        raise ValueError("Order must permute every legal candidate exactly once")
    slots = [0, 1, *(i + 2 for i in order)]
    batch = step.map_batch
    return replace(step,
        map_batch=MapBatch(batch.vectors[:, slots], batch.roles[:, slots],
                           batch.candidate_ids.clone(), batch.valid[:, slots]),
        candidate_actions=tuple(step.candidate_actions[i] for i in order),
        candidate_destinations=tuple(step.candidate_destinations[i] for i in order),
        candidate_map_distances=tuple(step.candidate_map_distances[i] for i in order),
        map_minimal_candidates=tuple(j + 1 for j, i in enumerate(order)
                                      if i + 1 in step.map_minimal_candidates))


def distance_answer(step, chosen, precision):
    answer = decision_text(step, chosen)
    numeric = [f"Current map distance to goal: {step.current_map_distance:.{precision}f}.",
        "Candidate map distances to goal: " + "; ".join(
            f"{i}: {distance:.{precision}f}" for i, distance in
            enumerate(step.candidate_map_distances, 1)) + "."]
    first, rest = answer.split("\n", 1)
    return "\n".join([first, *numeric, rest])


def numbering_plans(demo, count, seed):
    """Balanced per-turn permutations and distinct whole-trajectory variants.

    For k <= 3 cycle all k! permutations. Larger catalogues sample distinct
    permutations. The physical demonstration and its tie choices stay fixed.
    """
    rng = np.random.default_rng(seed)
    per_turn = []
    for turn in demo.turns:
        k = len(turn.step.candidate_actions) if not turn.step.done else 0
        if k <= 3:
            pool = list(permutations(range(k)))
            rng.shuffle(pool)
        else:
            pool = []
            while len(pool) < count:
                order = tuple(map(int, rng.permutation(k)))
                if order not in pool:
                    pool.append(order)
        per_turn.append([pool[i % len(pool)] for i in range(count)])
    plans = []
    for i in range(count):
        plan = tuple(orders[i] for orders in per_turn)
        if plan not in plans:
            plans.append(plan)
    # k=1 throughout legitimately admits only one physical numbering.
    return plans


def renumber_demonstration(demo, plan, task, precision):
    from .prompt import turn_prompt as graph_prompt
    from .blocks_prompt import turn_prompt as blocks_prompt
    if len(plan) != len(demo.turns):
        raise ValueError("A numbering plan must cover every turn")
    turns, actions = [], []
    for turn, order in zip(demo.turns, plan):
        # Terminal maps can still contain legal actions, but have no decision.
        step = turn.step if turn.step.done else numbered_step(turn.step, order)
        prompt = graph_prompt(step, turn.executed_path) if task == "graph" else blocks_prompt(step, actions)
        if not turns:
            prompt = turn.user_text.split("[Environment update]", 1)[0] + prompt
        chosen = None
        answer = turn.answer_text
        if not step.done:
            actual = turn.step.candidate_actions[turn.chosen_id - 1]
            chosen = step.candidate_actions.index(actual) + 1
            answer = distance_answer(step, chosen, precision)
            actions.append(actual)
        turns.append(replace(turn, step=step, chosen_id=chosen,
                             user_text=prompt, answer_text=answer))
    return Demonstration(tuple(turns), demo.executed_path, demo.success)
