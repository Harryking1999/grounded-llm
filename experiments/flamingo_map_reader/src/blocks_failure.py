"""Exhausted failed rollouts; earlier actions are context, never SFT labels."""

import re
import numpy as np

from .blocks import blocks_step
from .blocks_prompt import initial_prompt, turn_prompt
from .sft import Demonstration, SupervisedTurn, decision_text


NO_SOLUTION = ("No solution: no legal moves remain and the goal has not been reached.\n"
               "<action>none</action>\n<done/>")


def declares_no_solution(answer):
    return (answer.count("<done/>") == 1 and answer.count("<action>") == 1
            and re.search(r"<action>\s*none\s*</action>\s*<done/>", answer) is not None)


def no_solution_terminal(step):
    return not step.done and not step.candidate_actions


def failure_demonstration(qmap, record, *, max_actions=32, reported=10):
    """Continue normal greedy moves until the legal catalogue is empty, or reach the goal."""
    start, goal = int(record["start"]), int(record["goal"])
    rng = np.random.default_rng(record["sample_seed"])
    path, actions, turns = [start], [], []
    for _ in range(max_actions + 1):
        step = blocks_step(qmap, path[-1], goal, rng=rng)
        user = turn_prompt(step, actions)
        if not turns:
            user = initial_prompt(start, goal) + "\n\n" + user
        if no_solution_terminal(step):
            turns.append(SupervisedTurn(user, NO_SOLUTION, step, tuple(path), None))
            return Demonstration(tuple(turns), tuple(path), False, True)
        if step.done or len(actions) == max_actions:
            return None
        chosen = int(rng.choice(step.map_minimal_candidates))
        turns.append(SupervisedTurn(user, decision_text(step, chosen, reported), step,
                                    tuple(path), chosen, supervise=False))
        action, destination = step.execute(chosen)
        actions.append(action)
        path.append(destination)
    raise AssertionError("unreachable")
