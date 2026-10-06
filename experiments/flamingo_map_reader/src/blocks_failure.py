"""Oracle-certified dead states; failed actions are context, never SFT labels."""

from dataclasses import replace
import re
import numpy as np

from .blocks import blocks_step
from .blocks_prompt import initial_prompt, turn_prompt
from .sft import Demonstration, SupervisedTurn, decision_text


NO_SOLUTION = ("No solution: none of the legal next moves can reach the goal.\n"
               "<action>none</action>\n<done/>")


def declares_no_solution(answer):
    return (answer.count("<done/>") == 1 and answer.count("<action>") == 1
            and re.search(r"<action>\s*none\s*</action>\s*<done/>", answer) is not None)


def failure_demonstration(qmap, record, oracle, *, direct=False, max_actions=32, reported=10):
    """Follow map greedily until the first certified dead state, without lookahead advice."""
    start, goal = int(record["start"]), int(record["goal"])
    rng = np.random.default_rng(record["sample_seed"])
    path, actions, turns = [start], [], []
    for _ in range(max_actions + 1):
        step = blocks_step(qmap, path[-1], goal, rng=rng)
        remaining = oracle.distance(step.current, goal)
        user = turn_prompt(step, actions)
        if not turns:
            user = initial_prompt(start, goal) + "\n\n" + user
        if remaining < 0:
            if direct:
                user = initial_prompt(step.current, goal) + "\n\n" + turn_prompt(step, [])
                turns, path = [], [step.current]
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


def direct_failure(demo):
    turn = demo.turns[-1]
    step = turn.step
    turn = replace(turn, user_text=initial_prompt(step.current, step.goal) + "\n\n" +
                   turn_prompt(step, []), executed_path=(step.current,))
    return Demonstration((turn,), (step.current,), False, True)
