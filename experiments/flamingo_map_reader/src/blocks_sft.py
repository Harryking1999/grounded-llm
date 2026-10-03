"""Complete blocks trajectories supervised with frozen-map rankings."""

from .blocks import blocks_step
from .blocks_prompt import initial_prompt, turn_prompt, terminal_text
from .sft import Demonstration, SupervisedTurn, decision_text


def greedy_demonstration(qmap, start, goal, *, rng, max_actions=32, reported=None):
    first = initial_prompt(start, goal)
    path, actions, turns = [start], [], []
    for _ in range(max_actions + 1):
        step = blocks_step(qmap, path[-1], goal, rng=rng)
        update = turn_prompt(step, actions)
        user = first + "\n\n" + update if not turns else update
        if step.done:
            turns.append(SupervisedTurn(user, terminal_text(actions), step, tuple(path), None))
            return Demonstration(tuple(turns), tuple(path), True)
        if not step.candidate_actions or len(actions) == max_actions:
            return Demonstration(tuple(turns), tuple(path), False)
        chosen = int(rng.choice(step.map_minimal_candidates))
        turns.append(SupervisedTurn(user, decision_text(step, chosen, reported), step, tuple(path), chosen))
        action, actual = step.execute(chosen)
        actions.append(action)
        path.append(actual)
    raise AssertionError("unreachable loop exit")
