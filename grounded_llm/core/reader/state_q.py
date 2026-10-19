"""Map slots from a frozen reader and the task's legal successors."""
import numpy as np
import torch
from .memory import q_map_batch

@torch.no_grad()
def puzzle_q_batch(reader, current, goal, *, rng=None):
    """Return the same MapBatch and its local-ID action names for Hanoi/cube."""
    env = reader.env
    env.validate(current)
    env.validate(goal)
    children = env.successors([current])[0]
    actions = np.flatnonzero(children >= 0)
    if rng is not None:
        actions = rng.permutation(actions)
    destinations = children[actions]
    vectors = reader.encode([current, goal, *map(int, destinations)]).cpu()
    return q_map_batch(vectors), tuple(env.actions[int(action)] for action in actions)
