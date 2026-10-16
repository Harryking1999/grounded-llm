"""Actual-successor Q slots shared with blocks; no learned action-vector table."""
import numpy as np
import torch

from .graph import GraphStep
from .memory import q_map_batch


def path_q_step(environment, qmap, current, goal, *, rng=None):
    if not np.array_equal(environment.adjacency, qmap.adjacency):
        raise ValueError("The frozen Q map belongs to a different graph")
    actions = sorted(environment.legal_actions(current), key=lambda a: int(environment.actions[a, 1]))
    if rng is not None:
        actions = rng.permutation(actions).tolist()
    destinations = tuple(int(environment.actions[a, 1]) for a in actions)
    vectors = qmap.encode([current, goal, *destinations])
    scores = qmap.distance(vectors, vectors[1])
    candidates = tuple(map(float, scores[2:]))
    minimum = min(candidates, default=float("inf"))
    minimal = tuple(i for i, value in enumerate(candidates, 1)
                    if np.isclose(value, minimum, rtol=1e-10, atol=1e-12))
    return GraphStep(current, goal, q_map_batch(vectors), tuple(actions), destinations,
                     float(scores[0]), candidates, minimal, current == goal)


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
