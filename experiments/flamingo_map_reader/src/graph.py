"""Turn an existing graph Q/V and legal-action catalogue into map slots."""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

from experiments.external_map_interface.src.q_map import GraphQMap
from grounded_llm.tasks.path.environment import GraphEnvironment

from grounded_llm.tasks.path.qmap import GraphStep
from grounded_llm.core.reader.memory import CURRENT, GOAL, SUCCESSOR, MapBatch, MapTimeline
from grounded_llm.tasks.blocks.qmap import BlocksStep


def graph_step(
    environment: GraphEnvironment,
    qmap: GraphQMap,
    current: int,
    goal: int,
    *,
    executed_path: Sequence[int],
    rng: np.random.Generator | None = None,
) -> GraphStep:
    node_count = len(environment.adjacency)
    if not (0 <= current < node_count and 0 <= goal < node_count):
        raise ValueError("current and goal must be graph nodes")
    if qmap.q.shape[0] != node_count or qmap.v.shape[0] != len(environment.actions):
        raise ValueError("Q/V does not match the environment catalogue")
    if not executed_path or executed_path[-1] != current:
        raise ValueError("executed path must end at the actual current node")
    if len(set(executed_path)) != len(executed_path):
        raise ValueError("executed path may not revisit nodes")
    visited = set(executed_path)
    actions = sorted((action for action in environment.legal_actions(current)
                      if int(environment.actions[action, 1]) not in visited),
                     key=lambda action: int(environment.actions[action, 1]))
    if rng is not None:
        actions = rng.permutation(actions).tolist()
    current_vector = qmap.q[current]
    goal_vector = qmap.q[goal]
    successors = [current_vector + qmap.v[action] for action in actions]
    vectors = np.stack([current_vector, goal_vector, *successors])
    candidate_distances = tuple(float(np.linalg.norm(vector - goal_vector))
                                for vector in successors)
    minimum = min(candidate_distances, default=float("inf"))
    minimal_ids = tuple(index for index, distance in enumerate(candidate_distances, 1)
                        if np.isclose(distance, minimum, rtol=1e-10, atol=1e-12))
    count = len(actions)
    map_batch = MapBatch(
        vectors=torch.from_numpy(vectors.astype(np.float32)).unsqueeze(0),
        roles=torch.tensor([[CURRENT, GOAL, *([SUCCESSOR] * count)]]),
        candidate_ids=torch.tensor([[0, 0, *range(1, count + 1)]]),
        valid=torch.ones((1, count + 2), dtype=torch.bool),
    )
    return GraphStep(
        current=current,
        goal=goal,
        map_batch=map_batch,
        candidate_actions=tuple(actions),
        candidate_destinations=tuple(int(environment.actions[action, 1]) for action in actions),
        current_map_distance=float(np.linalg.norm(current_vector - goal_vector)),
        candidate_map_distances=candidate_distances,
        map_minimal_candidates=minimal_ids,
        done=current == goal,
    )


from grounded_llm.core.reader.batching import batch_maps, timeline_maps
