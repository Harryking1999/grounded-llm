"""Turn an existing graph Q/V and legal-action catalogue into map slots."""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

from experiments.external_map_interface.src.q_map import GraphQMap
from experiments.external_map_interface.src.transitions import GraphEnvironment

from .memory import CURRENT, GOAL, SUCCESSOR, MapBatch


@dataclass(frozen=True)
class GraphStep:
    current: int
    goal: int
    map_batch: MapBatch
    candidate_actions: tuple[int, ...]  # local ID i maps to candidate_actions[i-1]
    candidate_destinations: tuple[int, ...]
    current_map_distance: float
    candidate_map_distances: tuple[float, ...]
    map_minimal_candidates: tuple[int, ...]
    done: bool  # exact environment goal, never inferred from Q coordinates

    def execute(self, environment: GraphEnvironment, local_id: int) -> tuple[int, int]:
        if not isinstance(local_id, int) or not 1 <= local_id <= len(self.candidate_actions):
            raise ValueError("candidate ID is not present in this step")
        action_id = self.candidate_actions[local_id - 1]
        return action_id, environment.execute(self.current, action_id)


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


def batch_maps(steps: list[GraphStep]) -> MapBatch:
    if not steps:
        raise ValueError("cannot batch zero graph steps")
    return MapBatch(
        vectors=pad_sequence([s.map_batch.vectors[0] for s in steps], batch_first=True),
        roles=pad_sequence([s.map_batch.roles[0] for s in steps], batch_first=True),
        candidate_ids=pad_sequence([s.map_batch.candidate_ids[0] for s in steps],
                                   batch_first=True),
        valid=pad_sequence([s.map_batch.valid[0] for s in steps], batch_first=True),
    )
