"""Pure state-Q path slots and physical action IDs."""
from dataclasses import dataclass
import numpy as np
from .environment import GraphEnvironment
from grounded_llm.core.reader.memory import MapBatch, q_map_batch

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
