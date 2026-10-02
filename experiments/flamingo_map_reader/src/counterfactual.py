"""Readout-only map interventions; never use these steps as environment rollouts."""

from dataclasses import replace

import numpy as np

from .memory import MapBatch


def swap_best_worst_q(step):
    """Keep text, physical actions, and IDs fixed while swapping two Q contents."""
    distances = list(step.candidate_map_distances)
    if len(step.map_minimal_candidates) != 1:
        return None
    best = step.map_minimal_candidates[0] - 1
    worst = int(np.argmax(distances))
    if best == worst or np.isclose(distances[best], distances[worst],
                                   rtol=1e-10, atol=1e-12):
        return None
    source = step.map_batch
    vectors = source.vectors.clone()
    vectors[:, best + 2] = source.vectors[:, worst + 2]
    vectors[:, worst + 2] = source.vectors[:, best + 2]
    distances[best], distances[worst] = distances[worst], distances[best]
    minimum = min(distances)
    best_ids = tuple(index for index, distance in enumerate(distances, 1)
                     if np.isclose(distance, minimum, rtol=1e-10, atol=1e-12))
    if best_ids != (worst + 1,):
        raise AssertionError("Q swap did not move a unique nearest candidate")
    return replace(step, map_batch=MapBatch(vectors, source.roles,
        source.candidate_ids, source.valid), candidate_map_distances=tuple(distances),
        map_minimal_candidates=best_ids)
