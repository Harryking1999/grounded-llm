"""Keep candidate IDs, actions and map slots aligned."""
from dataclasses import replace
from .memory import MapBatch

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
