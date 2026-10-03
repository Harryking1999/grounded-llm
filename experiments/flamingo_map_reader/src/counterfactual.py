"""Readout-only map interventions; never use these steps as environment rollouts."""

from dataclasses import replace

import numpy as np
import torch

from .memory import MapBatch


def reorder_candidates(step, order):
    """Keep physical actions and Q vectors paired while assigning new local IDs."""
    count = len(step.candidate_actions)
    if sorted(order) != list(range(count)):
        raise ValueError("candidate order must be a permutation")
    slots = [0, 1, *(index + 2 for index in order)]
    source = step.map_batch
    ids = torch.tensor([[0, 0, *range(1, count + 1)]], dtype=torch.long,
                       device=source.candidate_ids.device)
    distances = tuple(step.candidate_map_distances[index] for index in order)
    minimum = min(distances)
    best = tuple(index for index, distance in enumerate(distances, 1)
                 if np.isclose(distance, minimum, rtol=1e-10, atol=1e-12))
    return replace(step,
        map_batch=MapBatch(source.vectors[:, slots], source.roles[:, slots], ids,
                           source.valid[:, slots]),
        candidate_actions=tuple(step.candidate_actions[index] for index in order),
        candidate_destinations=tuple(step.candidate_destinations[index] for index in order),
        candidate_map_distances=distances, map_minimal_candidates=best)


def renumbered_prompt(task, step, original_text):
    """Rebuild only the first-turn candidate listing after consistent renumbering."""
    from .blocks_prompt import turn_prompt as blocks_turn_prompt
    from .prompt import turn_prompt as graph_turn_prompt

    marker = "\n\n[Environment update]"
    if marker not in original_text:
        raise ValueError("first-turn prompt lacks its environment update")
    opening = original_text.rsplit(marker, 1)[0]
    update = (graph_turn_prompt(step, [step.current]) if task == "graph" else
              blocks_turn_prompt(step, []))
    return opening + "\n\n" + update


def swap_candidate_q(step, pair):
    """Swap two specified non-tied candidates; leave all text and addresses fixed."""
    left, right = pair
    count = len(step.candidate_actions)
    if left == right or not (1 <= left <= count and 1 <= right <= count):
        raise ValueError("Q swap needs two distinct legal candidate IDs")
    distances = list(step.candidate_map_distances)
    if np.isclose(distances[left - 1], distances[right - 1], rtol=1e-10, atol=1e-12):
        return None
    source = step.map_batch
    vectors = source.vectors.clone()
    vectors[:, left + 1] = source.vectors[:, right + 1]
    vectors[:, right + 1] = source.vectors[:, left + 1]
    distances[left - 1], distances[right - 1] = distances[right - 1], distances[left - 1]
    minimum = min(distances)
    minimal = tuple(i for i, distance in enumerate(distances, 1)
                    if np.isclose(distance, minimum, rtol=1e-10, atol=1e-12))
    return replace(step, map_batch=MapBatch(vectors, source.roles,
        source.candidate_ids, source.valid), candidate_map_distances=tuple(distances),
        map_minimal_candidates=minimal)


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
