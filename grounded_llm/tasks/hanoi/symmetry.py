"""Joint isometric augmentation with explicit split rejection, never a Q loss."""
import numpy as np


def augment(env, states, allowed, rng, copies, max_attempts):
    """Transform every state in one row together, retaining padding (-1).

    A failed split-constrained draw falls back to the original training row.
    The returned row indices keep labels aligned even when a row falls back.
    """
    if copies < 0 or max_attempts < 1:
        raise ValueError("copies must be nonnegative; max_attempts must be positive")
    indices = np.tile(np.arange(len(states)), copies)
    result = states[indices].copy()
    if copies:
        if copies != env.symmetry_count:
            raise ValueError("Hanoi augmentation enumerates all six peg permutations")
        transform = np.repeat(np.arange(env.symmetry_count), len(states))
        result = env.symmetry(np.maximum(result, 0), transform[:, None])
        result = np.where(states[indices] >= 0, result, -1)
        if not allowed(result).all():
            raise ValueError("Hanoi split must keep entire symmetry orbits together")
        return result, indices, {"requested": len(result), "accepted_draws": len(result),
                                 "changed_rows": int(np.any(result != states[indices], axis=1).sum()),
                                 "fallback_rows": 0}
    return result, indices, {"requested": 0, "accepted_draws": 0, "changed_rows": 0, "fallback_rows": 0}
