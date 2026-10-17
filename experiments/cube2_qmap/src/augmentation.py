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
    pending = np.arange(len(result))
    accepted = 0
    for _ in range(max_attempts):
        if not len(pending):
            break
        original = states[indices[pending]]
        transform = env.sample_symmetry(rng, len(pending))
        candidate = env.symmetry(np.maximum(original, 0), transform[:, None])
        candidate = np.where(original >= 0, candidate, -1)
        keep = allowed(candidate)
        result[pending[keep]] = candidate[keep]
        accepted += int(keep.sum())
        pending = pending[~keep]
    if len(result) and not allowed(result).all():
        raise ValueError("Augmentation fallback was not valid training supervision")
    changed = np.any(result != states[indices], axis=1)
    return result, indices, {"requested": len(result), "accepted_draws": accepted,
                             "changed_rows": int(changed.sum()), "fallback_rows": len(pending)}
