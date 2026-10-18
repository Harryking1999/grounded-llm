"""Audit the unmodified greedy Q policy; no search is given to the agent."""
from collections import deque
import numpy as np
import torch


def resolve(pointer, horizon):
    """Exact eventual reachability and path lengths in a finite functional graph."""
    pointer = np.asarray(pointer, dtype=np.int64)
    if pointer[0] != 0 or np.any((pointer < 0) | (pointer >= len(pointer))):
        raise ValueError("Expected a finite policy with absorbing goal zero")
    jump = pointer.copy()
    length = np.ones(len(pointer), dtype=np.int64)
    length[0] = 0
    for _ in range(len(pointer).bit_length()):
        length, jump = length + length[jump], jump[jump]
    solved = jump == 0
    within = solved & (length <= horizon)
    return solved, within, np.where(solved, length, -1)


def cycle_nodes(pointer):
    """Peel incoming trees; residual non-goal vertices are actual cycle members."""
    degree = np.bincount(pointer, minlength=len(pointer))
    queue = deque(np.flatnonzero(degree == 0).tolist())
    while queue:
        state = queue.popleft()
        child = int(pointer[state])
        degree[child] -= 1
        if degree[child] == 0:
            queue.append(child)
    result = degree > 0
    result[0] = False
    return result


@torch.no_grad()
def audit(model, env, device, reference=None, cohort=None, horizon=62):
    model.eval()
    height = np.empty(env.size, dtype=np.float32)
    goal = model.encode(torch.as_tensor(env.features([0]), device=device))
    for begin in range(0, env.size, 8192):
        ids = np.arange(begin, min(begin + 8192, env.size))
        values = model.encode(torch.as_tensor(env.features(ids), device=device))
        height[ids] = model.distance(values, goal).cpu().numpy()
    valid = env.next >= 0
    scores = np.where(valid, height[np.maximum(env.next, 0)], np.inf)
    action = scores.argmin(-1)
    pointer = env.next[np.arange(env.size), action].astype(np.int64)
    pointer[0] = 0
    depth = env.distance(np.arange(env.size), 0)
    correct = depth[pointer] == depth - 1
    correct[0] = True
    good = valid & (depth[np.maximum(env.next, 0)] == depth[:, None] - 1)
    margin = np.where(valid & ~good, scores, np.inf).min(-1) - np.where(good, scores, np.inf).min(-1)
    solved, within, lengths = resolve(pointer, horizon)
    cyclic = cycle_nodes(pointer)
    ids = np.arange(env.size)
    pairs = np.flatnonzero((pointer[pointer] == ids) & (ids < pointer) & (ids != 0))
    report = {"nonterminal_states": env.size - 1, "horizon": horizon,
              "eventually_solved": int(solved[1:].sum()), "within_budget": int(within[1:].sum()),
              "within_budget_rate": float(within[1:].mean()), "shortest_action_rate": float(correct[1:].mean()),
              "wrong_decisions": int((~correct[1:]).sum()), "two_cycles": len(pairs),
              "cycle_nodes": int(cyclic.sum()), "other_cycle_nodes": int(cyclic.sum() - 2 * len(pairs)),
              "local_minima": int((scores[1:].min(-1) >= height[1:]).sum()),
              "restoration_mae": float(np.abs(height - depth).mean()),
              "cycle_examples": [{"states": [int(s), int(pointer[s])],
                  "depth": depth[[s, pointer[s]]].tolist(), "q_distance": height[[s, pointer[s]]].tolist(),
                  "exit_candidates": env.next[s][good[s]].tolist()} for s in pairs[:5]]}
    if reference is not None:
        report["previously_solved_lost"] = int((reference[1:] & ~within[1:]).sum())
        report["previously_failed_gained"] = int((~reference[1:] & within[1:]).sum())
    if cohort is not None:
        report["same1024"] = {"count": len(cohort), "solved": int(within[cohort].sum()),
                              "cycle_count": int((~solved[cohort]).sum()),
                              "mean_extra_steps_on_solved": float((lengths[cohort][within[cohort]] - depth[cohort][within[cohort]]).mean()) if within[cohort].any() else None}
    # Mine only training-goal errors, independent of validation/test rollouts.
    parents = np.flatnonzero(cyclic | ~correct)
    return report, {"pointer": pointer, "within": within, "correct": correct,
                    "margin": margin, "hard_parents": parents, "height": height}
