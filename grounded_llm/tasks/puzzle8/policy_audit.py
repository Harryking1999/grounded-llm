"""Audit the unmodified greedy Q policy; no search is given to the agent."""
from collections import deque
import numpy as np
import torch


def resolve(pointer, horizon, goal=0):
    """Exact eventual reachability and path lengths in a finite functional graph."""
    pointer = np.asarray(pointer, dtype=np.int64)
    if pointer[goal] != goal or np.any((pointer < 0) | (pointer >= len(pointer))):
        raise ValueError("Expected a finite policy with the specified absorbing goal")
    jump = pointer.copy()
    length = np.ones(len(pointer), dtype=np.int64)
    length[goal] = 0
    for _ in range(len(pointer).bit_length()):
        length, jump = length + length[jump], jump[jump]
    solved = jump == goal
    within = solved & (length <= horizon)
    return solved, within, np.where(solved, length, -1)


def cycle_nodes(pointer, goal=0):
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
    result[goal] = False
    return result


@torch.no_grad()
def audit(model, env, device, reference=None, cohort=None, horizon=62, goal=0):
    model.eval()
    height = np.empty(env.size, dtype=np.float32)
    target = model.encode(torch.as_tensor(env.features([goal]), device=device))
    for begin in range(0, env.size, 8192):
        ids = np.arange(begin, min(begin + 8192, env.size))
        values = model.encode(torch.as_tensor(env.features(ids), device=device))
        height[ids] = model.distance(values, target).cpu().numpy()
    valid = env.next >= 0
    scores = np.where(valid, height[np.maximum(env.next, 0)], np.inf)
    action = scores.argmin(-1)
    pointer = env.next[np.arange(env.size), action].astype(np.int64)
    pointer[goal] = goal
    depth = env.distance(np.arange(env.size), goal)
    correct = depth[pointer] == depth - 1
    correct[goal] = True
    good = valid & (depth[np.maximum(env.next, 0)] == depth[:, None] - 1)
    margin = np.where(valid & ~good, scores, np.inf).min(-1) - np.where(good, scores, np.inf).min(-1)
    solved, within, lengths = resolve(pointer, horizon, goal)
    cyclic = cycle_nodes(pointer, goal)
    ids = np.arange(env.size)
    pairs = np.flatnonzero((pointer[pointer] == ids) & (ids < pointer) & (ids != goal))
    nonterminal = ids != goal
    report = {"nonterminal_states": env.size - 1, "horizon": horizon,
              "eventually_solved": int(solved[nonterminal].sum()), "within_budget": int(within[nonterminal].sum()),
              "within_budget_rate": float(within[nonterminal].mean()), "shortest_action_rate": float(correct[nonterminal].mean()),
              "wrong_decisions": int((~correct[nonterminal]).sum()), "two_cycles": len(pairs),
              "cycle_nodes": int(cyclic.sum()), "other_cycle_nodes": int(cyclic.sum() - 2 * len(pairs)),
              "local_minima": int((scores[nonterminal].min(-1) >= height[nonterminal]).sum()),
              "restoration_mae": float(np.abs(height - depth).mean()),
              "cycle_examples": [{"states": [int(s), int(pointer[s])],
                  "depth": depth[[s, pointer[s]]].tolist(), "q_distance": height[[s, pointer[s]]].tolist(),
                  "exit_candidates": env.next[s][good[s]].tolist()} for s in pairs[:5]]}
    if reference is not None:
        report["previously_solved_lost"] = int((reference[nonterminal] & ~within[nonterminal]).sum())
        report["previously_failed_gained"] = int((~reference[nonterminal] & within[nonterminal]).sum())
    if cohort is not None:
        report["same1024"] = {"count": len(cohort), "solved": int(within[cohort].sum()),
                              "cycle_count": int((~solved[cohort]).sum()),
                              "mean_extra_steps_on_solved": float((lengths[cohort][within[cohort]] - depth[cohort][within[cohort]]).mean()) if within[cohort].any() else None}
    # Mine only training-goal errors, independent of validation/test rollouts.
    parents = np.flatnonzero(cyclic | ~correct)
    return report, {"pointer": pointer, "within": within, "correct": correct,
                    "margin": margin, "hard_parents": parents, "height": height}
