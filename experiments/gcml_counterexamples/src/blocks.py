"""Exact GCML tiling rules and offline diagnostics; never a model-side tool."""
from functools import lru_cache
import json
from pathlib import Path

CONFIG = json.loads((Path(__file__).parents[1] / "configs/pilot.json").read_text())
SIZE = CONFIG["blocks"]["grid_size"]
SHAPES = tuple(tuple((r, c) for r, row in enumerate(shape.split("/"))
                     for c, x in enumerate(row) if x == "1")
               for shape in CONFIG["blocks"]["shape_rows"])
PLACEMENTS = []
BY_CELL = [[] for _ in range(SIZE * SIZE)]
BY_ACTION = {}
for shape_id, offsets in enumerate(SHAPES):
    for row in range(SIZE - max(r for r, _ in offsets)):
        for col in range(SIZE - max(c for _, c in offsets)):
            cells = tuple((row + r) * SIZE + col + c for r, c in offsets)
            mask = sum(1 << cell for cell in cells)
            action = {"shape_id": shape_id, "row": row, "col": col}
            index = len(PLACEMENTS)
            PLACEMENTS.append((mask, action))
            BY_ACTION[(shape_id, row, col)] = index
            for cell in cells:
                BY_CELL[cell].append(index)


def cells(mask):
    while mask:
        bit = mask & -mask
        yield bit.bit_length() - 1
        mask ^= bit


def from_grid(grid):
    rows = grid.split("/") if isinstance(grid, str) else grid
    if len(rows) != SIZE or any(len(row) != SIZE or set(row) - {"0", "1"} for row in rows):
        raise ValueError("Expected a complete 10x10 binary grid")
    return sum(1 << i for i, x in enumerate("".join(rows)) if x == "1")


def to_grid(mask):
    return "/".join("".join(str((mask >> (r * SIZE + c)) & 1)
                             for c in range(SIZE)) for r in range(SIZE))


def normalized_key(mask):
    occupied = list(cells(mask))
    if not occupied:
        return "empty"
    r0, r1 = min(i // SIZE for i in occupied), max(i // SIZE for i in occupied)
    c0, c1 = min(i % SIZE for i in occupied), max(i % SIZE for i in occupied)
    return "/".join("".join(str((mask >> (r * SIZE + c)) & 1)
                             for c in range(c0, c1 + 1)) for r in range(r0, r1 + 1))


def legal_indices(mask):
    return [i for i, (tile, _) in enumerate(PLACEMENTS) if mask & tile == tile]


def apply(mask, action):
    if not isinstance(action, dict):
        raise ValueError("malformed action")
    key = tuple(action.get(k) for k in ("shape_id", "row", "col"))
    if any(type(x) is not int for x in key):
        raise ValueError("shape_id, row, and col must be integers")
    index = BY_ACTION.get(key)
    if index is None:
        raise ValueError("unknown shape or out-of-bounds anchor")
    tile = PLACEMENTS[index][0]
    if mask & tile != tile:
        raise ValueError("shape overlaps an empty cell")
    return mask ^ tile


def solver():
    """Return an independent per-board exact-cover DP (minimum removals, witness).

    Choosing any remaining cell is complete: every valid tiling must contain
    one of its available placements. Disjoint removals commute in this task.
    None means untileable even without a step budget, not just over budget.
    """
    @lru_cache(None)
    def solve(mask):
        if not mask:
            return 0, ()
        choices = None
        for cell in cells(mask):
            options = [i for i in BY_CELL[cell]
                       if PLACEMENTS[i][0] & mask == PLACEMENTS[i][0]]
            if not options:
                return None, ()
            if choices is None or len(options) < len(choices):
                choices = options
        best, witness = None, ()
        for index in sorted(choices, key=lambda i: -PLACEMENTS[i][0].bit_count()):
            count, suffix = solve(mask ^ PLACEMENTS[index][0])
            if count is not None and (best is None or count + 1 < best):
                best, witness = count + 1, (index,) + suffix
            if best == (mask.bit_count() + 2) // 3:
                break
        return best, witness
    return solve


def locally_supported(mask):
    """Every occupied cell belongs to at least one currently legal placement."""
    return all(any(PLACEMENTS[i][0] & mask == PLACEMENTS[i][0]
                   for i in BY_CELL[cell]) for cell in cells(mask))


def plausible_prefix(mask, cap):
    """Longest legal prefix up to cap with local support at each state.

    This is an operational diagnostic, not a lower bound on human/LLM effort.
    For an untileable input it shows that local cell coverage can stay plausible
    along a branch although an exact completion does not exist.
    """
    if not locally_supported(mask):
        return -1
    if cap == 0 or mask == 0:
        return 0
    best = 0
    for index in legal_indices(mask):
        depth = plausible_prefix(mask ^ PLACEMENTS[index][0], cap - 1)
        if depth >= 0:
            best = max(best, depth + 1)
        if best == cap:
            break
    return best


def diagnose(mask, budget, rng, rollouts=16, prefix_cap=3):
    solve = solver()
    minimum, witness = solve(mask)
    if minimum is None or minimum > budget:
        raise ValueError("Candidate has no solution within its action budget")
    legal = legal_indices(mask)
    traps = []
    budget_traps = 0
    for index in legal:
        after = mask ^ PLACEMENTS[index][0]
        remaining_min, _ = solve(after)
        if remaining_min is None:
            supported = locally_supported(after)
            traps.append({"action": PLACEMENTS[index][1], "locally_supported": supported,
                          "plausible_prefix_cap": prefix_cap,
                          "plausible_prefix_length_capped": plausible_prefix(after, prefix_cap)
                          if supported else -1})
        elif remaining_min > budget - 1:
            budget_traps += 1
    successes = {"random": 0, "largest_first": 0}
    for policy in successes:
        for _ in range(rollouts):
            remaining = mask
            for _ in range(budget):
                options = legal_indices(remaining)
                if not options:
                    break
                if policy == "largest_first":
                    area = max(PLACEMENTS[i][0].bit_count() for i in options)
                    options = [i for i in options if PLACEMENTS[i][0].bit_count() == area]
                remaining ^= PLACEMENTS[rng.choice(options)][0]
            successes[policy] += remaining == 0
    count = len(legal)
    return {"occupied_cells": mask.bit_count(), "minimum_actions": minimum,
            "initial_legal_actions": count, "initial_dead_end_actions": len(traps),
            "initial_locally_supported_dead_end_actions": sum(t["locally_supported"] for t in traps),
            "initial_budget_only_traps": budget_traps,
            "dead_end_fraction": len(traps) / count,
            "locally_supported_dead_end_fraction": sum(t["locally_supported"] for t in traps) / count,
            "proxy_rollouts_per_policy": rollouts, "proxy_successes": successes,
            "initial_traps": traps,
            "reference_actions": [PLACEMENTS[i][1] for i in witness]}


def judge(grid, actions, budget):
    mask = from_grid(grid)
    if not isinstance(actions, list):
        return {"pass": False, "legal": False, "failure_type": "missing_actions", "trace": []}
    solve = solver()
    trace, first_loss, illegal = [], None, None
    for step, action in enumerate(actions[:budget], 1):
        before = mask
        try:
            mask = apply(mask, action)
        except ValueError as error:
            illegal = {"step": step, "reason": str(error), "action": action}
            break
        minimum, _ = solve(mask)
        record = {"step": step, "action": action, "remaining_grid": to_grid(mask),
                  "remaining_cells": mask.bit_count(), "minimum_remaining_actions": minimum}
        if first_loss is None and (minimum is None or minimum > budget - step):
            first_loss = {"step": step, "kind": "untileable" if minimum is None else "budget_only",
                          "before_grid": to_grid(before), "after_grid": to_grid(mask),
                          "action": action, "locally_supported_after": locally_supported(mask)}
        trace.append(record)
    solved = mask == 0
    failure = ("illegal_action" if illegal else "too_many_actions" if len(actions) > budget
               else "dead_end" if not solved and first_loss and first_loss["kind"] == "untileable"
               else "budget_exhausted" if not solved and len(actions) >= budget
               else "stopped_early" if not solved else None)
    return {"pass": failure is None, "legal": illegal is None, "solved": solved,
            "attempted_actions": len(actions), "executed_actions": len(trace),
            "failure_type": failure, "illegal_action": illegal,
            "first_irrecoverable_action": first_loss,
            "remaining_grid": to_grid(mask), "remaining_cells": mask.bit_count(), "trace": trace}
