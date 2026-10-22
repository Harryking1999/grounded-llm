"""Exact three-operation witnesses and an integer-intermediate restriction probe."""
from functools import lru_cache
from .environment import Game24

ENV = Game24()


@lru_cache(maxsize=100000)
def solve(state, integer_only=False):
    if len(state) == 1:
        return () if state == ENV.goal else None
    seen = set()
    for action, child, _ in ENV.transitions(state):
        if child in seen or integer_only and any(v.denominator != 1 for v in child):
            continue
        seen.add(child)
        tail = solve(child, integer_only)
        if tail is not None:
            return (ENV.actions[action],) + tail
    return None
