"""Text and strict physical replay for full token swapping."""
from functools import lru_cache
from .environment import TokenSwapping


@lru_cache(maxsize=128)
def environment(n, edges):
    return TokenSwapping(n, edges)


def env(case):
    return environment(len(case['initial']), tuple(tuple(e) for e in case['edges']))


def fields(case):
    n = len(case['initial'])
    return {'vertex_count': n, 'max_vertex_id': n-1,
            'edges': str(case['edges']),
            'initial_tokens': str(case['initial']), 'goal_tokens': str(list(range(n)))}


def initial(case):
    return tuple(case['initial'])


def step(case, state, action):
    return env(case).apply(state, action)


def solved(case, state):
    return state == env(case).goal
