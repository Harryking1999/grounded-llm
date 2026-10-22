"""ASCII text fields and physical action replay for Sokoban."""
from functools import lru_cache
from .environment import Sokoban


@lru_cache(maxsize=256)
def environment(lines):
    return Sokoban(lines)


def fields(case):
    return {'board_height': len(case['initial']), 'board_width': len(case['initial'][0]),
            'initial_board': '\n'.join(case['initial'])}


def initial(case):
    return environment(tuple(case['initial'])).start


def step(case, state, action):
    env = environment(tuple(case['initial']))
    if not isinstance(action, str) or action not in env.actions:
        raise ValueError('Expected U, D, L or R')
    for index, target, cost in env.transitions(state):
        if env.actions[index] == action:
            return target, cost
    raise ValueError('Move blocked by a wall or an unpushable box')


def solved(case, state):
    return environment(tuple(case['initial'])).terminal(state)
