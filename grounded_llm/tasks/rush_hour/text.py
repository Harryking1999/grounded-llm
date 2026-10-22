"""Typed rendering and replay for Rush Hour."""
from functools import lru_cache
from .environment import RushHour


@lru_cache(maxsize=128)
def environment(board):
    return RushHour(board)


def fields(case):
    env = environment(case['initial'])
    return {'initial_board': '\n'.join(env.render(env.start)),
            'vehicle_catalogue': '\n'.join(f'{label}: {"horizontal" if h else "vertical"}, length {size}'
                                          for label, h, size, _ in env.pieces)}


def initial(case):
    return environment(case['initial']).start


def step(case, state, action):
    return environment(case['initial']).apply(state, action)


def solved(case, state):
    return environment(case['initial']).terminal(state)
