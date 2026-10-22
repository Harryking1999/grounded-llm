"""Typed rendering and replay for Rush Hour."""
from functools import lru_cache
from .environment import RushHour


@lru_cache(maxsize=128)
def environment(board):
    return RushHour(board)


def fields(case):
    env = environment(case['initial'])
    target = goal(case)
    return {'initial_board': '\n'.join(env.render(env.start)),
            'goal_rule': ('The goal is to match the specified goal board exactly, including the positions of every vehicle. Do not move A outside the board or remove it.'
                          if target is not None else
                          'The goal is to place the horizontal target vehicle A at the rightmost position in its row, occupying columns 4 and 5 (zero-based). Other vehicles may end anywhere legal. Do not move A outside the board or remove it.'),
            'goal_board_section': '\n\nGoal board, top row first:\n'+'\n'.join(env.render(target)) if target is not None else '',
            'vehicle_catalogue': '\n'.join(f'{label}: {"horizontal" if h else "vertical"}, length {size}'
                                          for label, h, size, _ in env.pieces)}


def goal(case):
    if 'goal_board' not in case:
        return None
    source = environment(case['initial'])
    target = environment(case['goal_board'])
    if (source.walls != target.walls or source.pieces != target.pieces
            or not target.terminal(target.start)):
        raise ValueError('Goal board must preserve the vehicle layout and put A at the rightmost position')
    return target.start


def initial(case):
    return environment(case['initial']).start


def step(case, state, action):
    return environment(case['initial']).apply(state, action)


def solved(case, state):
    target = goal(case)
    return state == target if target is not None else environment(case['initial']).terminal(state)
