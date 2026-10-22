"""ASCII text fields and physical action replay for Sokoban."""
from functools import lru_cache
from .environment import Sokoban


@lru_cache(maxsize=256)
def environment(lines):
    return Sokoban(lines)


def fields(case):
    target = goal(case)
    return {'board_height': len(case['initial']), 'board_width': len(case['initial'][0]),
            'initial_board': '\n'.join(case['initial']),
            'task_objective': ('Find a valid sequence of moves that exactly matches the specified goal board, as short as you can.'
                               if target is not None else
                               'Find a valid sequence of moves that puts every box on a target, as short as you can.'),
            'goal_rule': ('The goal is to match the specified goal board exactly, including the player and all box positions. Boxes are indistinguishable.'
                          if target is not None else
                          "The goal is that every box is on a target; the player's final position does not matter."),
            'goal_board_section': '\n\nGoal board:\n'+'\n'.join(case['goal_board']) if target is not None else ''}


def goal(case):
    if 'goal_board' not in case:
        return None
    source = environment(tuple(case['initial']))
    target = environment(tuple(case['goal_board']))
    if ((source.height, source.width, source.floor, source.goals) !=
            (target.height, target.width, target.floor, target.goals)
            or not target.terminal(target.start)):
        raise ValueError('Goal board must preserve the layout and put every box on a target')
    return target.start


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
    target = goal(case)
    return state == target if target is not None else environment(tuple(case['initial'])).terminal(state)
