"""Typed rendering and replay for Rush Hour."""
from functools import lru_cache
from .environment import RushHour


@lru_cache(maxsize=128)
def environment(board):
    return RushHour(board)


def fields(case):
    from grounded_llm.tasks.text import section_text
    env = environment(case['initial'])
    target = goal(case)
    return {'initial_board': '\n'.join(env.render(env.start)),
            'goal_rule': section_text('rush_hour', '指定完整目标文字' if target is not None else '历史目标文字：仅回放'),
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
