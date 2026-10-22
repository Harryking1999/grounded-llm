"""Current sorted slots and exact rational replay for Game24."""
from .environment import Game24

ENV = Game24()


def fields(case):
    return {'initial_numbers': ' '.join(str(v) for v in sorted(case['initial']))}


def initial(case):
    return ENV.canonical(case['initial'])


def step(case, state, action):
    if (not isinstance(action, list) or len(action) != 3 or type(action[0]) is not int
            or type(action[2]) is not int or not isinstance(action[1], str)):
        raise ValueError('Expected [integer_i, operation, integer_j]')
    key = tuple(action)
    for index, target, cost in ENV.transitions(state):
        if ENV.actions[index] == key:
            return target, cost
    raise ValueError('Invalid current-slot indices, operation or division by zero')


def solved(case, state):
    return state == ENV.goal
