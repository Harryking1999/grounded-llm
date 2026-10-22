"""Text fields and typed plan replay for bridge and torch."""
from functools import lru_cache
from .environment import Bridge


@lru_cache(maxsize=256)
def environment(times):
    return Bridge(times)


def fields(case):
    env = environment(tuple(case['times']))
    mask, side = divmod(case['initial'], 2)
    return {'people_count': env.n, 'max_person_id': env.n - 1,
            'crossing_times': '\n'.join(f'{i}: {time}' for i, time in enumerate(env.times)),
            'initial_left': [i for i in range(env.n) if not (mask >> i) & 1],
            'initial_right': [i for i in range(env.n) if (mask >> i) & 1],
            'initial_torch': 'right' if side else 'left',
            'all_people': list(range(env.n))}


def initial(case):
    return case['initial']


def step(case, state, action):
    if not isinstance(action, list) or not 1 <= len(action) <= 2 or any(type(x) is not int for x in action):
        raise ValueError('Expected one or two integer person IDs')
    key = tuple(sorted(action))
    env = environment(tuple(case['times']))
    for index, target, cost in env.transitions(state):
        if env.actions[index] == key:
            return target, cost
    raise ValueError('Travellers must be distinct people on the torch bank')


def solved(case, state):
    return state == environment(tuple(case['times'])).goal
