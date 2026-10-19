"""Size-aware sliding-puzzle transitions; no search or agent-side tools."""
from math import isqrt

from grounded_llm.tasks.puzzle8.rules import MOVES


def size_of(state):
    size = isqrt(len(state))
    if size < 2 or size*size != len(state):
        raise ValueError('Expected a square board')
    return size


def step(state,action):
    if not isinstance(action,str) or action not in MOVES:
        raise ValueError('Unknown direction')
    size = size_of(state)
    row,col = divmod(state.index(0),size)
    dr,dc = MOVES[action]
    rr,cc = row+dr,col+dc
    if not (0 <= rr < size and 0 <= cc < size):
        raise ValueError('Blank leaves board')
    result = list(state)
    source,target = row*size+col,rr*size+cc
    result[source],result[target] = result[target],0
    return tuple(result)


def render(state):
    size = size_of(state)
    return '\n'.join(' '.join(map(str,state[i:i+size])) for i in range(0,len(state),size))


def manhattan(state):
    size = size_of(state)
    return sum(abs(i//size-value//size)+abs(i%size-value%size) for i,value in enumerate(state) if value)
