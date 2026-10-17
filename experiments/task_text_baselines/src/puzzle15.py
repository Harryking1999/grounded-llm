"""Independent 4x4 blank transitions for the harder sliding-puzzle condition."""
from .puzzle8 import MOVES


def step(state,action):
    if not isinstance(action,str) or action not in MOVES:
        raise ValueError("Unknown direction")
    row,col = divmod(state.index(0),4)
    dr,dc = MOVES[action]
    rr,cc = row+dr,col+dc
    if not (0 <= rr < 4 and 0 <= cc < 4):
        raise ValueError("Blank leaves board")
    result = list(state)
    target = rr*4+cc
    result[row*4+col],result[target] = result[target],0
    return tuple(result)


def render(state):
    return "\n".join(" ".join(map(str,state[offset:offset+4])) for offset in range(0,16,4))


def manhattan(state):
    return sum(abs(i//4-value//4)+abs(i%4-value%4) for i,value in enumerate(state) if value)
