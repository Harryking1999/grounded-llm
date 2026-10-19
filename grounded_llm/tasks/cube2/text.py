"""Exact current cube environment and its XoT-compatible sticker presentation."""
from functools import lru_cache

from grounded_llm.tasks.cube2.environment import Cube2, CORNER_NAMES

COLORS = dict(U=0, R=1, F=2, D=3, L=4, B=5)
FACE_SLOTS = {"Upper": ("U", (2,3,1,0)), "Front": ("F", (1,0,5,4)),
              "Down": ("D", (5,4,6,7)), "Left": ("L", (2,1,6,5)),
              "Right": ("R", (0,3,4,7)), "Back": ("B", (3,2,7,6))}


@lru_cache(None)
def environment():
    return Cube2(metric="QTM_URF_6")


def grids(state):
    desc = environment().describe(state)
    result = {}
    for label, (face, slots) in FACE_SLOTS.items():
        values = []
        for slot in slots:
            cubie, twist = desc["permutation"][slot], desc["orientation"][slot]
            index = (CORNER_NAMES[slot].index(face) - twist) % 3
            values.append(COLORS[CORNER_NAMES[cubie][index]])
        result[label] = values
    return result


def from_grids(values):
    at_slot = [{} for _ in range(8)]
    for label, (face, slots) in FACE_SLOTS.items():
        for slot, color in zip(slots, values[label]):
            at_slot[slot][face] = color
    cp, co = [], []
    for slot, observed in enumerate(at_slot):
        cubie = next(i for i, name in enumerate(CORNER_NAMES) if {COLORS[f] for f in name} == set(observed.values()))
        twist = next(t for t in range(3) if all(observed[face] == COLORS[CORNER_NAMES[cubie][(j-t) % 3]] for j, face in enumerate(CORNER_NAMES[slot])))
        cp.append(cubie)
        co.append(twist)
    return environment().state_id(cp, co)


def render(state):
    return "\n".join(label + ":\n" + " ".join(map(str,v[:2])) + "\n" + " ".join(map(str,v[2:])) for label, v in grids(state).items())


def step(state, action):
    if not isinstance(action, str):
        raise ValueError("Turn must be a string")
    return environment().step(int(state), action)
