"""Sticker permutations for the harder 3x3 condition, with fixed face frames."""
from functools import lru_cache

FACES = ("Upper", "Front", "Down", "Left", "Right", "Back")
LETTERS = dict(zip("UFDLRB", FACES))
COLORS = (0, 2, 3, 4, 1, 5)
NORMALS = ((0,1,0),(0,0,1),(0,-1,0),(-1,0,0),(1,0,0),(0,0,-1))
ACTIONS = tuple(letter + suffix for letter in LETTERS for suffix in ("", "'"))


@lru_cache(None)
def stickers(size=3):
    axis = [-1 + 2*i/(size-1) for i in range(size)]
    result = []
    for face, normal in enumerate(NORMALS):
        for row in range(size):
            for col in range(size):
                a,b = axis[row],axis[col]
                point = ((b,1,a),(b,-a,1),(b,-1,-a),(-1,-a,b),(1,-a,-b),(-b,-a,-1))[face]
                result.append((point,normal))
    return tuple(result)


def rotate(vector,axis,sign):
    # A clockwise face turn is -90 degrees about its outward normal.
    x,y,z = vector
    if axis == 0:
        return (x,sign*z,-sign*y)
    if axis == 1:
        return (-sign*z,y,sign*x)
    return (sign*y,-sign*x,z)


@lru_cache(None)
def permutation(action,size=3):
    if action not in ACTIONS:
        raise ValueError("Unknown quarter-turn")
    normal = NORMALS[FACES.index(LETTERS[action[0]])]
    axis = next(i for i,value in enumerate(normal) if value)
    layer = normal[axis]
    sign = layer * (-1 if action.endswith("'") else 1)
    positions = stickers(size)
    index = {value:i for i,value in enumerate(positions)}
    destination = []
    for point,direction in positions:
        changed = (rotate(point,axis,sign),rotate(direction,axis,sign)) if point[axis] == layer else (point,direction)
        destination.append(index[changed])
    return tuple(destination)


def solved(size=3):
    return tuple(color for color in COLORS for _ in range(size*size))


def step(state,action,size=3):
    if len(state) != 6*size*size:
        raise ValueError("Wrong sticker count")
    result = list(state)
    for source,target in enumerate(permutation(action,size)):
        result[target] = state[source]
    return tuple(result)


def render(state,size=3):
    lines = []
    for face,label in enumerate(FACES):
        lines.append(label+":")
        values = state[face*size*size:(face+1)*size*size]
        lines.extend(" ".join(map(str,values[row*size:(row+1)*size])) for row in range(size))
    return "\n".join(lines)
