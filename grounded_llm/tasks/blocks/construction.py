"""Seeded nonoverlapping constructions under the original eight shapes."""
import random
from . import rules


def construct(count, placements, seed):
    rng = random.Random(seed)
    roots, seen = [], set()
    while len(roots) < count:
        state, witness = 0, []
        for _ in range(placements):
            shape = rng.randrange(len(rules.SHAPES))
            legal = [(tile, action) for tile, action in rules.PLACEMENTS
                     if action['shape_id'] == shape and not tile & state]
            if not legal:
                break
            tile, action = rng.choice(legal)
            state |= tile
            witness.append(action)
        if len(witness) == placements and state not in seen:
            seen.add(state)
            roots.append(dict(initial=str(state), construction=witness))
    return roots
