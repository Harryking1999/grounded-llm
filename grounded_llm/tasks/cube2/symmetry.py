"""Exact six-axis projection, with permutations rather than equal state Qs."""
import numpy as np

from grounded_llm.tasks.cube2.environment import Cube2


def axis_projection_indices():
    env = Cube2(metric="QTM_URF_6")
    maps = env.axis_corner_map
    product = np.empty((6, 6), dtype=np.int64)
    for left in range(6):
        for right in range(6):
            composed = maps[left, maps[right]]
            product[left, right] = np.flatnonzero((maps == composed).all(-1))[0]
    inverse = np.array([np.flatnonzero(product[g] == 0)[0] for g in range(6)])
    # Flat feature index = position * 24 + piece * 3 + orientation.
    position, piece, orientation = np.indices((8, 8, 3)).reshape(3, -1)
    feature_gather = []
    for g in range(6):
        twist = (env.axis_sign[g] * orientation + env.axis_offset[g, position]
                 - env.axis_offset[g, piece]) % 3
        destination = maps[g, position] * 24 + maps[g, piece] * 3 + twist
        feature_gather.append(np.argsort(destination))
    # 21 copies of the regular representation, plus two invariant channels.
    # Q(s) = mean_h rho(h)^-1 F(phi_h(s)); hence Q(phi_g(s)) = rho(g) Q(s).
    def channels(indices):
        return np.r_[(np.arange(21)[:, None] * 6 + indices).ravel(), 126, 127]
    align = np.stack([channels(product[h]) for h in range(6)])
    action = np.stack([channels(product[inverse[g]]) for g in range(6)])
    return np.asarray(feature_gather), align, action
