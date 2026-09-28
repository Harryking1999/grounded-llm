"""Goal-independent board Q encoder and fixed directed distance."""
import numpy as np
import torch
from torch import nn


def distance(source, target, metric):
    difference = source - target
    if metric == "directed_max":
        return difference.amax(dim=-1).clamp_min(0)
    if metric == "directed_sum":
        return difference.clamp_min(0).sum(dim=-1) / source.shape[-1] ** 0.5
    if metric == "euclidean":
        return difference.norm(dim=-1)
    raise ValueError(metric)


def board_bits(masks):
    """100 occupancy bits in the exact row-major order used by the rules."""
    values = [int(mask) for mask in masks]
    return torch.from_numpy(np.fromiter(((mask >> bit) & 1 for mask in values for bit in range(100)),
                                        dtype=np.float32, count=len(values) * 100).reshape(-1, 100))


class BoardEncoder(nn.Module):
    def __init__(self, dimension, hidden, cap, metric):
        super().__init__()
        self.cap = cap
        self.metric = metric
        self.network = nn.Sequential(nn.Linear(100, hidden), nn.ReLU(),
                                     nn.Linear(hidden, hidden), nn.ReLU(),
                                     nn.Linear(hidden, dimension))
        nn.init.normal_(self.network[-1].weight, std=.01)
        nn.init.zeros_(self.network[-1].bias)

    def encode(self, bits):
        return self.cap * torch.sigmoid(self.network(bits))

    def forward(self, bit_table, pairs):
        return distance(self.encode(bit_table[pairs[:, 0]]),
                        self.encode(bit_table[pairs[:, 1]]), self.metric)


def strata(pairs):
    """Half weight to finite distances, half to certified negatives."""
    finite = [(pairs[:, 2] == 1), (pairs[:, 2] == 2),
              ((pairs[:, 2] >= 3) & (pairs[:, 2] <= 4)), (pairs[:, 2] >= 5)]
    negative = [(pairs[:, 3] == 1), (pairs[:, 3] == 2)]
    families = [[mask.nonzero(as_tuple=True)[0] for mask in family if mask.any()]
                for family in (finite, negative)]
    present = sum(bool(family) for family in families)
    return [(ids, 1 / present / len(family)) for family in families for ids in family]
