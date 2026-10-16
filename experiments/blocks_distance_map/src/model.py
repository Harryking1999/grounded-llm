"""Shared state MLP and relation objective; legacy Q table remains reproducible."""
import numpy as np
import torch
from torch import nn


def distance(source, target, metric, group_size=8):
    difference = source - target
    if metric == "directed_max":
        return difference.amax(dim=-1).clamp_min(0)
    if metric == "directed_sum":
        return difference.clamp_min(0).sum(dim=-1) / source.shape[-1] ** 0.5
    if metric in ("euclidean", "l2"):
        return difference.norm(dim=-1)
    if metric == "l1":
        return (distance(source, target, "directed_sum") +
                distance(target, source, "directed_sum")) / 2
    if metric == "linf":
        return difference.abs().amax(dim=-1)
    if metric == "max_l1":
        if group_size < 1 or difference.shape[-1] % group_size:
            raise ValueError("Q dimension must be divisible by the positive group size")
        return difference.abs().reshape(*difference.shape[:-1], -1, group_size).sum(-1).amax(-1)
    raise ValueError(metric)


def soft_rank_loss(scores, truth, valid, temperature):
    """One soft-distance ranking objective, shared by blocks and all Qmap tasks."""
    target = torch.softmax((-truth.float() / temperature).masked_fill(~valid, -torch.inf), dim=-1)
    log_prob = torch.log_softmax((-scores / temperature).masked_fill(~valid, -torch.inf), dim=-1)
    return -(target * log_prob.masked_fill(~valid, 0)).sum(dim=-1).mean()


class QMap(nn.Module):
    def __init__(self, states, dimension, cap, metric, init_std):
        super().__init__()
        self.q = nn.Embedding(states, dimension)
        self.cap = cap
        self.metric = metric
        nn.init.normal_(self.q.weight, mean=cap / 2, std=init_std)
        self.project()

    def forward(self, pairs):
        return distance(self.q(pairs[:, 0]), self.q(pairs[:, 1]), self.metric)

    @torch.no_grad()
    def project(self):
        self.q.weight.clamp_(0, self.cap)


def board_bits(masks):
    """100 occupancy bits in the exact row-major order used by the rules."""
    values = [int(mask) for mask in masks]
    return torch.from_numpy(np.fromiter(((mask >> bit) & 1 for mask in values for bit in range(100)),
                                        dtype=np.float32, count=len(values) * 100).reshape(-1, 100))


class StateEncoder(nn.Module):
    """The same two-hidden-layer Q encoder; only input encoding and metric vary."""
    def __init__(self, dimension, hidden, cap, metric, input_dim=100, group_size=8):
        super().__init__()
        self.cap = cap
        self.metric = metric
        self.group_size = group_size
        self.network = nn.Sequential(nn.Linear(input_dim, hidden), nn.ReLU(),
                                     nn.Linear(hidden, hidden), nn.ReLU(),
                                     nn.Linear(hidden, dimension))
        nn.init.normal_(self.network[-1].weight, std=.01)
        nn.init.zeros_(self.network[-1].bias)

    def encode(self, bits):
        return self.cap * torch.sigmoid(self.network(bits))

    def forward(self, bit_table, pairs):
        return self.distance(self.encode(bit_table[pairs[:, 0]]),
                             self.encode(bit_table[pairs[:, 1]]))

    def distance(self, source, target):
        return distance(source, target, self.metric, self.group_size)


class BoardEncoder(StateEncoder):
    """Historical constructor name; state_dict keys and outputs are preserved."""


def strata(pairs):
    """Half weight to finite distances, half to certified negatives."""
    finite = [(pairs[:, 2] == 1), (pairs[:, 2] == 2),
              ((pairs[:, 2] >= 3) & (pairs[:, 2] <= 4)), (pairs[:, 2] >= 5)]
    negative = [(pairs[:, 3] == 1), (pairs[:, 3] == 2)]
    families = [[mask.nonzero(as_tuple=True)[0] for mask in family if mask.any()]
                for family in (finite, negative)]
    present = sum(bool(family) for family in families)
    return [(ids, 1 / present / len(family)) for family in families for ids in family]
