"""A goal-independent Q table and fixed distance functions; no V or decoder."""
import torch
from torch import nn


def distance(source, target, metric):
    difference = source - target
    if metric == "directed_max":
        return difference.amax(dim=-1).clamp_min(0)
    if metric == "euclidean":
        return difference.norm(dim=-1)
    raise ValueError(metric)


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


def strata(pairs):
    """Half weight to finite distances, half to certified negatives."""
    finite = [(pairs[:, 2] == 1), (pairs[:, 2] == 2),
              ((pairs[:, 2] >= 3) & (pairs[:, 2] <= 4)), (pairs[:, 2] >= 5)]
    negative = [(pairs[:, 3] == 1), (pairs[:, 3] == 2)]
    families = [[mask.nonzero(as_tuple=True)[0] for mask in family if mask.any()]
                for family in (finite, negative)]
    present = sum(bool(family) for family in families)
    return [(ids, 1 / present / len(family)) for family in families for ids in family]
