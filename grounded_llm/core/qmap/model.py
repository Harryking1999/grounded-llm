"""State encoder, fixed distances and candidate ranking objective."""
import torch
from math import prod
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


class StateEncoder(nn.Module):
    """Independent state encoder; existing checkpoints default to the same MLP."""
    def __init__(self, dimension, hidden, cap, metric, input_dim=100, group_size=8, output_activation='sigmoid',
                 architecture='mlp',spatial_shape=None,conv_channels=16,spatial_hidden=112):
        super().__init__()
        self.cap = cap
        self.metric = metric
        self.group_size = group_size
        if output_activation not in ('sigmoid','linear','softsign'):
            raise ValueError('Expected sigmoid, linear or softsign Q output')
        self.output_activation = output_activation
        if architecture=='mlp':
            if spatial_shape is not None: raise ValueError('Spatial shape requires a spatial encoder')
            self.network = nn.Sequential(nn.Linear(input_dim, hidden), nn.ReLU(),
                                         nn.Linear(hidden, hidden), nn.ReLU(),
                                         nn.Linear(hidden, dimension))
        elif architecture=='spatial_cnn':
            if spatial_shape is None or len(spatial_shape)!=3 or min(spatial_shape)<1 or prod(spatial_shape)!=input_dim:
                raise ValueError('Spatial shape must be channels, height, width matching the raw input')
            if conv_channels<1 or spatial_hidden<1: raise ValueError('Positive spatial widths required')
            channels,height,width=spatial_shape
            self.network=nn.Sequential(nn.Unflatten(-1,tuple(spatial_shape)),
                nn.Conv2d(channels,conv_channels,3,padding=1),nn.ReLU(),
                nn.Conv2d(conv_channels,conv_channels,3,padding=1),nn.ReLU(),
                nn.Conv2d(conv_channels,conv_channels,3,padding=1),nn.ReLU(),
                nn.Flatten(start_dim=-3),nn.Linear(conv_channels*height*width,spatial_hidden),nn.ReLU(),
                nn.Linear(spatial_hidden,dimension))
        else: raise ValueError('Expected mlp or spatial_cnn architecture')
        nn.init.normal_(self.network[-1].weight, std=.01)
        nn.init.zeros_(self.network[-1].bias)

    def encode(self, bits):
        values=self.network(bits)
        if self.output_activation=='sigmoid': values=torch.sigmoid(values)
        elif self.output_activation=='softsign':
            # Same range, centre and derivative at zero as sigmoid; slower tails.
            values=.5*(1+torch.nn.functional.softsign(values/2))
        return self.cap * values

    def forward(self, bit_table, pairs):
        return self.distance(self.encode(bit_table[pairs[:, 0]]),
                             self.encode(bit_table[pairs[:, 1]]))

    def distance(self, source, target):
        return distance(source, target, self.metric, self.group_size)
