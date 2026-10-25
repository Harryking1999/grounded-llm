"""State encoder, fixed distances and candidate ranking objective."""
import torch
from math import prod
from torch import nn


def widen_mlp_weights(state,factor,seed,noise_fraction=.01):
    """Replicate hidden units, preserving the initial function up to rounding.

    Zero-sum outgoing perturbations let replicated units diverge in training.
    This changes optimization conditioning; it is not an exact continuation.
    """
    if not isinstance(factor,int) or factor<2:raise ValueError('Integer widening factor >=2 required')
    values={key:value.detach().cpu().clone() for key,value in state.items()}
    first,middle,last=[values[f'network.{i}.weight'] for i in (0,2,4)]
    width=first.shape[0]
    if first.ndim!=2 or middle.shape!=(width,width) or last.ndim!=2 or last.shape[1]!=width:
        raise ValueError('Widening requires the existing two-hidden-layer MLP')
    rng=torch.Generator(device='cpu').manual_seed(seed)
    values['network.0.weight']=first.repeat_interleave(factor,dim=0)
    for key in ('network.0.bias','network.2.bias'):
        if key in values:values[key]=values[key].repeat_interleave(factor)
    for index,weight in ((2,middle),(4,last)):
        expanded=(weight.repeat_interleave(factor,dim=0) if index==2 else weight).repeat_interleave(factor,dim=1)/factor
        noise=torch.randn(expanded.shape,generator=rng,dtype=expanded.dtype).reshape(expanded.shape[0],width,factor)
        noise-=noise.mean(dim=-1,keepdim=True)
        expanded+=noise.reshape_as(expanded)*(float(weight.std())*noise_fraction/factor)
        values[f'network.{index}.weight']=expanded
    return values

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


def soft_rank_loss(scores, truth, valid, temperature, target_mode='soft_distance'):
    """One soft-distance ranking objective, shared by blocks and all Qmap tasks."""
    if target_mode=='soft_distance':
        target = torch.softmax((-truth.float() / temperature).masked_fill(~valid, -torch.inf), dim=-1)
    elif target_mode=='optimal_only':
        best=truth.float().masked_fill(~valid,torch.inf).amin(dim=-1,keepdim=True)
        target=((truth==best)&valid).float();target/=target.sum(dim=-1,keepdim=True)
    else:raise ValueError('Unknown candidate ranking target mode')
    log_prob = torch.log_softmax((-scores / temperature).masked_fill(~valid, -torch.inf), dim=-1)
    return -(target * log_prob.masked_fill(~valid, 0)).sum(dim=-1).mean()


def transition_difference_loss(parent,successor,parent_truth,successor_truth,valid):
    """Match remaining-cost changes across real legal edges at one query goal.

    Exact optimal edges decrease remaining cost by their immediate cost;
    finite detours may increase it. Unreachable/illegal branches receive no
    finite difference target and remain handled by distance/ranking losses.
    """
    finite=valid & torch.isfinite(successor_truth) & torch.isfinite(parent_truth[:,None])
    predicted=parent[:,None]-successor
    if not bool(finite.any()):return predicted.sum()*0
    target=parent_truth[:,None]-successor_truth
    return torch.nn.functional.smooth_l1_loss(predicted[finite],target[finite])


class StateEncoder(nn.Module):
    """Independent state encoder; existing checkpoints default to the same MLP."""
    def __init__(self, dimension, hidden, cap, metric, input_dim=100, group_size=8, output_activation='sigmoid',
                 architecture='mlp',spatial_shape=None,conv_channels=16,spatial_hidden=112,bias=True):
        super().__init__()
        self.cap = cap
        self.metric = metric
        self.group_size = group_size
        if not bias and architecture!='mlp':raise ValueError('Bias-free construction currently applies to the MLP')
        if output_activation not in ('sigmoid','linear','softsign'):
            raise ValueError('Expected sigmoid, linear or softsign Q output')
        self.output_activation = output_activation
        if architecture=='mlp':
            if spatial_shape is not None: raise ValueError('Spatial shape requires a spatial encoder')
            self.network = nn.Sequential(nn.Linear(input_dim, hidden,bias=bias), nn.ReLU(),
                                         nn.Linear(hidden, hidden,bias=bias), nn.ReLU(),
                                         nn.Linear(hidden, dimension,bias=bias))
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
        if self.network[-1].bias is not None:nn.init.zeros_(self.network[-1].bias)

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
