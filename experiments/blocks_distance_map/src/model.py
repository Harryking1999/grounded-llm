"""Shared state MLP and relation objective; legacy Q table remains reproducible."""
import numpy as np
import torch
from torch import nn


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


from grounded_llm.core.qmap.model import distance, soft_rank_loss, StateEncoder


from grounded_llm.tasks.blocks.model import board_bits, BoardEncoder


from grounded_llm.tasks.blocks.sampling import strata
