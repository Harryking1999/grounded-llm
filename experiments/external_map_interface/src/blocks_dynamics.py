"""Tabular Blocks Q and a shared, state-conditioned action displacement."""

import torch
from torch import nn


class BlocksDynamics(nn.Module):
    def __init__(self, state_count, action_count, config):
        super().__init__()
        dim = config["state_dim"]
        self.q = nn.Embedding(state_count, dim)
        self.action = nn.Embedding(action_count, config["action_dim"])
        self.displacement = nn.Sequential(
            nn.Linear(dim + config["action_dim"], config["hidden_dim"]),
            nn.ReLU(),
            nn.Linear(config["hidden_dim"], dim),
        )
        nn.init.normal_(self.q.weight, std=config["q_init_std"])

    def predict(self, current_q, action_ids):
        """Also accepts a predicted Q during open-loop rollout; no state lookup."""
        delta = self.displacement(torch.cat((current_q, self.action(action_ids)), dim=-1))
        return current_q + delta

    def forward(self, state_ids, action_ids):
        return self.predict(self.q(state_ids), action_ids)

    def candidate_distances(self, current_q, action_ids, goal_q):
        current = current_q.expand(len(action_ids), -1)
        return torch.linalg.vector_norm(self.predict(current, action_ids) - goal_q, dim=-1)
