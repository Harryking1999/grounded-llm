"""Encode current, goal, and variable-length successor map vectors."""

from dataclasses import dataclass

import torch
from torch import Tensor, nn


CURRENT = 0
GOAL = 1
SUCCESSOR = 2


@dataclass(frozen=True)
class MapBatch:
    vectors: Tensor  # [batch, slots, map_dim]
    roles: Tensor  # 0=current, 1=goal, 2=successor
    candidate_ids: Tensor  # 0=none/padding, 1..k=local candidate index
    valid: Tensor  # False for padding slots

    def validate(self, max_candidates: int) -> None:
        if self.vectors.ndim != 3:
            raise ValueError("vectors must have shape [batch, slots, map_dim]")
        shape = self.vectors.shape[:2]
        if any(t.shape != shape for t in (self.roles, self.candidate_ids, self.valid)):
            raise ValueError("roles, candidate_ids and valid must match vector slots")
        if self.roles.dtype != torch.long or self.candidate_ids.dtype != torch.long:
            raise TypeError("roles and candidate_ids must be int64 tensors")
        if self.valid.dtype != torch.bool:
            raise TypeError("valid must be a boolean tensor")
        if not torch.isfinite(self.vectors).all():
            raise ValueError("map vectors contain non-finite values")
        if not torch.all((self.roles >= CURRENT) & (self.roles <= SUCCESSOR)):
            raise ValueError("unknown map role")
        if not torch.all((self.candidate_ids >= 0) & (self.candidate_ids <= max_candidates)):
            raise ValueError("candidate ID exceeds configured range")
        if not torch.all(((self.roles == CURRENT) & self.valid).sum(dim=1) == 1):
            raise ValueError("each sample needs one current slot")
        if not torch.all(((self.roles == GOAL) & self.valid).sum(dim=1) == 1):
            raise ValueError("each sample needs one goal slot")
        no_id = (self.roles != SUCCESSOR) & self.valid
        if not torch.all(self.candidate_ids[no_id] == 0):
            raise ValueError("current and goal slots must use candidate ID 0")
        successor = (self.roles == SUCCESSOR) & self.valid
        if not torch.all(self.candidate_ids[successor] > 0):
            raise ValueError("successor slots need a positive candidate ID")


@dataclass(frozen=True)
class MapTimeline:
    """Map snapshots for one trajectory and the snapshot assigned to each token."""

    snapshots: MapBatch  # snapshot dimension is MapBatch's batch dimension
    token_map_ids: Tensor  # [1, text_length], zero-based snapshot indices

    def validate(self, max_candidates: int, text_length: int) -> None:
        self.snapshots.validate(max_candidates)
        count = self.snapshots.vectors.shape[0]
        if self.token_map_ids.shape != (1, text_length):
            raise ValueError("token_map_ids must match the text length")
        if self.token_map_ids.dtype != torch.long:
            raise TypeError("token_map_ids must be int64")
        if not torch.all((self.token_map_ids >= 0) & (self.token_map_ids < count)):
            raise ValueError("token_map_ids refers to a missing map snapshot")


class MapMemoryEncoder(nn.Module):
    """Shared P, role/ID embeddings, then G from section 4.1."""

    def __init__(
        self,
        map_dim: int,
        language_dim: int,
        max_candidates: int,
        projection_dim: int,
        label_dim: int,
    ) -> None:
        super().__init__()
        if min(map_dim, language_dim, max_candidates, projection_dim, label_dim) <= 0:
            raise ValueError("all dimensions and max_candidates must be positive")
        self.map_dim = map_dim
        self.max_candidates = max_candidates
        self.project = nn.Linear(map_dim, projection_dim)
        self.role_embedding = nn.Embedding(3, label_dim)
        self.id_embedding = nn.Embedding(max_candidates + 1, label_dim)
        self.combine = nn.Linear(projection_dim + 2 * label_dim, language_dim)

    def forward(self, batch: MapBatch) -> Tensor:
        batch.validate(self.max_candidates)
        if batch.vectors.shape[-1] != self.map_dim:
            raise ValueError("map vector dimension differs from encoder")
        features = torch.cat(
            (
                self.project(batch.vectors),
                self.role_embedding(batch.roles),
                self.id_embedding(batch.candidate_ids),
            ),
            dim=-1,
        )
        return self.combine(features).masked_fill(~batch.valid.unsqueeze(-1), 0)
