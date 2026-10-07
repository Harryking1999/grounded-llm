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
    """Concatenated snapshots, trajectory counts, and local per-token turn IDs."""

    snapshots: MapBatch  # snapshot dimension is MapBatch's batch dimension
    token_map_ids: Tensor  # [batch, text_length], local zero-based turn indices
    snapshot_counts: tuple[int, ...] | None = None

    def validate(self, max_candidates: int, text_length: int) -> None:
        self.snapshots.validate(max_candidates)
        count = self.snapshots.vectors.shape[0]
        counts = self.snapshot_counts or (count,)
        if min(counts) <= 0 or sum(counts) != count:
            raise ValueError("snapshot counts must partition the map snapshots")
        if self.token_map_ids.shape != (len(counts), text_length):
            raise ValueError("token_map_ids must match the text length")
        if self.token_map_ids.dtype != torch.long:
            raise TypeError("token_map_ids must be int64")
        limits = torch.tensor(counts, device=self.token_map_ids.device)[:, None]
        if not torch.all((self.token_map_ids >= 0) & (self.token_map_ids < limits)):
            raise ValueError("token_map_ids refers to a missing map snapshot")


@dataclass(frozen=True)
class AddressedMemory:
    """Key/value features for attention; joint-feature mode supplies the same features."""

    keys: Tensor
    values: Tensor


class MapMemoryEncoder(nn.Module):
    """Shared P, role/ID embeddings, then G from section 4.1.

    Deprecated: the mixed "joint" encoder, which projects state and identity into
    one vector and feeds it as both the attention key and the value. Every archived
    run that used it fitted its training first turn but scored 0/4 on held-out
    turns, and the diagnosis never localized why. Kept only so those archived
    configs stay reproducible. Current path configs use `AddressedMapMemoryEncoder`;
    blocks retraining uses `JointFeatureMapMemoryEncoder`. See
    ../results/archive/abandoned_readout_variants.md.
    """

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


class AddressedMapMemoryEncoder(nn.Module):
    """Encode roles/IDs as addresses and full Q vectors as state content."""

    def __init__(self, map_dim: int, language_dim: int, max_candidates: int,
                 projection_dim: int, label_dim: int, feature_ffn_hidden_dim: int = 0) -> None:
        super().__init__()
        if min(map_dim, language_dim, max_candidates, projection_dim, label_dim) <= 0:
            raise ValueError("all dimensions and max_candidates must be positive")
        self.map_dim = map_dim
        self.max_candidates = max_candidates
        self.language_dim = language_dim
        self.key_dim = 2 * label_dim
        self.value_dim = projection_dim
        self.project = nn.Linear(map_dim, projection_dim)
        if feature_ffn_hidden_dim < 0:
            raise ValueError("feature FFN width must be nonnegative")
        self.feature_ffn = (nn.Sequential(nn.Linear(projection_dim, feature_ffn_hidden_dim),
            nn.GELU(), nn.Linear(feature_ffn_hidden_dim, projection_dim))
            if feature_ffn_hidden_dim else None)
        self.role_embedding = nn.Embedding(3, label_dim)
        self.id_embedding = nn.Embedding(max_candidates + 1, label_dim)

    def forward(self, batch: MapBatch) -> AddressedMemory:
        batch.validate(self.max_candidates)
        if batch.vectors.shape[-1] != self.map_dim:
            raise ValueError("map vector dimension differs from encoder")
        addresses = torch.cat((self.role_embedding(batch.roles),
                               self.id_embedding(batch.candidate_ids)), dim=-1)
        values = self.project(batch.vectors)
        if self.feature_ffn is not None:
            values = values + self.feature_ffn(values)
        mask = ~batch.valid.unsqueeze(-1)
        return AddressedMemory(addresses.masked_fill(mask, 0),
                               values.masked_fill(mask, 0))


class JointFeatureMapMemoryEncoder(nn.Module):
    """One shared FFN fuses state, role and ID before separate attention K/V projections."""

    def __init__(self, map_dim: int, language_dim: int, max_candidates: int,
                 projection_dim: int, label_dim: int, feature_ffn_hidden_dim: int) -> None:
        super().__init__()
        if min(map_dim, language_dim, max_candidates, projection_dim, label_dim,
               feature_ffn_hidden_dim) <= 0:
            raise ValueError("joint feature dimensions must be positive")
        self.map_dim = map_dim
        self.language_dim = language_dim
        self.max_candidates = max_candidates
        self.key_dim = self.value_dim = projection_dim
        self.role_embedding = nn.Embedding(3, label_dim)
        self.id_embedding = nn.Embedding(max_candidates + 1, label_dim)
        self.feature_ffn = nn.Sequential(
            nn.Linear(map_dim + 2 * label_dim, feature_ffn_hidden_dim),
            nn.GELU(), nn.Linear(feature_ffn_hidden_dim, projection_dim))

    def forward(self, batch: MapBatch) -> AddressedMemory:
        batch.validate(self.max_candidates)
        if batch.vectors.shape[-1] != self.map_dim:
            raise ValueError("map vector dimension differs from encoder")
        inputs = torch.cat((batch.vectors, self.role_embedding(batch.roles),
                            self.id_embedding(batch.candidate_ids)), dim=-1)
        features = self.feature_ffn(inputs).masked_fill(~batch.valid.unsqueeze(-1), 0)
        return AddressedMemory(features, features)
