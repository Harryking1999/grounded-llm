"""Learn contextual map features before language-model cross-attention."""
import torch
from torch import nn
from torch.nn import functional as F

from .memory import (CURRENT, GOAL, SUCCESSOR, AddressedMapMemoryEncoder,
                     AddressedMemory, MapBatch)


class MapEncoderBlock(nn.Module):
    def __init__(self, dimension, heads, hidden):
        super().__init__()
        if dimension % heads:
            raise ValueError('Map encoder dimension must be divisible by heads')
        self.heads = heads
        self.attention_norm = nn.LayerNorm(dimension)
        self.qkv = nn.Linear(dimension, 3 * dimension)
        self.output = nn.Linear(dimension, dimension)
        self.ffn_norm = nn.LayerNorm(dimension)
        self.ffn = nn.Sequential(nn.Linear(dimension, hidden), nn.GELU(),
                                 nn.Linear(hidden, dimension))

    def forward(self, features, valid):
        batch, slots, dimension = features.shape
        qkv = self.qkv(self.attention_norm(features)).reshape(
            batch, slots, 3, self.heads, dimension // self.heads)
        query, key, value = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        read = F.scaled_dot_product_attention(query, key, value,
            attn_mask=valid[:, None, None, :], dropout_p=0.0)
        features = features + self.output(read.transpose(1, 2).reshape(batch, slots, dimension))
        features = features + self.ffn(self.ffn_norm(features))
        return features.masked_fill(~valid[..., None], 0)


class ContextualMapMemoryEncoder(AddressedMapMemoryEncoder):
    """Keep all slot addresses; contextualize jointly or in shared triplets.

    Paired mode encodes [current, goal] once for their two output slots, and
    [current, goal, candidate] independently for each candidate output. Padding
    never becomes a candidate. The same blocks encode both pairs and triplets.
    Each timeline snapshot is an independent item, as in the existing reader.
    """
    def __init__(self, map_dim, language_dim, max_candidates, projection_dim,
                 label_dim, *, layers, heads, hidden, grouping, content_keys):
        super().__init__(map_dim, language_dim, max_candidates, projection_dim, label_dim)
        if layers < 1 or grouping not in ('joint', 'paired'):
            raise ValueError('Expected positive encoder depth and joint/paired grouping')
        self.grouping = grouping
        self.content_keys = content_keys
        self.key_dim = projection_dim if content_keys else 2 * label_dim
        self.input_projection = nn.Linear(projection_dim + 2 * label_dim, projection_dim)
        self.blocks = nn.ModuleList(MapEncoderBlock(projection_dim, heads, hidden)
                                    for _ in range(layers))
        self.final_norm = nn.LayerNorm(projection_dim)

    def encode_features(self, features, valid):
        for block in self.blocks:
            features = block(features, valid)
        return self.final_norm(features).masked_fill(~valid[..., None], 0)

    def forward(self, batch: MapBatch):
        batch.validate(self.max_candidates)
        if batch.vectors.shape[-1] != self.map_dim:
            raise ValueError('Map vector dimension differs from encoder')
        addresses = torch.cat((self.role_embedding(batch.roles),
                               self.id_embedding(batch.candidate_ids)), dim=-1)
        features = self.input_projection(torch.cat((self.project(batch.vectors), addresses), dim=-1))
        features = features.masked_fill(~batch.valid[..., None], 0)
        if self.grouping == 'joint':
            encoded = self.encode_features(features, batch.valid)
        else:
            indices = torch.arange(len(features), device=features.device)
            current = ((batch.roles == CURRENT) & batch.valid).long().argmax(-1)
            goal = ((batch.roles == GOAL) & batch.valid).long().argmax(-1)
            pair = torch.stack((features[indices, current], features[indices, goal]), dim=1)
            # Pad the base pair to three slots to batch it with all real triplets.
            base = F.pad(pair, (0, 0, 0, 1))
            owners, candidates = ((batch.roles == SUCCESSOR) & batch.valid).nonzero(as_tuple=True)
            triplets = torch.cat((pair[owners], features[owners, candidates, None]), dim=1)
            grouped = torch.cat((base, triplets), dim=0)
            grouped_valid = torch.ones(grouped.shape[:2], device=features.device, dtype=torch.bool)
            grouped_valid[:len(features), 2] = False
            result = self.encode_features(grouped, grouped_valid)
            encoded = result.new_zeros(features.shape)
            encoded[indices, current] = result[:len(features), 0]
            encoded[indices, goal] = result[:len(features), 1]
            encoded[owners, candidates] = result[len(features):, 2]
        mask = ~batch.valid[..., None]
        keys = encoded if self.content_keys else addresses
        return AddressedMemory(keys.masked_fill(mask, 0), encoded.masked_fill(mask, 0))
