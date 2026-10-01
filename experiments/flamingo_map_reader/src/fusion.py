"""Flamingo-style gated cross-attention inserted before frozen decoder layers."""

from collections.abc import Iterator
from contextlib import contextmanager

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .memory import MapBatch, MapMemoryEncoder


class GatedMapCrossAttention(nn.Module):
    def __init__(self, language_dim: int, heads: int, head_dim: int = 64) -> None:
        super().__init__()
        if min(language_dim, heads, head_dim) <= 0:
            raise ValueError("attention dimensions must be positive")
        self.heads = heads
        self.head_dim = head_dim
        inner_dim = heads * head_dim
        self.query_norm = nn.LayerNorm(language_dim)
        self.memory_norm = nn.LayerNorm(language_dim)
        self.to_query = nn.Linear(language_dim, inner_dim, bias=False)
        self.to_key_value = nn.Linear(language_dim, 2 * inner_dim, bias=False)
        self.to_output = nn.Linear(inner_dim, language_dim, bias=False)
        self.gate = nn.Parameter(torch.zeros(()))

    def forward(self, hidden: Tensor, memory: Tensor, valid: Tensor) -> Tensor:
        if hidden.ndim != 3 or memory.ndim != 3 or valid.shape != memory.shape[:2]:
            raise ValueError("expected hidden [B,T,D], memory [B,N,D], valid [B,N]")
        if hidden.shape[0] != memory.shape[0] or hidden.shape[-1] != memory.shape[-1]:
            raise ValueError("hidden and memory batch/feature dimensions differ")
        if not torch.all(valid.any(dim=1)):
            raise ValueError("every sample needs at least one valid map slot")
        batch_size, text_length, _ = hidden.shape
        memory_length = memory.shape[1]
        query = self.to_query(self.query_norm(hidden))
        key, value = self.to_key_value(self.memory_norm(memory)).chunk(2, dim=-1)
        query = query.reshape(batch_size, text_length, self.heads, self.head_dim).transpose(1, 2)
        key = key.reshape(batch_size, memory_length, self.heads, self.head_dim).transpose(1, 2)
        value = value.reshape(batch_size, memory_length, self.heads, self.head_dim).transpose(1, 2)
        read = F.scaled_dot_product_attention(
            query, key, value, attn_mask=valid[:, None, None, :], dropout_p=0.0,
        )
        read = read.transpose(1, 2).reshape(batch_size, text_length, -1)
        read = self.to_output(read)
        return hidden + torch.tanh(self.gate) * read


class ConditionedDecoderLayer(nn.Module):
    """Preserve the base layer call signature while adding map attention."""

    def __init__(self, decoder_layer: nn.Module, adapter: GatedMapCrossAttention) -> None:
        super().__init__()
        self.decoder_layer = decoder_layer
        self.map_attention = adapter
        # Current Qwen3Model reads this field while preparing each layer's mask.
        self.attention_type = getattr(decoder_layer, "attention_type", None)
        self._memory: Tensor | None = None
        self._valid: Tensor | None = None

    def condition(self, memory: Tensor | None, valid: Tensor | None) -> None:
        self._memory, self._valid = memory, valid

    def forward(self, hidden_states: Tensor, *args, **kwargs):
        if self._memory is not None:
            hidden_states = self.map_attention(hidden_states, self._memory, self._valid)
        return self.decoder_layer(hidden_states, *args, **kwargs)


class MapReader(nn.Module):
    """Freeze a causal LM and insert a trainable map reader at selected layers."""

    def __init__(
        self,
        base_model: nn.Module,
        memory_encoder: MapMemoryEncoder,
        heads: int,
        head_dim: int = 64,
        every_n_layers: int = 1,
        decoder_path: str = "model.layers",
    ) -> None:
        super().__init__()
        if every_n_layers <= 0:
            raise ValueError("every_n_layers must be positive")
        base_model.requires_grad_(False)
        self.base_model = base_model
        self.memory_encoder = memory_encoder
        self.decoder_path = decoder_path
        parent = base_model
        parts = decoder_path.split(".")
        for part in parts[:-1]:
            parent = getattr(parent, part)
        original = getattr(parent, parts[-1])
        if not isinstance(original, nn.ModuleList):
            raise TypeError("decoder_path must point to an nn.ModuleList")
        wrapped = nn.ModuleList()
        self.conditioned_layers: list[ConditionedDecoderLayer] = []
        for index, layer in enumerate(original):
            if (index + 1) % every_n_layers == 0:
                adapter = GatedMapCrossAttention(
                    memory_encoder.combine.out_features, heads, head_dim,
                )
                layer = ConditionedDecoderLayer(layer, adapter)
                self.conditioned_layers.append(layer)
            wrapped.append(layer)
        if not self.conditioned_layers:
            raise ValueError("selected interval inserts no cross-attention layers")
        setattr(parent, parts[-1], wrapped)

    @contextmanager
    def conditioned(self, batch: MapBatch) -> Iterator[None]:
        if getattr(self.base_model, "is_gradient_checkpointing", False):
            raise RuntimeError("gradient checkpointing needs map conditioning during backward")
        memory = self.memory_encoder(batch)
        for layer in self.conditioned_layers:
            layer.condition(memory, batch.valid)
        try:
            yield
        finally:
            for layer in self.conditioned_layers:
                layer.condition(None, None)

    def forward(self, map_batch: MapBatch, **model_inputs):
        with self.conditioned(map_batch):
            return self.base_model(**model_inputs)

    def generate(self, map_batch: MapBatch, **generation_inputs):
        with self.conditioned(map_batch):
            return self.base_model.generate(**generation_inputs)

    def adapter_state_dict(self) -> dict:
        return {
            "memory_encoder": self.memory_encoder.state_dict(),
            "layers": [layer.map_attention.state_dict() for layer in self.conditioned_layers],
        }

    def load_adapter_state_dict(self, state: dict) -> None:
        if len(state["layers"]) != len(self.conditioned_layers):
            raise ValueError("adapter checkpoint has a different layer count")
        self.memory_encoder.load_state_dict(state["memory_encoder"])
        for layer, weights in zip(self.conditioned_layers, state["layers"]):
            layer.map_attention.load_state_dict(weights)
