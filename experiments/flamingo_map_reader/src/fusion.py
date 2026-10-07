"""Flamingo-style gated cross-attention inserted before frozen decoder layers.

Two knobs here are deprecated and are left at their defaults by both current
configs (path uses `address_key_state_value`, blocks retraining uses
`joint_feature_kv`): `value_scale` (state-value scale
calibration) and `fixed_gate_tanh` (freezing the residual gate). They survive
only so the archived runs that used them stay reproducible; see
../results/archive/abandoned_readout_variants.md.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from math import atanh

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.checkpoint import checkpoint

from .memory import (AddressedMapMemoryEncoder, AddressedMemory, JointFeatureMapMemoryEncoder, MapBatch,
                     MapMemoryEncoder, MapTimeline)


class GatedMapCrossAttention(nn.Module):
    def __init__(self, language_dim: int, heads: int, head_dim: int = 64,
                 key_dim: int | None = None, value_dim: int | None = None,
                 # Deprecated: calibrated the state-value branch to a shared scale.
                 # Only defined for addressed memory; current configs leave it at 1.0.
                 value_scale: float = 1.0) -> None:
        super().__init__()
        if min(language_dim, heads, head_dim) <= 0:
            raise ValueError("attention dimensions must be positive")
        if (key_dim is None) != (value_dim is None):
            raise ValueError("key and value dimensions must be specified together")
        if key_dim is not None and min(key_dim, value_dim) <= 0:
            raise ValueError("key and value dimensions must be positive")
        if value_scale <= 0 or not float(value_scale) < float("inf"):
            raise ValueError("value scale must be finite and positive")
        if key_dim is None and value_scale != 1.0:
            raise ValueError("value scale is only defined for addressed memory")
        self.heads = heads
        self.head_dim = head_dim
        self.addressed = key_dim is not None
        self.value_scale = float(value_scale)
        inner_dim = heads * head_dim
        self.query_norm = nn.LayerNorm(language_dim)
        self.to_query = nn.Linear(language_dim, inner_dim, bias=False)
        if self.addressed:
            self.key_norm = nn.LayerNorm(key_dim)
            self.to_key = nn.Linear(key_dim, inner_dim, bias=False)
            self.to_value = nn.Linear(value_dim, inner_dim, bias=False)
        else:
            self.memory_norm = nn.LayerNorm(language_dim)
            self.to_key_value = nn.Linear(language_dim, 2 * inner_dim, bias=False)
        self.to_output = nn.Linear(inner_dim, language_dim, bias=False)
        self.gate = nn.Parameter(torch.zeros(()))

    def project_memory(self, memory: Tensor | AddressedMemory) -> tuple[Tensor, Tensor]:
        keys = memory.keys if self.addressed else memory
        values = memory.values if self.addressed else memory
        if keys.ndim == 4:
            keys = keys.reshape(keys.shape[0], -1, keys.shape[-1])
            values = values.reshape(values.shape[0], -1, values.shape[-1])
        if self.addressed:
            key = self.to_key(self.key_norm(keys))
            value = self.to_value(values) * self.value_scale
        else:
            key, value = self.to_key_value(self.memory_norm(keys)).chunk(2, dim=-1)
        shape = (keys.shape[0], keys.shape[1], self.heads, self.head_dim)
        return key.reshape(shape).transpose(1, 2), value.reshape(shape).transpose(1, 2)

    def forward(self, hidden: Tensor, memory: Tensor | AddressedMemory, valid: Tensor,
                token_map_ids: Tensor | None = None,
                projected_key_value: tuple[Tensor, Tensor] | None = None) -> Tensor:
        if isinstance(memory, AddressedMemory) != self.addressed:
            raise TypeError("map memory and attention mode differ")
        keys = memory.keys if self.addressed else memory
        values = memory.values if self.addressed else memory
        if (hidden.ndim != 3 or keys.ndim not in (3, 4) or
                valid.shape != keys.shape[:-1] or
                values.shape[:-1] != keys.shape[:-1]):
            raise ValueError("invalid text or map memory shapes")
        if hidden.shape[0] != keys.shape[0] or (
                not self.addressed and hidden.shape[-1] != keys.shape[-1]):
            raise ValueError("hidden and memory batch/feature dimensions differ")
        if projected_key_value is None and not torch.all(valid.reshape(valid.shape[0], -1).any(dim=1)):
            raise ValueError("every sample needs at least one valid map slot")
        batch_size, text_length, _ = hidden.shape
        if keys.ndim == 4:
            if token_map_ids is None or token_map_ids.shape != (batch_size, text_length):
                raise ValueError("timeline memory needs one map ID per text token")
            snapshots, slots = keys.shape[1:3]
            if projected_key_value is None and not torch.all((token_map_ids >= 0) & (token_map_ids < snapshots)):
                raise ValueError("text token refers to a missing map snapshot")
            snapshot_ids = torch.arange(snapshots, device=keys.device).repeat_interleave(slots)
            slot_mask = valid.reshape(batch_size, -1)
            attention_mask = (
                slot_mask[:, None, :] &
                (token_map_ids[:, :, None] == snapshot_ids[None, None, :])
            )[:, None, :, :]
        else:
            if token_map_ids is not None:
                raise ValueError("single-map memory must not have token map IDs")
            attention_mask = valid[:, None, None, :]
        query = self.to_query(self.query_norm(hidden))
        key, value = projected_key_value if projected_key_value is not None else self.project_memory(memory)
        query = query.reshape(batch_size, text_length, self.heads, self.head_dim).transpose(1, 2)
        read = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0,
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
        self._memory: Tensor | AddressedMemory | None = None
        self._valid: Tensor | None = None
        self._token_map_ids: Tensor | None = None
        self._generated_suffix = False
        self.checkpoint_training = False
        self.cache_generation = False
        self._projected_key_value = None

    def condition(self, memory: Tensor | AddressedMemory | None, valid: Tensor | None,
                  token_map_ids: Tensor | None = None,
                  generated_suffix: bool = False) -> None:
        self._memory, self._valid = memory, valid
        self._token_map_ids = token_map_ids
        self._generated_suffix = generated_suffix
        self._projected_key_value = None
        if (memory is not None and generated_suffix and self.cache_generation and
                not self.training and not torch.is_grad_enabled()):
            self._projected_key_value = self.map_attention.project_memory(memory)

    def forward(self, hidden_states: Tensor, *args, **kwargs):
        if self._memory is not None:
            token_map_ids = self._token_map_ids
            if token_map_ids is not None and hidden_states.shape[1] != token_map_ids.shape[1]:
                if not self._generated_suffix:
                    raise ValueError("text length differs from its map timeline")
                if hidden_states.shape[1] == 1:
                    token_map_ids = token_map_ids[:, -1:]
                elif hidden_states.shape[1] > token_map_ids.shape[1]:
                    suffix = token_map_ids[:, -1:].expand(-1,
                        hidden_states.shape[1] - token_map_ids.shape[1])
                    token_map_ids = torch.cat((token_map_ids, suffix), dim=1)
                else:
                    raise ValueError("generation supplied an unexpected text prefix")
            memory, valid = self._memory, self._valid
            if self.checkpoint_training and self.training and torch.is_grad_enabled():
                # Capture this forward's map, since the conditioning context is
                # cleared before backward recomputes the layer.
                def conditioned_forward(hidden):
                    hidden = self.map_attention(hidden, memory, valid, token_map_ids)
                    return self.decoder_layer(hidden, *args, **kwargs)
                return checkpoint(conditioned_forward, hidden_states, use_reentrant=False)
            hidden_states = self.map_attention(hidden_states, memory, valid, token_map_ids,
                                               projected_key_value=self._projected_key_value)
        return self.decoder_layer(hidden_states, *args, **kwargs)


class MapReader(nn.Module):
    """Freeze a causal LM and insert a trainable map reader at selected layers."""

    def __init__(
        self,
        base_model: nn.Module,
        memory_encoder: MapMemoryEncoder | AddressedMapMemoryEncoder | JointFeatureMapMemoryEncoder,
        heads: int,
        head_dim: int = 64,
        every_n_layers: int = 1,
        decoder_path: str = "model.layers",
        # Deprecated: freezes the cross-attention gate at a constant instead of
        # letting it train. Only the archived pilot_path256_fixed_gate run set it.
        fixed_gate_tanh: float | None = None,
        value_scale: float = 1.0,
        checkpoint_layers: bool = False,
        loss_chunk_tokens: int = 0,
        sequence_mean_loss: bool = False,
        cache_generation: bool = False,
    ) -> None:
        super().__init__()
        if every_n_layers <= 0:
            raise ValueError("every_n_layers must be positive")
        if fixed_gate_tanh is not None and not 0 < fixed_gate_tanh < 1:
            raise ValueError("fixed_gate_tanh must be between zero and one")
        base_model.requires_grad_(False)
        self.base_model = base_model
        self.memory_encoder = memory_encoder
        self.loss_chunk_tokens = loss_chunk_tokens
        if sequence_mean_loss and loss_chunk_tokens <= 0:
            raise ValueError("Sequence-mean loss requires chunked supervised-token CE")
        self.sequence_mean_loss = sequence_mean_loss
        self.cache_generation = cache_generation
        addressed = isinstance(memory_encoder, (AddressedMapMemoryEncoder, JointFeatureMapMemoryEncoder))
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
                language_dim = (memory_encoder.language_dim if addressed else
                                memory_encoder.combine.out_features)
                adapter = GatedMapCrossAttention(language_dim, heads, head_dim,
                    key_dim=memory_encoder.key_dim if addressed else None,
                    value_dim=memory_encoder.value_dim if addressed else None,
                    value_scale=value_scale)
                if fixed_gate_tanh is not None:
                    adapter.gate.data.fill_(atanh(fixed_gate_tanh))
                    adapter.gate.requires_grad_(False)
                layer = ConditionedDecoderLayer(layer, adapter)
                layer.checkpoint_training = checkpoint_layers
                layer.cache_generation = cache_generation
                self.conditioned_layers.append(layer)
            wrapped.append(layer)
        if not self.conditioned_layers:
            raise ValueError("selected interval inserts no cross-attention layers")
        setattr(parent, parts[-1], wrapped)

    @contextmanager
    def conditioned(self, batch: MapBatch | MapTimeline,
                    generated_suffix: bool = False) -> Iterator[None]:
        if getattr(self.base_model, "is_gradient_checkpointing", False):
            raise RuntimeError("gradient checkpointing needs map conditioning during backward")
        if isinstance(batch, MapTimeline):
            batch.validate(self.memory_encoder.max_candidates, batch.token_map_ids.shape[1])
            counts = batch.snapshot_counts or (batch.snapshots.vectors.shape[0],)
            encoded = self.memory_encoder(batch.snapshots)
            if isinstance(encoded, AddressedMemory):
                memory = AddressedMemory(
                    pad_sequence(encoded.keys.split(counts), batch_first=True),
                    pad_sequence(encoded.values.split(counts), batch_first=True))
            else:
                memory = pad_sequence(encoded.split(counts), batch_first=True)
            valid = pad_sequence(batch.snapshots.valid.split(counts), batch_first=True)
            token_map_ids = batch.token_map_ids
        else:
            memory = self.memory_encoder(batch)
            valid = batch.valid
            token_map_ids = None
        if generated_suffix and self.cache_generation:
            # This memory and timeline stay fixed for the entire generate call.
            # Validate once rather than synchronize CUDA at every layer/token.
            if not torch.all(valid.reshape(valid.shape[0], -1).any(dim=1)):
                raise ValueError("every sample needs at least one valid map slot")
            if token_map_ids is not None and not torch.all(
                    (token_map_ids >= 0) & (token_map_ids < valid.shape[1])):
                raise ValueError("text token refers to a missing map snapshot")
        for layer in self.conditioned_layers:
            layer.condition(memory, valid, token_map_ids, generated_suffix)
        try:
            yield
        finally:
            for layer in self.conditioned_layers:
                layer.condition(None, None)

    def forward(self, map_batch: MapBatch | MapTimeline, **model_inputs):
        with self.conditioned(map_batch):
            if self.loss_chunk_tokens and self.training and "labels" in model_inputs:
                from transformers.modeling_outputs import CausalLMOutputWithPast
                labels = model_inputs.pop("labels")[:, 1:]
                outputs = self.base_model.model(**model_inputs)
                selected = labels != -100
                hidden = outputs.last_hidden_state[:, :-1][selected]
                targets = labels[selected]
                if not len(targets):
                    raise ValueError("Batch has no supervised next-token targets")
                weights = None
                if self.sequence_mean_loss:
                    counts = selected.sum(dim=1)
                    if bool((counts == 0).any()):
                        raise ValueError("Sequence-mean loss requires supervision in every example")
                    weights = (1 / (len(counts) * counts.float()))[:, None].expand_as(labels)[selected]
                def token_loss(features, target, weight):
                    if weight is not None:
                        return (F.cross_entropy(self.base_model.lm_head(features).float(), target,
                                                reduction="none") * weight).sum()
                    return F.cross_entropy(self.base_model.lm_head(features).float(), target, reduction="sum")
                chunks = hidden.split(self.loss_chunk_tokens)
                weight_chunks = weights.split(self.loss_chunk_tokens) if weights is not None else [None] * len(chunks)
                losses = [checkpoint(token_loss, features, target, weight, use_reentrant=False)
                          for features, target, weight in zip(chunks, targets.split(self.loss_chunk_tokens), weight_chunks)]
                loss = torch.stack(losses).sum()
                return CausalLMOutputWithPast(loss=loss if self.sequence_mean_loss else loss / len(targets))
            return self.base_model(**model_inputs)

    def generate(self, map_batch: MapBatch | MapTimeline, **generation_inputs):
        with self.conditioned(map_batch, generated_suffix=True):
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
