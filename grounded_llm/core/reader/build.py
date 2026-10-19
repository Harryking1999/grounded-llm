"""Construct a reader from the explicit map configuration."""
from .fusion import MapReader
from .memory import AddressedMapMemoryEncoder, JointFeatureMapMemoryEncoder, MapMemoryEncoder

def build_reader(base, config, *, sequence_mean_loss=False, cache_generation=False):
    spec = config["map"]
    mode = spec.get("memory_mode", "joint")
    if mode == "joint":
        # Deprecated fallback, reached only by configs that predate addressed memory
        # or by a config that omits memory_mode. Current configs set it explicitly.
        encoder = MapMemoryEncoder
    elif mode == "address_key_state_value":
        encoder = AddressedMapMemoryEncoder
    elif mode == "joint_feature_kv":
        encoder = JointFeatureMapMemoryEncoder
    else:
        raise ValueError(f"Unknown map memory mode: {mode}")
    memory = encoder(spec["state_dim"], base.config.hidden_size,
                     spec["maximum_candidates"], spec["projection_dim"],
                     spec["role_and_id_dim"], **({"feature_ffn_hidden_dim":
                         spec.get("feature_ffn_hidden_dim", 0)} if mode != "joint" else {}))
    return MapReader(base, memory, spec["attention_heads"], spec["attention_head_dim"],
                      spec["cross_attention_every_n_layers"],
                      # Both deprecated; the current configs set neither. Read through
                      # only so the archived runs that did stay reproducible.
                      fixed_gate_tanh=spec.get("fixed_gate_tanh"),
                      value_scale=spec.get("value_scale", 1.0),
                      checkpoint_layers=config["training"].get("checkpoint_layers", False),
                      loss_chunk_tokens=config["training"].get("loss_chunk_tokens", 0),
                      sequence_mean_loss=sequence_mean_loss, cache_generation=cache_generation)
