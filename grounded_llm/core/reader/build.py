"""Construct a reader from the explicit map configuration."""
from .fusion import MapReader
from .memory import AddressedMapMemoryEncoder, JointFeatureMapMemoryEncoder, MapMemoryEncoder
from .contextual import ContextualMapMemoryEncoder

def build_reader(base, config, *, sequence_mean_loss=False, cache_generation=False):
    spec = config["map"]
    mode = spec.get("memory_mode", "joint")
    contextual = spec.get('encoder')
    if contextual:
        if mode not in ('address_key_state_value', 'joint_feature_kv'):
            raise ValueError('Contextual encoder requires explicit address or content K/V')
        memory = ContextualMapMemoryEncoder(spec['state_dim'], base.config.hidden_size,
            spec['maximum_candidates'], spec['projection_dim'], spec['role_and_id_dim'],
            layers=contextual['layers'], heads=contextual['heads'], hidden=contextual['hidden_dim'],
            grouping=contextual['grouping'], content_keys=mode == 'joint_feature_kv')
    elif mode == "joint":
        # Deprecated fallback, reached only by configs that predate addressed memory
        # or by a config that omits memory_mode. Current configs set it explicitly.
        encoder = MapMemoryEncoder
    elif mode == "address_key_state_value":
        encoder = AddressedMapMemoryEncoder
    elif mode == "joint_feature_kv":
        encoder = JointFeatureMapMemoryEncoder
    else:
        raise ValueError(f"Unknown map memory mode: {mode}")
    if not contextual:
        memory = encoder(spec["state_dim"], base.config.hidden_size,
                         spec["maximum_candidates"], spec["projection_dim"],
                         spec["role_and_id_dim"], **({"feature_ffn_hidden_dim":
                             spec.get("feature_ffn_hidden_dim", 0)} if mode != "joint" else {}))
    return MapReader(base, memory, spec["attention_heads"], spec["attention_head_dim"],
                      spec["cross_attention_every_n_layers"],
                      last_n_layers=spec.get("cross_attention_last_n_layers"),
                      # Both deprecated; the current configs set neither. Read through
                      # only so the archived runs that did stay reproducible.
                      fixed_gate_tanh=spec.get("fixed_gate_tanh"),
                      value_scale=spec.get("value_scale", 1.0),
                      checkpoint_layers=config["training"].get("checkpoint_layers", False),
                      loss_chunk_tokens=config["training"].get("loss_chunk_tokens", 0),
                      sequence_mean_loss=sequence_mean_loss, cache_generation=cache_generation)
