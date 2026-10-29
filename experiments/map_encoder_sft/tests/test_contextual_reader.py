"""Check cross-slot isolation, pairing, padding, and gradient flow to the encoder."""
from dataclasses import replace
import copy
import json
from pathlib import Path

import pytest
import torch

from grounded_llm.core.reader.build import build_reader
from grounded_llm.core.reader.memory import MapBatch
from grounded_llm.core.reader.batching import timeline_maps
from types import SimpleNamespace
from transformers import Qwen2Config, Qwen2ForCausalLM


def batch():
    return MapBatch(torch.randn(2, 5, 8),
        torch.tensor([[0, 1, 2, 2, 2], [0, 1, 2, 0, 0]]),
        torch.tensor([[0, 0, 1, 2, 3], [0, 0, 1, 0, 0]]),
        torch.tensor([[True]*5, [True, True, True, False, False]]))


def reader(grouping, content, layers=4):
    base = Qwen2ForCausalLM(Qwen2Config(vocab_size=32, hidden_size=16,
        intermediate_size=32, num_hidden_layers=2, num_attention_heads=2,
        num_key_value_heads=1, pad_token_id=0))
    config = dict(map=dict(memory_mode='joint_feature_kv' if content else 'address_key_state_value',
        state_dim=8, maximum_candidates=4, projection_dim=16, role_and_id_dim=4,
        attention_heads=2, attention_head_dim=4, cross_attention_every_n_layers=1,
        encoder=dict(layers=layers, heads=2, hidden_dim=32, grouping=grouping)),
        training=dict(checkpoint_layers=True, loss_chunk_tokens=8))
    return build_reader(base, config)


@pytest.mark.parametrize('grouping', ['joint', 'paired'])
@pytest.mark.parametrize('content', [True, False])
@pytest.mark.parametrize('layers', [2, 4])
def test_padding_slot_permutation_and_snapshot_isolation(grouping, content, layers):
    torch.manual_seed(5)
    model = reader(grouping, content, layers).eval()
    encoder = model.memory_encoder
    data = batch()
    original = encoder(data)
    assert original.values.shape == (2, 5, 16)
    assert torch.count_nonzero(original.values[1, 3:]) == 0
    changed = data.vectors.clone(); changed[1, 3:] = 10000
    torch.testing.assert_close(encoder(replace(data, vectors=changed)).values, original.values)
    changed = data.vectors.clone(); changed[1, :3] += 50
    torch.testing.assert_close(encoder(replace(data, vectors=changed)).values[0], original.values[0])
    order = torch.tensor([1, 4, 0, 2, 3])
    permuted = MapBatch(*(getattr(data, field)[:, order] for field in ('vectors','roles','candidate_ids','valid')))
    torch.testing.assert_close(encoder(permuted).values, original.values[:, order], atol=2e-6, rtol=2e-5)
    if content:
        torch.testing.assert_close(original.keys, original.values)
    else:
        changed = data.vectors + 1
        torch.testing.assert_close(encoder(replace(data, vectors=changed)).keys, original.keys)


def test_paired_candidate_has_goal_context_but_no_other_candidate_context():
    model = reader('paired', True).eval(); data = batch(); encoder = model.memory_encoder
    initial = encoder(data).values
    changed = data.vectors.clone(); changed[0, 3] += torch.arange(8)
    after = encoder(replace(data, vectors=changed)).values
    torch.testing.assert_close(after[0, [0,1,2,4]], initial[0, [0,1,2,4]])
    assert not torch.allclose(after[0, 3], initial[0, 3])
    changed = data.vectors.clone(); changed[0, 1] += torch.arange(8)
    after = encoder(replace(data, vectors=changed)).values
    assert not torch.allclose(after[0, 2:], initial[0, 2:])
    empty = MapBatch(data.vectors[:, :2], data.roles[:, :2], data.candidate_ids[:, :2], data.valid[:, :2])
    assert torch.isfinite(encoder(empty).values).all()


@pytest.mark.parametrize('grouping', ['joint','paired'])
@pytest.mark.parametrize('content', [True,False])
@pytest.mark.parametrize('full_llm', [True,False])
def test_timeline_backward_and_checkpoint_roundtrip(grouping, content, full_llm):
    torch.manual_seed(7)
    model = reader(grouping,content)
    if full_llm:
        model.base_model.requires_grad_(True)
    # The production gate starts at zero; open it here to test the whole path.
    for layer in model.conditioned_layers:
        layer.map_attention.gate.data.fill_(.2)
    data = batch()
    steps = [SimpleNamespace(candidate_actions=list(range(3 if i == 0 else 1)),
             map_batch=MapBatch(*(getattr(data,field)[i:i+1, :5 if i == 0 else 3] for field in
             ('vectors','roles','candidate_ids','valid')))) for i in range(2)]
    timeline = timeline_maps(steps, [0,0,1,1,1])
    inputs = dict(input_ids=torch.tensor([[1,2,3,4,5]]),attention_mask=torch.ones(1,5,dtype=torch.long),
                  labels=torch.tensor([[-100,2,3,4,5]]),use_cache=False)
    output = model(map_batch=timeline,**inputs); output.loss.backward()
    grad = model.memory_encoder.blocks[0].qkv.weight.grad
    assert grad is not None and grad.abs().sum() > 0
    assert (model.base_model.model.embed_tokens.weight.grad is not None) == full_llm
    state = copy.deepcopy(model.adapter_state_dict())
    other = reader(grouping,content); other.load_adapter_state_dict(state)
    for a,b in zip(model.memory_encoder.parameters(),other.memory_encoder.parameters()):
        torch.testing.assert_close(a,b)


@pytest.mark.parametrize('baseline', list(json.loads(
    (Path(__file__).parents[1] / 'configs/five_task_50k.json').read_text())['baselines']))
def test_formal_config_reaches_the_expected_encoder_and_trainable_scope(monkeypatch, baseline):
    from grounded_llm.core.sft import modeling
    config = json.loads((Path(__file__).parents[1] / 'configs/five_task_50k.json').read_text())
    base = Qwen2ForCausalLM(Qwen2Config(vocab_size=32, hidden_size=16,
        intermediate_size=32, num_hidden_layers=2, num_attention_heads=2,
        num_key_value_heads=1, pad_token_id=0))
    monkeypatch.setattr(modeling.AutoModelForCausalLM, 'from_pretrained', lambda *a, **k: base)
    model = modeling.build(config, baseline, 'unused')
    spec = config['baselines'][baseline]
    encoder = model.memory_encoder
    assert encoder.grouping == spec['map']['encoder']['grouping']
    assert len(encoder.blocks) == 4
    assert encoder.content_keys == (spec['map']['memory_mode'] == 'joint_feature_kv')
    assert all(p.requires_grad for p in encoder.parameters())
    assert base.model.embed_tokens.weight.requires_grad == spec['full_llm']
    assert base.lm_head.weight.requires_grad == spec['full_llm']
    assert len(model.conditioned_layers) == base.config.num_hidden_layers
    for layer in model.conditioned_layers:
        assert all(p.requires_grad == spec['full_llm'] for p in layer.decoder_layer.parameters())
        assert all(p.requires_grad for p in layer.map_attention.parameters())
        assert layer.map_attention.to_key.in_features == (256 if encoder.content_keys else 64)
        assert layer.map_attention.to_value.in_features == 256
    data = batch()
    memory = encoder(replace(data, vectors=torch.randn(2, 5, 128)))
    assert memory.values.shape == (2, 5, 256)


@pytest.mark.parametrize('grouping', ['joint', 'paired'])
def test_zero_gate_can_open_and_then_train_encoder(grouping):
    torch.manual_seed(17)
    model = reader(grouping, True)
    optimizer = torch.optim.SGD((p for p in model.parameters() if p.requires_grad), lr=.1)
    inputs = dict(input_ids=torch.tensor([[1,2,3,4], [4,3,2,1]]),
        attention_mask=torch.ones(2,4,dtype=torch.long),
        labels=torch.tensor([[-100,2,3,4], [-100,3,2,1]]), use_cache=False)
    data = batch()
    model(map_batch=data, **inputs).loss.backward()
    assert all(layer.map_attention.gate.grad.abs() > 0 for layer in model.conditioned_layers)
    grad = model.memory_encoder.blocks[0].qkv.weight.grad
    assert grad is not None and grad.abs().sum() == 0
    optimizer.step(); optimizer.zero_grad(set_to_none=True)
    model(map_batch=data, **inputs).loss.backward()
    assert model.memory_encoder.blocks[0].qkv.weight.grad.abs().sum() > 0


@pytest.mark.parametrize('grouping', ['joint', 'paired'])
@pytest.mark.parametrize('content', [True, False])
def test_future_snapshot_does_not_change_earlier_logits(grouping, content):
    torch.manual_seed(23)
    model = reader(grouping, content).eval()
    for layer in model.conditioned_layers:
        layer.map_attention.gate.data.fill_(.5)
    data = batch()
    from grounded_llm.core.reader.memory import MapTimeline
    ids = torch.tensor([[0,0,1,1]])
    inputs = dict(input_ids=torch.tensor([[1,2,3,4]]), use_cache=False)
    with torch.no_grad():
        initial = model(map_batch=MapTimeline(data, ids), **inputs).logits
        changed = data.vectors.clone(); changed[1, :3] += torch.arange(8)
        after = model(map_batch=MapTimeline(replace(data, vectors=changed), ids), **inputs).logits
    torch.testing.assert_close(initial[:, :2], after[:, :2])
    assert not torch.allclose(initial[:, 2:], after[:, 2:])
