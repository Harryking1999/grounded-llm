"""Cross-task integration and compatibility at the new public boundaries."""
import copy
import json
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from grounded_llm.core.qmap.model import StateEncoder
from grounded_llm.core.reader.batching import collate_examples
from grounded_llm.core.reader.build import build_reader
from grounded_llm.core.reader.memory import MapBatch
from grounded_llm.core.reader.state_q import puzzle_q_batch
from grounded_llm.core.sft.modeling import TokenLM
from grounded_llm.core.sft.dataset import PreparedRoutes
from grounded_llm.core.sft.training import train_update
from grounded_llm.core.sft.transcript import EncodedTrajectory
from grounded_llm.tasks.blocks.model import BoardEncoder
from grounded_llm.tasks.blocks.qmap import FrozenBoardMap, blocks_step
from grounded_llm.tasks.blocks.rules import PLACEMENTS
from grounded_llm.tasks.cube2.environment import Cube2
from grounded_llm.tasks.cube2.model import MapReader as CubeReader, QMap as CubeQ
from grounded_llm.tasks.hanoi.environment import Hanoi
from grounded_llm.tasks.hanoi.model import MapReader as HanoiReader, QMap as HanoiQ
from grounded_llm.tasks.path.environment import GraphEnvironment, action_catalog
from grounded_llm.tasks.path.qmap import path_q_step
from grounded_llm.tasks.path.reader import FrozenPathMap
from grounded_llm.tasks.puzzle8.environment import Puzzle8
from grounded_llm.tasks.puzzle8.model import MapReader as PuzzleReader, QMap as PuzzleQ


def tiny_lm():
    return Qwen2ForCausalLM(Qwen2Config(vocab_size=32, hidden_size=16,
        intermediate_size=32, num_hidden_layers=2, num_attention_heads=2,
        num_key_value_heads=1, max_position_embeddings=32, pad_token_id=0))


def test_adam_calibration_preserves_weights_and_global_lr_schedule():
    from grounded_llm.core.sft.schedule import learning_rate_factor
    torch.manual_seed(9)
    model = TokenLM(tiny_lm(), 8)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=.01)
    recipe = dict(warmup_fraction=.05, cosine_minimum_ratio=.1)
    for group in optimizer.param_groups:
        group['initial_lr'] = 2e-5
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: learning_rate_factor(s, 250000, recipe), last_epoch=149999)
    expected_lr = 2e-5 * learning_rate_factor(150000, 250000, recipe)
    assert scheduler.last_epoch == 150000
    assert optimizer.param_groups[0]['lr'] == expected_lr
    before = {name: p.detach().clone() for name, p in model.named_parameters()}
    batch = (None, dict(input_ids=torch.tensor([[2, 3, 4]]),
                        attention_mask=torch.ones(1, 3, dtype=torch.long),
                        labels=torch.tensor([[-100, 3, 4]])))
    optimizer.param_groups[0]['lr'] = 0.0
    for _ in range(3):
        train_update(model, [batch], optimizer, None, device='cpu',
                     interface=False, gradient_clip_norm=1.)
    for name, p in model.named_parameters():
        assert torch.equal(p, before[name])
    states = [state for state in optimizer.state.values() if state]
    assert states and all(state['step'] == 3 for state in states)
    assert any(state['exp_avg'].abs().sum() > 0 for state in states)
    assert any(state['exp_avg_sq'].sum() > 0 for state in states)
    assert scheduler.last_epoch == 150000
    optimizer.param_groups[0]['lr'] = expected_lr
    train_update(model, [batch], optimizer, scheduler, device='cpu',
                 interface=False, gradient_clip_norm=1.)
    assert scheduler.last_epoch == 150001
    assert all(state['step'] == 4 for state in states)
    assert any(not torch.equal(p, before[name]) for name, p in model.named_parameters())


def test_five_tasks_load_existing_checkpoint_formats_and_share_sft_update(tmp_path):
    torch.set_num_threads(2)
    torch.manual_seed(41)
    readers, steps = [], []
    # Preserve the board/path formats and the two existing puzzle formats.
    board = BoardEncoder(8, 16, 8, 'directed_sum')
    path = tmp_path / 'blocks.pt'
    torch.save(dict(config={'model': {'state_dim': 8, 'encoder_hidden_dim': 16}},
                    model=board.state_dict(), cap=8, metric=board.metric), path)
    reader = FrozenBoardMap.load(path)
    readers.append(reader)
    step = blocks_step(reader, PLACEMENTS[0][0], 0)
    steps.append(step)
    torch.testing.assert_close(reader.encode([step.current]), board.encode(reader.features([step.current])))

    adjacency = ~np.eye(3, dtype=bool)
    env = GraphEnvironment(adjacency, action_catalog(adjacency)[0])
    encoder = StateEncoder(8, 16, 2, 'l1', input_dim=3)
    path = tmp_path / 'path.pt'
    torch.save(dict(model_type='state_q_path', model=encoder.state_dict(), cap=2,
        adjacency=torch.tensor(adjacency), config={'model': {'state_dim': 8,
            'encoder_hidden_dim': 16, 'metric': 'l1'}}), path)
    reader = FrozenPathMap.load(path)
    readers.append(reader)
    steps.append(path_q_step(env, reader, 1, 0))
    torch.testing.assert_close(reader.encode([1]), encoder.encode(torch.eye(3)[[1]]))

    for name, env, model_class, reader_class in (
        ('hanoi', Hanoi(3), HanoiQ, HanoiReader),
        ('cube2', Cube2(metric='QTM_URF_6'), CubeQ, CubeReader),
        ('puzzle8', Puzzle8(), PuzzleQ, PuzzleReader),
    ):
        spec = dict(dimension=8, hidden=16, metric='l1')
        model = model_class(**spec) if name == 'puzzle8' else model_class(env.feature_dim, **spec, cap=env.diameter)
        saved = dict(model=model.state_dict(), actions=list(env.actions), task=env.name)
        if name == 'puzzle8':
            saved.update(model_type='puzzle8_pure_q', config={'model': spec})
        else:
            saved.update(model_type='pure_q_symmetric_blocks', model_config=spec, cap=env.diameter)
        path = tmp_path / (name + '.pt')
        torch.save(saved, path)
        reader = reader_class(env, path)
        readers.append(reader)
        batch, actions = puzzle_q_batch(reader, 1, 0)
        steps.append(SimpleNamespace(map_batch=batch, candidate_actions=actions))
        torch.testing.assert_close(reader.encode([1, 0]), model.encode(torch.tensor(env.features([1, 0]))))
        children = env.successors([1])[0]
        legal = np.flatnonzero(children >= 0)
        expected = int(legal[reader.scores(children[legal], [0]).argmin()])
        assert reader.choose(1, 0) == (env.actions[expected], int(children[expected]))

    config = dict(map=dict(memory_mode='address_key_state_value', state_dim=8,
        maximum_candidates=max(len(s.candidate_actions) for s in steps), projection_dim=8,
        role_and_id_dim=4, attention_heads=2, attention_head_dim=4, cross_attention_every_n_layers=1),
        training=dict(loss_chunk_tokens=8))
    model = build_reader(tiny_lm(), config)
    for layer in model.conditioned_layers:
        layer.map_attention.gate.data.fill_(.5)
    before = model.memory_encoder.project.weight.detach().clone()
    examples = [(EncodedTrajectory([2, 3, 4], [-100, 3, 4], [0, 0, 0], 2), [s]) for s in steps]
    data_root = tmp_path / 'data'
    (data_root / 'trajectories').mkdir(parents=True)
    rows = []
    for index, (encoded, task_steps) in enumerate(examples):
        name = f'{index}.pt'
        torch.save({'map': {'encoded': encoded, 'steps': task_steps}}, data_root / 'trajectories' / name)
        rows.append({'prepared_file': name})
    (data_root / 'dataset.json').write_text(json.dumps({'rows': rows}), encoding='utf-8')
    data = PreparedRoutes(data_root, real_map=True)
    microbatches = [collate_examples([data[index]], 0) for index in range(len(data))]
    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=.1)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.)
    loss, grad, count = train_update(model, microbatches, optimizer, scheduler,
        device='cpu', interface=True, gradient_clip_norm=1.)
    assert count == 10 and torch.isfinite(loss) and torch.isfinite(grad)
    assert not torch.equal(before, model.memory_encoder.project.weight)
    assert all(p.grad is None for r in readers for p in r.encoder.parameters())
    assert all(p.grad is None for p in model.base_model.parameters() if not p.requires_grad)


def test_accumulated_update_matches_combined_supervised_token_mean():
    torch.manual_seed(7)
    first = TokenLM(tiny_lm(), 8)
    second = copy.deepcopy(first)
    tokens = torch.tensor([[2, 3, 4, 5], [6, 7, 8, 9]])
    labels = torch.tensor([[-100, 3, 4, 5], [-100, -100, -100, 9]])
    inputs = dict(input_ids=tokens, labels=labels, attention_mask=torch.ones_like(tokens))
    micro = [(None, {k: v[i:i+1] for k, v in inputs.items()}) for i in range(2)]
    for model, batches in ((first, micro), (second, [(None, inputs)])):
        optimizer = torch.optim.SGD(model.parameters(), lr=.01)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.)
        train_update(model, batches, optimizer, scheduler, device='cpu',
                     interface=False, gradient_clip_norm=100.)
    for left, right in zip(first.parameters(), second.parameters()):
        torch.testing.assert_close(left, right, atol=1e-7, rtol=1e-6)


def test_legacy_pickled_trajectory_loads_through_old_module_names(tmp_path):
    batch = MapBatch(torch.zeros(1, 2, 8), torch.tensor([[0, 1]]),
                     torch.zeros(1, 2, dtype=torch.long), torch.ones(1, 2, dtype=torch.bool))
    encoded = EncodedTrajectory([2, 3], [-100, 3], [0, 0], 1)
    path = tmp_path / 'old-cache.pt'
    with patch.object(MapBatch, '__module__', 'experiments.flamingo_map_reader.src.memory'), \
         patch.object(EncodedTrajectory, '__module__', 'experiments.flamingo_map_reader.src.transcript'):
        torch.save(dict(map=batch, encoded=encoded), path)
    loaded = torch.load(path, weights_only=False)
    assert isinstance(loaded['map'], MapBatch)
    assert isinstance(loaded['encoded'], EncodedTrajectory)
    assert loaded['encoded'].labels == [-100, 3]
