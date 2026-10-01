"""Targeted checks for map slots, masking, gating, and frozen-layer wrapping."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from torch import nn

from experiments.flamingo_map_reader.src.fusion import GatedMapCrossAttention, MapReader
from experiments.flamingo_map_reader.src.checkpoint import load_checkpoint, save_checkpoint
from experiments.flamingo_map_reader.src.memory import MapBatch, MapMemoryEncoder


class FakeLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([nn.Linear(8, 8) for _ in range(3)])

    def forward(self, hidden_states):
        for layer in self.model.layers:
            hidden_states = layer(hidden_states)
        return hidden_states


def batch():
    return MapBatch(
        vectors=torch.tensor([[[0., 0.], [1., 0.], [0., 1.], [999., 999.]]]),
        roles=torch.tensor([[0, 1, 2, 0]]),
        candidate_ids=torch.tensor([[0, 0, 1, 0]]),
        valid=torch.tensor([[True, True, True, False]]),
    )


class InterfaceTest(unittest.TestCase):
    def test_slot_validation_rejects_unidentified_successor(self):
        bad = batch()
        bad.candidate_ids[0, 2] = 0
        with self.assertRaisesRegex(ValueError, "positive candidate ID"):
            bad.validate(4)

    def test_padding_does_not_change_readout(self):
        torch.manual_seed(3)
        encoder = MapMemoryEncoder(2, 8, 4, 5, 3)
        attention = GatedMapCrossAttention(8, 2)
        attention.gate.data.fill_(1)
        first = batch()
        second = batch()
        second.vectors[0, 3] = torch.tensor([-500., 800.])
        hidden = torch.randn(1, 2, 8)
        self.assertTrue(torch.allclose(
            attention(hidden, encoder(first), first.valid),
            attention(hidden, encoder(second), second.valid),
        ))

    def test_zero_gate_preserves_frozen_lm_and_trains_gate(self):
        torch.manual_seed(7)
        base = FakeLM()
        hidden = torch.randn(1, 2, 8)
        original = base(hidden).detach().clone()
        reader = MapReader(base, MapMemoryEncoder(2, 8, 4, 5, 3), heads=2)
        output = reader(map_batch=batch(), hidden_states=hidden)
        self.assertTrue(torch.equal(output, original))
        self.assertFalse(any(p.requires_grad for layer in reader.conditioned_layers
                             for p in layer.decoder_layer.parameters()))
        output.sum().backward()
        self.assertTrue(all(layer.map_attention.gate.grad is not None
                            for layer in reader.conditioned_layers))

    def test_layer_interval_and_adapter_checkpoint(self):
        first = MapReader(FakeLM(), MapMemoryEncoder(2, 8, 4, 5, 3), 2, every_n_layers=2)
        self.assertEqual(len(first.conditioned_layers), 1)
        state = first.adapter_state_dict()
        second = MapReader(FakeLM(), MapMemoryEncoder(2, 8, 4, 5, 3), 2, every_n_layers=2)
        second.load_adapter_state_dict(state)
        self.assertTrue(torch.equal(first.memory_encoder.project.weight,
                                    second.memory_encoder.project.weight))

    def test_training_checkpoint_restores_adapter_and_step(self):
        reader = MapReader(FakeLM(), MapMemoryEncoder(2, 8, 4, 5, 3), 2)
        parameters = [p for p in reader.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(parameters, lr=0.01)
        reader(map_batch=batch(), hidden_states=torch.ones(1, 2, 8)).sum().backward()
        optimizer.step()
        with TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint_000001.pt"
            save_checkpoint(path, reader, optimizer, 1, {"name": "smoke"},
                            data_state={"next_sample": 8})
            saved_gate = reader.conditioned_layers[0].map_attention.gate.detach().clone()
            reader.conditioned_layers[0].map_attention.gate.data.add_(3)
            step, data_state = load_checkpoint(path, reader, optimizer,
                                               {"name": "smoke"})
            self.assertEqual(step, 1)
            self.assertEqual(data_state, {"next_sample": 8})
            self.assertTrue(torch.equal(saved_gate,
                                        reader.conditioned_layers[0].map_attention.gate))


if __name__ == "__main__":
    unittest.main()
