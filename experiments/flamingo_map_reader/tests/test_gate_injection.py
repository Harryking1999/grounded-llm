"""Validate residual observation against real gated attention and bf16 rounding."""

import unittest

import torch

from experiments.flamingo_map_reader.src.diagnose_gate_injection import ResidualObserver
from experiments.flamingo_map_reader.src.fusion import GatedMapCrossAttention


class GateInjectionTest(unittest.TestCase):
    @torch.no_grad()
    def test_zero_gate_and_measurement_preserve_model_behavior(self):
        torch.manual_seed(4)
        attention = GatedMapCrossAttention(8, 2, 4)
        hidden, memory = torch.randn(1, 3, 8), torch.randn(1, 4, 8)
        valid = torch.ones(1, 4, dtype=torch.bool)
        for gate in (0.0, 0.03):
            attention.gate.data.fill_(gate)
            expected = attention(hidden, memory, valid)
            observer = ResidualObserver([attention])
            observer.start_sample(torch.tensor([[False, True, True]]))
            actual = attention(hidden, memory, valid)
            stats = observer.result()["scopes"]["supervised_prediction_tokens"]["overall"]
            observer.close()
            torch.testing.assert_close(actual, expected)
            direct = (actual[:, 1:] - hidden[:, 1:]).norm() / hidden[:, 1:].norm()
            self.assertAlmostEqual(stats["realized_residual_to_hidden_rms"], direct.item(), places=6)
            self.assertEqual(stats["coordinates"], 16)
            if gate == 0:
                self.assertEqual(stats["intended_residual_to_hidden_rms"], 0.0)
                self.assertGreater(stats["ungated_read_to_hidden_rms"], 0.0)

    @torch.no_grad()
    def test_bf16_addition_can_erase_a_small_nonzero_residual(self):
        torch.manual_seed(3)
        attention = GatedMapCrossAttention(8, 2, 4)
        attention.gate.data.fill_(1e-6)
        hidden = torch.ones(1, 2, 8, dtype=torch.bfloat16)
        memory, valid = torch.randn(1, 3, 8), torch.ones(1, 3, dtype=torch.bool)
        observer = ResidualObserver([attention])
        observer.start_sample(torch.ones(1, 2, dtype=torch.bool))
        with torch.autocast("cpu", dtype=torch.bfloat16):
            actual = attention(hidden, memory, valid)
        stats = observer.result()["scopes"]["all_tokens"]["overall"]
        observer.close()
        torch.testing.assert_close(actual, hidden)
        self.assertGreater(stats["intended_residual_to_hidden_rms"], 0)
        self.assertEqual(stats["realized_residual_to_hidden_rms"], 0)


if __name__ == "__main__":
    unittest.main()
