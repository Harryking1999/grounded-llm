"""Check that the diagnostic measures sample means, not pooled token means."""

import unittest

import numpy as np
import torch
from torch.nn import functional as F

from experiments.flamingo_map_reader.src.diagnose_sample_gradients import batch_measurement


class SampleGradientTest(unittest.TestCase):
    def test_mean_sample_gradient_matches_mean_loss_with_unequal_token_counts(self):
        weight = torch.tensor([[0.2, -0.5], [0.4, 0.1]], requires_grad=True)
        examples = [(torch.tensor([[1.0, 0.0]]), torch.tensor([0])),
                    (torch.tensor([[0.0, 1.0], [1.0, 1.0], [-1.0, 0.0]]), torch.tensor([1, 0, 1]))]
        losses = [F.cross_entropy(features @ weight, targets) for features, targets in examples]
        gradients = torch.stack([torch.autograd.grad(loss, weight, retain_graph=True)[0].flatten()
                                 for loss in losses])
        actual = torch.autograd.grad(torch.stack(losses).mean(), weight)[0]
        measured = batch_measurement((gradients @ gradients.T).numpy(), [0, 1], 0.1)
        self.assertAlmostEqual(measured["raw_mean_norm"], actual.norm().item(), places=6)
        self.assertAlmostEqual(sum(measured["projection_shares"]), 1.0, places=6)
        weight.grad = actual.clone()
        before = torch.nn.utils.clip_grad_norm_([weight], 0.1)
        self.assertAlmostEqual(measured["raw_mean_norm"], before.item(), places=6)
        torch.testing.assert_close(weight.grad, actual * measured["clip_scale"])
        pooled = F.cross_entropy(torch.cat([x for x, _ in examples]) @ weight,
                                 torch.cat([y for _, y in examples]))
        pooled_gradient = torch.autograd.grad(pooled, weight)[0]
        self.assertFalse(torch.allclose(actual, pooled_gradient))

    def test_opposed_samples_cancel_before_clipping(self):
        vectors = np.array([[3.0, 0.0], [-3.0, 0.0], [0.0, 2.0], [0.0, 2.0]])
        measured = batch_measurement(vectors @ vectors.T, [0, 1, 2, 3], 1.0)
        self.assertEqual(measured["raw_mean_norm"], 1.0)
        np.testing.assert_allclose(measured["projection_shares"], [0, 0, 0.5, 0.5])


if __name__ == "__main__":
    unittest.main()
