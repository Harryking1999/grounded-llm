"""Relational coordinate bounds must preserve the intended feasible geometry."""
import unittest

import torch

from experiments.cube2_qmap.src.supervision_audit import dense_upper_loss


class SupervisionTests(unittest.TestCase):
    def test_upper_bound_corrects_every_violation_without_assigning_coordinates(self):
        q = torch.tensor([[[0., 0., 0.], [3., 2., .5]]], requires_grad=True)
        pairs = torch.tensor([[0, 1]])
        loss = dense_upper_loss(q, pairs, torch.tensor([1.]))
        loss.backward()
        self.assertTrue((q.grad[0, 1, :2] > 0).all())
        self.assertEqual(float(q.grad[0, 1, 2]), 0.)
        # Translation, reflection and coordinate permutation leave the loss
        # unchanged: no anchor identity or per-coordinate answer is supplied.
        transformed = -q.detach()[..., [2, 0, 1]] + torch.tensor([5., 8., 2.])
        self.assertAlmostEqual(float(dense_upper_loss(transformed, pairs, torch.tensor([1.]))), float(loss.detach()))

    def test_exact_finite_distance_embedding_has_zero_upper_penalty(self):
        distances = torch.tensor([[0., 1., 2., 1.], [1., 0., 1., 2.],
                                  [2., 1., 0., 1.], [1., 2., 1., 0.]])
        pairs = torch.triu_indices(4, 4, offset=1).T
        self.assertEqual(float(dense_upper_loss(distances[None], pairs, distances[pairs[:, 0], pairs[:, 1]])), 0.)


if __name__ == "__main__":
    unittest.main()
