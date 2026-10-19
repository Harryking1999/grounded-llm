import unittest
import torch

from experiments.blocks_distance_map.src.directed_audit import closed_chains
from experiments.blocks_distance_map.src.model import distance


class DirectedAuditTest(unittest.TestCase):
    def test_actual_metric_identity_and_constant_reverse_contradiction(self):
        z = torch.randn(12, 128, dtype=torch.float64)
        d = distance(z[:, None], z[None, :], "directed_sum")
        phi = z.sum(-1) / 128 ** .5
        torch.testing.assert_close(d - d.T, phi[:, None] - phi[None, :])
        labels = {(7, 3): (1, 2), (3, 1): (1, 2), (7, 1): (2, 1),
                  (3, 7): (-1, 1), (1, 3): (-1, 1), (1, 7): (-1, 1)}
        witness, pairs = closed_chains(labels)
        self.assertEqual(witness, [(7, 3, 1)])
        self.assertEqual(len(pairs), 6)
        del labels[(1, 7)]
        self.assertEqual(closed_chains(labels), ([], set()))

    def test_single_clear_target_has_no_six_pair_obstruction(self):
        truth = torch.arange(20, dtype=torch.float64)
        z = (1 + truth[:, None] / 128 ** .5).repeat(1, 128)
        self.assertTrue(((z > 0) & (z < 13)).all())
        torch.testing.assert_close(distance(z, z[0], "directed_sum"), truth)
