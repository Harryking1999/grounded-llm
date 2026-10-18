import unittest
import numpy as np
import torch
from experiments.puzzle8_qmap.src.policy_audit import resolve, cycle_nodes
from experiments.puzzle8_qmap.src.cycle_repair import exit_margins


class PolicyAuditTest(unittest.TestCase):
    def test_long_paths_cycles_and_budget(self):
        pointer = np.array([0, 0, 1, 2, 5, 4, 5, 8, 9, 7])
        solved, within, length = resolve(pointer, 2)
        np.testing.assert_array_equal(solved, [1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
        np.testing.assert_array_equal(within, [1, 1, 1, 0, 0, 0, 0, 0, 0, 0])
        np.testing.assert_array_equal(length, [0, 1, 2, 3, -1, -1, -1, -1, -1, -1])
        np.testing.assert_array_equal(cycle_nodes(pointer), [0, 0, 0, 0, 1, 1, 0, 1, 1, 1])
        with self.assertRaises(ValueError):
            resolve([1, 0], 2)

    def test_exit_loss_pushes_correct_neighbor_and_masks_illegal(self):
        scores = torch.tensor([[3., 1., -100.], [1., 3., -100.]], requires_grad=True)
        parent = torch.tensor([2., 2.], requires_grad=True)
        truth = torch.tensor([[1, 3, 0], [1, 3, 0]])
        valid = torch.tensor([[True, True, False], [True, True, False]])
        rank, progress = exit_margins(scores, parent, truth, torch.tensor([2, 2]), valid, .5)
        (rank + progress).backward()
        self.assertGreater(scores.grad[0, 0], 0)
        self.assertLess(scores.grad[0, 1], 0)
        self.assertEqual(float(scores.grad[:, 2].abs().sum()), 0)
        self.assertEqual(float(scores.grad[1].abs().sum()), 0)
        with self.assertRaises(ValueError):
            exit_margins(scores, parent, truth, torch.tensor([0, 0]), valid, .5)


if __name__ == "__main__":
    unittest.main()
