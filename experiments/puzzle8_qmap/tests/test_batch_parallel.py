import copy
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch.nn import functional as F

from experiments.puzzle8_qmap.src.batch_parallel import BatchParallel
from experiments.puzzle8_qmap.src.model import QMap, soft_rank_loss


class BatchParallelTest(unittest.TestCase):
    def test_uneven_shards_preserve_original_loss_gradient_clip_and_adam(self):
        torch.manual_seed(19)
        torch.set_num_threads(1)
        rng = np.random.default_rng(18)
        # Float64 isolates weighting/clipping/Adam correctness from float32
        # reduction-order noise in near-zero bias gradients at random init.
        original = QMap(dimension=8, hidden=16).double()
        replica = copy.deepcopy(original)
        features = torch.rand(19, 81, dtype=torch.float64)
        rows = np.column_stack((rng.integers(19, size=(17, 2)), rng.integers(32, size=17)))
        rows[0, :2] = 0  # Equal endpoint encodings, plus repeated IDs in both shards.
        tasks = rng.integers(19, size=(13, 2))
        children = rng.integers(-1, 19, size=(13, 4))
        children[:, 0] = 0
        labels = rng.integers(32, size=children.shape)
        optimizer = torch.optim.Adam(original.parameters(), lr=.0001)
        parallel_optimizer = torch.optim.Adam(replica.parameters(), lr=.0001)
        parallel = BatchParallel(replica, features, ["cpu", "cpu"])
        try:
            for _ in range(3):
                encoded = original.encode(features)
                loss = (F.smooth_l1_loss(original.distance(encoded[rows[:, 0]], encoded[rows[:, 1]]),
                                         torch.tensor(rows[:, 2], dtype=torch.float32))
                        + soft_rank_loss(original.distance(encoded[np.maximum(children, 0)],
                                                           encoded[tasks[:, 1]][:, None]),
                                         torch.tensor(labels), torch.tensor(children >= 0), 1.))
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(original.parameters(), 5.)
                optimizer.step()
                with patch("torch.nn.utils.clip_grad_norm_", wraps=torch.nn.utils.clip_grad_norm_) as clip:
                    a, b = parallel.update(rows, tasks, children, labels, parallel_optimizer, 1., 1.)
                    self.assertEqual(clip.call_count, 1)
                torch.testing.assert_close(a + b, loss.detach(), rtol=1e-6, atol=1e-6)
                for old, new in zip(original.parameters(), replica.parameters()):
                    torch.testing.assert_close(old.grad, new.grad, rtol=1e-4, atol=1e-6)
                    torch.testing.assert_close(old, new, rtol=1e-5, atol=1e-6)
                for primary, mirrored in zip(replica.parameters(), parallel.models[1].parameters()):
                    torch.testing.assert_close(primary, mirrored, rtol=0, atol=0)
        finally:
            parallel.close()


if __name__ == "__main__":
    unittest.main()
