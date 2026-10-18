from pathlib import Path
import tempfile
import unittest
import numpy as np
from scipy.sparse import csr_matrix
from experiments.puzzle8_qmap.src.full_oracle import initialize, fill, tensor_bfs


class FullOracleTest(unittest.TestCase):
    def test_disjoint_goal_chunks_write_exact_rows(self):
        graph = csr_matrix(np.array([[0, 1, 0, 0], [1, 0, 1, 0],
                                     [0, 1, 0, 1], [0, 0, 1, 0]], dtype=np.int8))
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "distances.npy"
            table = np.lib.format.open_memmap(path, mode="w+", dtype=np.int8, shape=(4, 4))
            initialize(graph, path)
            self.assertEqual(fill((2, 4)), 2)
            self.assertEqual(fill((0, 2)), 2)
            np.testing.assert_array_equal(table, np.abs(np.arange(4)[:, None] - np.arange(4)))
            # Release the worker's mapping before Windows temporary-file cleanup.
            import experiments.puzzle8_qmap.src.full_oracle as module
            del module.TABLE
            del table

    def test_tensor_frontiers_match_exact_shortest_paths(self):
        import torch
        dense = torch.tensor([[0., 1., 0., 0.], [1., 0., 1., 0.],
                              [0., 1., 0., 1.], [0., 0., 1., 0.]])
        actual = tensor_bfs(dense.to_sparse(), torch.tensor([3, 0]), 4, 3)
        np.testing.assert_array_equal(actual, [[3, 2, 1, 0], [0, 1, 2, 3]])


if __name__ == "__main__":
    unittest.main()
