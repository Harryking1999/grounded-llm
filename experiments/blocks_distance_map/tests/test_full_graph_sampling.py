import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from experiments.blocks_distance_map.src.full_graph_sampling import JointData
from experiments.blocks_distance_map.src.full_graph_data import local_tiles
from experiments.blocks_distance_map.src.symbolic_graph import build_symbolic, SymbolicComponent
from experiments.blocks_distance_map.src.oracle import DistanceOracle
from experiments.blocks_distance_map.src.full_graph_run import relation_batch
from experiments.blocks_distance_map.src.model import soft_rank_loss


class SamplingTest(unittest.TestCase):
    def test_variable_complete_forks_keep_original_soft_rank_and_gradient(self):
        masks = np.asarray([[1, 0], [2, 0], [3, 0], [4, 0], [5, 0]], dtype=np.uint64)
        batch = (masks[:1], masks[1:2], np.asarray([1]), masks, masks[::-1],
                 np.asarray([2, -1, 1, 0, 3]), np.asarray([1, 0, 1, 2, 1]))
        _, _, _, truth, order, slots, shape = relation_batch(batch, "cpu", 13)
        scores = torch.tensor([.5, 1.5, 2.5, 3.5, 4.5], requires_grad=True)
        matrix = torch.zeros(shape[0] * shape[1]).scatter(0, slots, scores[order]).reshape(shape)
        target = torch.zeros(shape[0] * shape[1]).scatter(0, slots, truth).reshape(shape)
        valid = torch.zeros(shape[0] * shape[1], dtype=torch.bool)
        valid[slots] = True
        actual = soft_rank_loss(matrix, target, valid.reshape(shape), 1.)
        wanted = sum(soft_rank_loss(scores[torch.tensor(batch[-1] == group)][None],
                                   torch.tensor(np.where(batch[-2][batch[-1] == group] < 0, 13,
                                                         batch[-2][batch[-1] == group]))[None],
                                   torch.ones((1, sum(batch[-1] == group)), dtype=torch.bool), 1.)
                     for group in range(3)) / 3
        torch.testing.assert_close(actual, wanted)
        torch.testing.assert_close(torch.autograd.grad(actual, scores, retain_graph=True)[0],
                                   torch.autograd.grad(wanted, scores)[0])

    def test_exact_labels_complete_pair_traversal_distributed_shards_and_resume(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)
            cells = [0, 1, 10]
            graph_path = path / "graphs" / "0"
            build_symbolic(3, local_tiles(cells), graph_path, 10000)
            graph = SymbolicComponent(graph_path)
            size = graph.size
            root = {"initial": "1027", "construction": [{}],
                    "components": [{"graph": 0, "cells": cells}]}
            (path / "roots.json").write_text(json.dumps([root]))
            (path / "summary.json").write_text(json.dumps({"complete": True,
                "total_root_local_pairs": size ** 2, "total_root_local_states": size}))
            masks = [sum(((int(mask) >> bit) & 1) << cell for bit, cell in enumerate(cells))
                     for mask in graph.states[np.arange(size)]]
            oracle = DistanceOracle()
            rows = [[i, j, oracle.distance(s, g), 0 if oracle.distance(s, g) >= 0 else 1]
                    for i, s in enumerate(masks) for j, g in enumerate(masks)]
            np.savez(path / "legacy.npz", states=np.asarray(masks, dtype=str), train=np.asarray(rows))
            options = {"seed": 18, "boards_per_batch": 1, "legacy_fraction": .1,
                       "broad_fraction": .5, "clear_fraction": .25}
            def make(rank=0, world=1):
                return JointData(path, path / "legacy.npz", options, rank, world)
            first, second = make(0, 2), make(1, 2)
            shards = []
            for sampler in (first, second):
                source, goal = sampler.broad_rows(0, size ** 2 // 2)
                shards.append(set(zip(source.ravel(), goal.ravel())))
            self.assertFalse(shards[0] & shards[1])
            self.assertEqual(len(shards[0] | shards[1]), size ** 2)
            sampler = make()
            batch = sampler.batch(128, 32)
            for source, goal, truth in ((batch[0], batch[1], batch[2]),
                                        (batch[3], batch[4], batch[5])):
                np.testing.assert_array_equal(truth, [oracle.distance(int(s[0]), int(g[0]))
                                                      for s, g in zip(source, goal)])
            state = copy.deepcopy(sampler.state_dict())
            expected = sampler.batch(128, 32)
            resumed = make()
            resumed.load_state_dict(state)
            for actual, wanted in zip(resumed.batch(128, 32), expected):
                np.testing.assert_array_equal(actual, wanted)
            for item in (first, second, sampler, resumed):
                item.close()


if __name__ == "__main__":
    unittest.main()
