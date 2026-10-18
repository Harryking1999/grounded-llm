import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from experiments.blocks_distance_map.src.full_graph_data import (
    Component, Root, build_graph, components, local_tiles,
)
from experiments.blocks_distance_map.src.oracle import DistanceOracle, successors
from experiments.blocks_distance_map.src.symbolic_graph import build_symbolic, SymbolicComponent


class CompleteGraphTest(unittest.TestCase):
    def test_symbolic_all_states_pairs_and_actions_equal_explicit_bfs(self):
        tiles = [3, 6, 12, 24, 48, 7, 14, 28, 56]
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)
            build_graph(6, tiles, path / "dense", 1000)
            build_symbolic(6, tiles, path / "symbolic", 10000)
            dense, symbolic = Component(path / "dense"), SymbolicComponent(path / "symbolic")
            ids = np.arange(symbolic.size)
            masks = symbolic.states[ids]
            self.assertEqual(set(masks), set(dense.states))
            np.testing.assert_array_equal(symbolic.rank(masks), ids)
            source, goal = np.meshgrid(masks, masks, indexing="ij")
            np.testing.assert_array_equal(symbolic.distance(source, goal), dense.distance(source, goal))
            for i, mask in enumerate(masks):
                _, children = symbolic.star(np.asarray([i]))
                self.assertEqual(set(symbolic.states[children]),
                                 {int(mask) ^ tile for tile in tiles if int(mask) & tile == tile})
            dense.close()

    def test_full_product_all_pairs_match_original_oracle_and_all_transitions(self):
        initial = (1 << 0) | (1 << 1) | (1 << 10) | (1 << 66) | (1 << 67)
        with tempfile.TemporaryDirectory() as temp:
            graphs, parts = {}, []
            for i, cells in enumerate(components(initial)):
                path = Path(temp) / str(i)
                build_graph(len(cells), local_tiles(cells), path, 1000)
                graphs[i] = Component(path)
                parts.append({"graph": i, "cells": cells})
            root = Root({"components": parts}, graphs)
            ids = np.arange(root.size)
            local = root.decode(ids)
            packed = root.packed(local)
            masks = [int(lo) | (int(hi) << 64) for lo, hi in packed]
            self.assertEqual(masks[0], initial)
            self.assertEqual(len(set(masks)), root.size)
            oracle = DistanceOracle()
            for source in ids:
                expected = [oracle.distance(masks[source], goal) for goal in masks]
                np.testing.assert_array_equal(root.distance(source, ids), expected)
                for _, child in successors(masks[source]):
                    self.assertIn(child, masks)
            advanced = root.packed(root.advance(local, np.random.default_rng(18), np.ones(root.size, dtype=int)))
            for source, (lo, hi) in zip(masks, advanced):
                legal = [child for _, child in successors(source)]
                self.assertIn(int(lo) | (int(hi) << 64), legal or [source])
            for graph in graphs.values():
                graph.close()

    def test_resource_limit_never_marks_a_partial_graph_complete(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "partial"
            with self.assertRaises(RuntimeError):
                build_graph(3, [3, 5, 7], path, 2)
            self.assertFalse((path / "summary.json").exists())


if __name__ == "__main__":
    unittest.main()
