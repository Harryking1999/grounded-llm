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
    def test_mask_walk_matches_index_walk_and_random_stream(self):
        with tempfile.TemporaryDirectory() as temp:
            graphs={};parts=[]
            for i in range(2):
                path=Path(temp)/str(i)
                builder,kind=(build_graph,Component) if i==0 else (build_symbolic,SymbolicComponent)
                builder(6,[3,6,12,24,48,7,14,28,56],path,10000)
                graphs[i]=kind(path);parts.append(dict(graph=i,cells=list(range(6))))
            root=Root(dict(components=parts),graphs)
            local=root.decode(np.arange(root.size));limits=np.arange(len(local))%5
            expected=local.copy();rng=np.random.default_rng(18)
            for move in range(int(limits.max())):
                degrees=np.stack([g.degrees(expected[:,i]) for i,(g,_) in enumerate(root.parts)],-1)
                total=degrees.sum(-1);ids=np.flatnonzero((total>0)&(move<limits))
                if not len(ids):break
                choice=(rng.random(len(ids))*total[ids]).astype(np.int64)
                ends=np.cumsum(degrees[ids],-1);selected=(choice[:,None]>=ends).sum(-1)
                starts=np.column_stack((np.zeros(len(ids),dtype=np.int64),ends[:,:-1]))
                offset=choice-starts[np.arange(len(ids)),selected]
                for i,(graph,_) in enumerate(root.parts):
                    keep=np.flatnonzero(selected==i);rows=ids[keep]
                    expected[rows,i]=graph.take(expected[rows,i],offset[keep])
            actual_rng=np.random.default_rng(18)
            np.testing.assert_array_equal(root.advance(local,actual_rng,limits),expected)
            self.assertEqual(actual_rng.bit_generator.state,rng.bit_generator.state)
            for graph in graphs.values():graph.close()

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
            np.testing.assert_array_equal(dense.states[dense.rank(masks)],masks)
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
