import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from experiments.blocks_distance_map.src.broad_sampling import BroadData, Cardinality
from experiments.blocks_distance_map.src.clear_distance_index import build_component
from experiments.blocks_distance_map.src.full_graph_data import local_tiles
from experiments.blocks_distance_map.src.symbolic_graph import build_symbolic, SymbolicComponent
from experiments.blocks_distance_map.src.oracle import DistanceOracle


class BroadSamplingTest(unittest.TestCase):
    def test_global_cardinality_deduplicates_shards_and_overlapping_roots(self):
        rng = np.random.default_rng(60)
        rows = rng.integers(1 << 60, size=(50000, 4), dtype=np.uint64)
        a, b = Cardinality(), Cardinality()
        a.add(rows[:35000]); b.add(rows[20000:]); b.add(rows[20000:])
        result = BroadData.merged_exposure([{"pairs": a.registers}, {"pairs": b.registers}])
        self.assertLess(abs(result["estimated_distinct_pairs"] - 50000), 2000)
        self.assertEqual(Cardinality().estimate(), 0)

    def test_stream_labels_forks_exposure_and_resume(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)
            cells = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
            build_symbolic(len(cells), local_tiles(cells), path / "graphs/0", 100000)
            graph = SymbolicComponent(path / "graphs/0")
            root = {"initial": "1023", "construction": [{}],
                    "components": [{"graph": 0, "cells": cells}]}
            (path / "roots.json").write_text(json.dumps([root]))
            (path / "summary.json").write_text(json.dumps({"complete": True,
                "total_root_local_pairs": graph.size ** 2, "total_root_local_states": graph.size}))
            masks = graph.states[np.arange(graph.size)]
            oracle = DistanceOracle()
            # A compact real finite/unreachable legacy training pool.
            ids = np.arange(min(30, graph.size))
            rows = [[int(i), int(j), oracle.distance(int(masks[i]), int(masks[j])),
                     int(oracle.distance(int(masks[i]), int(masks[j])) < 0)] for i in ids for j in ids]
            np.savez(path / "legacy.npz", states=masks.astype(str), train=np.asarray(rows))
            options = {"seed": 7, "boards_per_batch": 1, "legacy_fraction": .1,
                       "broad_fraction": .15, "clear_fraction": .25,
                       "stream_distance_cycle": [1, 2, 3, 4]}
            data = BroadData(path, path / "legacy.npz", options)
            batch = data.batch(128, 32)
            for source, goal, truth in ((batch[0], batch[1], batch[2]), (batch[3], batch[4], batch[5])):
                np.testing.assert_array_equal(truth,
                    [oracle.distance(int(s[0]), int(g[0])) for s, g in zip(source, goal)])
            data.record_update(batch, np.ones(len(batch[2]), bool), np.ones(len(batch[5]), bool))
            self.assertGreater(data.report()["stream_phase"]["local_estimated_distinct"]["goals"], 0)
            state = copy.deepcopy(data.state_dict())
            resumed = BroadData(path, path / "legacy.npz", options)
            resumed.load_state_dict(state)
            for a, b in zip(data.batch(128, 32), resumed.batch(128, 32)):
                np.testing.assert_array_equal(a, b)
            # Expand the domain while retaining old cursors and old affine factors.
            torch.save(state, path / "sampler_000001_rank0.pt")
            (path / "roots.json").write_text(json.dumps([root, root]))
            expanded_options = {**options, "previous_sampler_root": str(path), "previous_sampler_step": 1}
            expanded = BroadData(path, path / "legacy.npz", expanded_options)
            self.assertEqual(expanded.broad[0], state["broad"][0])
            self.assertEqual(expanded.broad[1], 0)
            self.assertEqual(expanded.affine[0], data.affine[0])
            self.assertEqual(set(expanded.order), {0, 1})
            for s in (data, resumed, expanded): s.close()
            build_component(path / "graphs/0", path / "clear/0")
            shifted = [c + 20 for c in cells]
            double = {"initial": str(1023 | (1023 << 20)), "construction": [{}],
                      "components": [{"graph": 0, "cells": cells}, {"graph": 0, "cells": shifted}]}
            (path / "roots.json").write_text(json.dumps([double]))
            indexed = BroadData(path, path / "legacy.npz", {**options,
                "clear_index_data": str(path / "clear"), "clear_broad_fraction": 0,
                "clear_distance_cycle": [6, 7, 8, 9, 10, 13],
                "stream_distance_cycle": [6, 7, 8], "medium_deep_source_fraction": .5})
            s, g = indexed.clear_rows(0, 180)
            distance = indexed.roots[0].distance_parts(s, g).sum(-1)
            self.assertTrue(set(distance).issubset({6, 7, 8}))
            self.assertTrue({6, 7, 8}.issubset(set(distance)))
            for mask, truth in zip(indexed.roots[0].packed(s), distance):
                self.assertEqual(oracle.distance(int(mask[0]), 0), truth)
            batch = indexed.batch(128, 32)
            for ss, gg, dd in ((batch[0], batch[1], batch[2]), (batch[3], batch[4], batch[5])):
                np.testing.assert_array_equal(dd, [oracle.distance(int(a[0]), int(b[0])) for a, b in zip(ss, gg)])
            saved = copy.deepcopy(indexed.state_dict())
            replay = BroadData(path, path / "legacy.npz", indexed.options)
            replay.load_state_dict(saved)
            for a, b in zip(indexed.batch(128, 32), replay.batch(128, 32)):
                np.testing.assert_array_equal(a, b)
            indexed.close(); replay.close()
            bundled = BroadData(path, path / "legacy.npz", {**options,
                "branch_bundle_fraction": .5, "stream_distance_cycle": [6, 7, 8]})
            source, goal = bundled.structure_rows(0, 120, True)
            root = bundled.roots[0]
            ss, gg = root.packed(source)[:, 0], root.packed(goal)[:, 0]
            # Five twelve-context groups explicitly share sources and goals.
            # First actions differ whenever enough branches exist; DAG merges
            # remain legal, and all cross-branch labels come from the oracle.
            ss, gg = ss[:60].reshape(-1, 12), gg[:60].reshape(-1, 12)
            np.testing.assert_array_equal(ss[:, 0], ss[:, 1])
            np.testing.assert_array_equal(ss[:, 0], ss[:, 2])
            np.testing.assert_array_equal(gg[:, 0], gg[:, 3])
            np.testing.assert_array_equal(gg[:, 0], gg[:, 6])
            np.testing.assert_array_equal(gg[:, 0], gg[:, 9])
            np.testing.assert_array_equal(gg[:, 1], gg[:, 4])
            np.testing.assert_array_equal(gg[:, 1], gg[:, 7])
            degrees = sum(g.degrees(source[:60:12, i]) for i, (g, _) in enumerate(root.parts))
            self.assertTrue((ss[degrees > 1, 3] != ss[degrees > 1, 6]).all())
            self.assertTrue((ss[degrees >= 3, 3] != ss[degrees >= 3, 9]).all())
            self.assertTrue((ss[degrees >= 3, 6] != ss[degrees >= 3, 9]).all())
            for a, b, c, d in zip(ss[:, 3], ss[:, 6], gg[:, 0], gg[:, 1]):
                self.assertGreaterEqual(oracle.distance(int(a), int(c)), 0)
                self.assertGreaterEqual(oracle.distance(int(b), int(d)), 0)
            self.assertEqual(bundled.phase_sketches["bundle_huber_relations"].estimate(), 0)
            batch = bundled.batch(256, 96)
            for s, g, d in ((batch[0], batch[1], batch[2]), (batch[3], batch[4], batch[5])):
                np.testing.assert_array_equal(d, [oracle.distance(int(a[0]), int(b[0])) for a, b in zip(s, g)])
            # Every selected bundle ranking parent retains its entire original
            # legal candidate set, including unreachable target continuations.
            source, goal = bundled.structure_rows(0, 120, False)
            star = bundled.stars(root, source, goal)
            degree = sum(g.degrees(source[:, i]) for i, (g, _) in enumerate(root.parts))
            np.testing.assert_array_equal(np.bincount(star[-1], minlength=len(source)), degree)
            # Begin a real batch again: diagnostics above must not be recorded
            # as optimizer exposure by the subsequent actual-update hook.
            batch = bundled.batch(256, 96)
            bundled.record_update(batch, np.ones(len(batch[2]), bool), np.ones(len(batch[5]), bool))
            self.assertGreater(bundled.phase_sketches["bundle_huber_relations"].estimate(), 0)
            self.assertGreater(bundled.phase_sketches["bundle_rank_candidate_relations"].estimate(), 0)
            self.assertGreater(bundled.phase_sketches["bundle_shared_source_groups"].estimate(), 0)
            self.assertGreater(bundled.phase_sketches["bundle_shared_goal_groups"].estimate(), 0)
            saved = copy.deepcopy(bundled.state_dict())
            replay = BroadData(path, path / "legacy.npz", bundled.options)
            replay.load_state_dict(saved)
            for a, b in zip(bundled.batch(256, 96), replay.batch(256, 96)):
                np.testing.assert_array_equal(a, b)
            bundled.close(); replay.close()


if __name__ == "__main__":
    unittest.main()
