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
            # Splitting two live streams into four must preserve exactly the
            # next disjoint relation set, rather than restarting sampling.
            parents = [BroadData(path, path / "legacy.npz", options, rank, 2) for rank in range(2)]
            for rank, parent in enumerate(parents):
                parent.broad_rows(0, 3)
                torch.save(parent.state_dict(), path / f"sampler_000002_rank{rank}.pt")
            def pairs(sampler, count):
                source, goal = sampler.broad_rows(0, count)
                return set(zip(map(tuple, source.tolist()), map(tuple, goal.tolist())))
            expected = set().union(*(pairs(parent, 8) for parent in parents))
            split_options = {**options, "previous_sampler_root": str(path),
                             "previous_sampler_step": 2, "previous_sampler_world": 2}
            children = [BroadData(path, path / "legacy.npz", split_options, rank, 4)
                        for rank in range(4)]
            actual = [pairs(child, 4) for child in children]
            self.assertEqual(sum(map(len, actual)), len(set().union(*actual)))
            self.assertEqual(expected, set().union(*actual))
            self.assertNotEqual(children[0].rng.bit_generator.state, children[2].rng.bit_generator.state)
            for sampler in parents + children:
                sampler.close()
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
            forked = BroadData(path, path / "legacy.npz", {**options,
                "clear_index_data": str(path / "clear"), "clear_broad_fraction": 0,
                "clear_distance_cycle": [8], "clear_fork_huber_fraction": .7,
                "clear_fraction": .9, "broad_fraction": .05})
            batch = forked.batch(512, 32)
            huber_sources = set(map(int, batch[0][:, 0]))
            initial = int(double["initial"])
            from experiments.blocks_distance_map.src.oracle import successors
            self.assertTrue({child for _, child in successors(initial)}.issubset(huber_sources))
            np.testing.assert_array_equal(batch[2], [oracle.distance(int(a[0]), int(b[0]))
                                                    for a, b in zip(batch[0], batch[1])])
            self.assertEqual(forked.phase_sketches["clear_fork_huber_groups"].estimate(), 0)
            forked.record_update(batch, np.ones(len(batch[2]), bool), np.ones(len(batch[5]), bool))
            self.assertGreater(forked.phase_sketches["clear_fork_huber_groups"].estimate(), 0)
            forked.close()
            # Two independent identical components double every finite exact
            # distance, providing historical short AND6--10-step relations.
            old_rows = np.asarray(rows + [[int(graph.initial_id), int(graph.zero_id),
                                          oracle.distance(1023, 0), 0]])
            old_rows[old_rows[:, 2] >= 0, 2] *= 2
            doubled_masks = np.asarray([str(int(m) | (int(m) << 20)) for m in masks])
            np.savez(path / "legacy_double.npz", states=doubled_masks, train=old_rows)
            retention = BroadData(path, path / "legacy_double.npz", {**options,
                "legacy_fraction": 1, "legacy_long_fraction": .5})
            retained_batch = retention.batch(6144, 96)
            truth = retained_batch[2]
            self.assertGreater(((truth >= 0) & (truth < 6)).mean(), .10)
            self.assertGreater((truth < 0).mean(), .10)
            self.assertGreater((truth >= 6).mean(), .45)
            retention.record_update(retained_batch, np.ones(len(truth), bool),
                                    np.ones(len(retained_batch[5]), bool))
            self.assertEqual(retention.legacy_update_distances,
                             {str(int(d)): int(n) for d, n in zip(*np.unique(truth, return_counts=True))})
            retention.close()
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
            layered = BroadData(path, path / "legacy.npz", {**options,
                "branch_bundle_fraction": .5, "exact_layer_fraction": 1,
                "stream_distance_cycle": [6, 7, 8, 13]})
            wanted = np.tile([2, 3, 6, 7, 8, 13], 20)
            goals, actual = layered._layer_targets(0, wanted)
            full = int(double["initial"])
            packed_goals = [int(a) | (int(b) << 20) for a, b in goals]
            self.assertTrue(all(packed_goals))
            np.testing.assert_array_equal(actual, [oracle.distance(full, g) for g in packed_goals])
            self.assertTrue((actual[wanted == 6] == 6).all())
            # The final Huber rows retain requested long finite relationships;
            # exact labels and full candidate sets are independently checked.
            batch = layered.batch(512, 96)
            for s, g, d in ((batch[0], batch[1], batch[2]), (batch[3], batch[4], batch[5])):
                np.testing.assert_array_equal(d, [oracle.distance(int(a[0]), int(b[0])) for a, b in zip(s, g)])
            source, goal = layered.structure_rows(0, 120, False)
            star = layered.stars(layered.roots[0], source, goal)
            degree = sum(g.degrees(source[:, i]) for i, (g, _) in enumerate(layered.roots[0].parts))
            np.testing.assert_array_equal(np.bincount(star[-1], minlength=len(source)), degree)
            groups = layered.roots[0].packed(source[:60])[:, 0].reshape(-1, 12)
            shared_goals = layered.roots[0].packed(goal[:60])[:, 0].reshape(-1, 12)
            self.assertTrue(all(oracle.distance(int(s), int(g)) >= 0
                                for s, g in zip(groups[:, 3], shared_goals[:, 3])))
            state = copy.deepcopy(layered.state_dict())
            resumed = BroadData(path, path / "legacy.npz", layered.options)
            resumed.load_state_dict(state)
            for a, b in zip(layered.batch(256, 96), resumed.batch(256, 96)):
                np.testing.assert_array_equal(a, b)
            layered.close(); resumed.close()


if __name__ == "__main__":
    unittest.main()
