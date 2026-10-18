import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from experiments.gcml_counterexamples.src import blocks
from experiments.blocks_distance_map.src.failure_sampling import FailureData
from experiments.blocks_distance_map.src.full_graph_data import local_tiles
from experiments.blocks_distance_map.src.symbolic_graph import build_symbolic, SymbolicComponent
from experiments.blocks_distance_map.src.oracle import DistanceOracle
from experiments.blocks_distance_map.src.full_graph_run import relation_batch


class FailureSamplingTest(unittest.TestCase):
    def test_exact_forks_actual_exposure_and_sampler_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            cells = [0, 1, 2, 10, 11, 12]
            initial = sum(1 << c for c in cells)
            build_symbolic(len(cells), local_tiles(cells), path / "graphs" / "0", 10000)
            graph = SymbolicComponent(path / "graphs" / "0")
            size = graph.size
            root = {"initial": str(initial), "construction": [{}],
                    "components": [{"graph": 0, "cells": cells}]}
            (path / "roots.json").write_text(json.dumps([root]))
            (path / "summary.json").write_text(json.dumps({"complete": True,
                "total_root_local_pairs": size ** 2, "total_root_local_states": size}))
            masks = [sum(((int(mask) >> bit) & 1) << cell for bit, cell in enumerate(cells))
                     for mask in graph.states[np.arange(size)]]
            oracle = DistanceOracle()
            rows = [[i, j, oracle.distance(s, g), int(oracle.distance(s, g) < 0)]
                    for i, s in enumerate(masks) for j, g in enumerate(masks)]
            np.savez(path / "legacy.npz", states=np.asarray(masks, dtype=str), train=np.asarray(rows))
            star = {"source": str(initial), "goal": str(initial ^ 3)}
            bad_action = blocks.BY_ACTION[(4, 0, 0)]
            bad = initial ^ blocks.PLACEMENTS[bad_action][0]
            star.update(chosen_action=bad_action, failure_type="removed_goal_cell",
                        candidates=[{"action": bad_action, "state": str(bad), "truth": -1}])
            diagnostic = {"training_long": {"rows": [{"board": 0, "source": str(initial),
                "success": False, "failure": star, "trajectory": [star]}]},
                "monitor_long": {"rows": [{"board": 999, "source": "invalid"}]}}
            (path / "diagnosis.json").write_text(json.dumps(diagnostic))
            retained = {"training_long": {"rows": [{"board": 0, "source": str(initial),
                        "success": True, "trajectory": [star]}]}}
            (path / "retention.json").write_text(json.dumps(retained))
            options = {"seed": 18, "boards_per_batch": 1, "legacy_fraction": .1,
                       "broad_fraction": .25, "clear_fraction": .35, "failure_fraction": .5,
                       "repair_descendant_steps": [0, 1, 2],
                       "failure_diagnosis": str(path / "diagnosis.json"),
                       "retention_diagnosis": str(path / "retention.json"),
                       "hard_negative_fraction": .25, "retention_fraction": .25,
                       "fresh_fraction": .25}
            def make():
                return FailureData(path, path / "legacy.npz", options)
            sampler = make()
            batch = sampler.batch(128, 32)
            for sources, goals, truth in ((batch[0], batch[1], batch[2]),
                                          (batch[3], batch[4], batch[5])):
                np.testing.assert_array_equal(truth, [oracle.distance(int(s[0]), int(g[0]))
                                                      for s, g in zip(sources, goals)])
            self.assertEqual(sum(sampler.report()["local_pair_draws"].values()), 128)
            self.assertTrue(sampler.repair_pairs_seen)
            self.assertTrue(sampler.repair_candidates_seen)
            self.assertGreater(sampler.hard_negative_draws, 0)
            self.assertGreater(sampler.retention_draws, 0)
            state = copy.deepcopy(sampler.state_dict())
            wanted = sampler.batch(128, 32)
            restored = make()
            restored.load_state_dict(state)
            for actual, expected in zip(restored.batch(128, 32), wanted):
                np.testing.assert_array_equal(actual, expected)
            self.assertEqual(restored.report(), sampler.report())
            torch.save(sampler.state_dict(), path / "sampler_001000_rank0.pt")
            options.update(previous_sampler_root=str(path), previous_sampler_step=1000)
            continued = make()
            np.testing.assert_array_equal(continued.broad, sampler.broad)
            self.assertEqual(continued.repair_pairs_seen, sampler.repair_pairs_seen)
            self.assertEqual(continued.repair_pair_draws, sampler.repair_pair_draws)
            self.assertEqual(continued.hard_negative_draws, sampler.hard_negative_draws)
            for item in (sampler, restored, continued):
                item.close()

    def test_only_unreachable_targets_change_and_label_exceeds_finite_bound(self):
        masks = np.asarray([[1, 0], [2, 0], [3, 0]], dtype=np.uint64)
        batch = (masks, masks, np.asarray([1, 14, -1]), masks, masks,
                 np.asarray([0, 13, -1]), np.zeros(3, dtype=int))
        old, new = relation_batch(batch, "cpu", 13), relation_batch(batch, "cpu", 19)
        torch.testing.assert_close(old[0], new[0])
        torch.testing.assert_close(old[1], new[1])
        self.assertEqual(new[2].tolist(), [1, 14, 19])
        self.assertEqual(new[3].tolist(), [0, 13, 19])
        sizes = [len(s) for s in blocks.SHAPES]
        self.assertLessEqual(12 * max(sizes) // min(sizes), 18)


if __name__ == "__main__":
    unittest.main()
