import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from experiments.cube2_qmap.src.data import build_breadth_depth
from experiments.cube2_qmap.src.environment import Cube2
from experiments.puzzle_qmap.src.data import cube_pairs
from experiments.puzzle_qmap.src.sampling import CoverageSampler


class CoverageDataTest(unittest.TestCase):
    def test_closed_paths_forks_relative_types_and_protected_endpoints(self):
        env, rng = Cube2(), np.random.default_rng(16)
        config = json.loads((Path(__file__).parents[1] / "configs/breadth_depth.json").read_text())
        config["data"].update(relative_limit=128,near_pairs_per_depth=16,trajectories=8,forks=4,
                              pairs_validation=24,pairs_test=24,decisions_validation=24,decisions_test=24)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            env.build_oracle(root / "oracle.npy")
            buckets = [np.flatnonzero(env.depth == d) for d in range(12)]
            reference = {}
            for split, label in (("validation", 1), ("test", 2)):
                rows = cube_pairs(env, rng, 32, label, buckets)
                reference[f"pairs_{split}"] = rows
                reference[f"tasks_{split}"] = rows[:, :2]
            source = root / "reference"
            source.mkdir()
            np.savez(source / "data.npz", **reference)
            (source / "summary.json").write_text(json.dumps({"source_commit": config["data"]["reference_source_commit"]}))
            config["data"]["reference_data"] = str(source / "data.npz")
            data, summary = build_breadth_depth(env, config, root)
        forbidden = np.zeros(env.size, dtype=bool)
        forbidden[data["protected_states"]] = True
        for key in ("tasks_train", "children_train"):
            self.assertFalse(forbidden[data[key]].any())
        self.assertFalse(forbidden[data["pairs_train"][:, :2]].any())
        n = int(data["broad_task_count"])
        tasks = data["tasks_train"]
        np.testing.assert_array_equal(env.relative(tasks[:n, 1], tasks[:n, 0]), data["broad_relative_ids"])
        np.testing.assert_array_equal(env.distance(data["children_train"], tasks[:, 1, None]), data["child_distances_train"])
        np.testing.assert_array_equal(env.distance(data["pairs_train"][:, 0], data["pairs_train"][:, 1]), data["pairs_train"][:, 2])
        path_count = int(data["path_task_count"])
        path = tasks[n:n + path_count]
        for begin, end in zip(data["trajectory_offsets"][:-1], data["trajectory_offsets"][1:]):
            rows = path[begin:end]
            self.assertTrue((rows[:, 1] == rows[0, 1]).all())
            np.testing.assert_array_equal(env.distance(rows[:, 0], rows[:, 1]), np.arange(len(rows), 0, -1))
            for i in range(len(rows) - 1):
                self.assertIn(rows[i + 1, 0], env.successors(rows[i, 0]))
        forks = tasks[n + path_count:].reshape(-1, 10, 2)
        np.testing.assert_array_equal(env.successors(forks[:, 0, 0]), forks[:, 1:, 0])
        self.assertTrue((forks[:, :, 1] == forks[:, :1, 1]).all())
        self.assertEqual(summary["relative_decision_types"], 128)

    def test_sampler_really_traverses_breadth_and_keeps_context(self):
        sampler = CoverageSampler(np.tile([1, 2, 3, 4], 5), np.random.default_rng(16), 10, .5, 10, .25)
        draws = np.concatenate([sampler.draw(8) for _ in range(3)])
        self.assertGreaterEqual(sampler.passes, 1)
        self.assertTrue(set(range(10)).issubset(draws))
        self.assertTrue(((draws >= 0) & (draws < 20)).all())
        self.assertTrue((draws >= 10).any())


if __name__ == "__main__":
    unittest.main()
