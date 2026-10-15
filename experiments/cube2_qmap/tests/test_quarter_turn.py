import json
from pathlib import Path
import tempfile
import unittest
import subprocess
import sys

import numpy as np
import torch

from experiments.cube2_qmap.src.environment import Cube2
from experiments.cube2_qmap.src.local_graph import LocalGraphSampler, build_local_graph, local_template
from experiments.puzzle_qmap.src.data import cube_pairs, pair_keys
from experiments.puzzle_qmap.src.model import MapReader
from experiments.puzzle_qmap.src.run import train


class QuarterTurnTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.env = Cube2(metric="QTM_URF_6")
        cls.env.build_oracle(Path(cls.tmp.name) / "oracle.npy")
        cls.config = json.loads((Path(__file__).parents[1] / "configs/quarter_turn_local.json").read_text())

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_full_qtm_graph_and_same_face_composition(self):
        env = self.env
        self.assertEqual(env.actions, ("U", "U'", "R", "R'", "F", "F'"))
        self.assertEqual(np.bincount(env.depth).tolist(),
                         [1, 6, 27, 120, 534, 2256, 8969, 33058, 114149, 360508,
                          930588, 1350852, 782536, 90280, 276])
        for begin in range(0, env.size, 50000):
            ids = np.arange(begin, min(begin + 50000, env.size))
            delta = env.depth[env.successors(ids)].astype(int) - env.depth[ids, None].astype(int)
            self.assertTrue((np.abs(delta) == 1).all())
        for face in ("U", "R", "F"):
            half = env.step(env.step(0, face), face)
            self.assertEqual(env.distance(0, half), 2)
            self.assertEqual(env.step(env.step(12345, face), face + "'"), 12345)
            with self.assertRaises(ValueError):
                env.step(0, face + "2")
        with self.assertRaises(ValueError):
            Cube2(Path(self.tmp.name) / "oracle.npy")

    def test_local_data_does_not_import_learning_stack(self):
        subprocess.run([sys.executable, "-c", "import sys; import experiments.cube2_qmap.src.local_graph; "
                        "assert 'torch' not in sys.modules"], check=True)

    def test_state_pair_keys_do_not_overflow_int32_cube_ids(self):
        rows = np.array([[3674158, 3674159], [3674159, 3674158]], dtype=np.int32)
        keys = pair_keys(rows, self.env.size)
        expected = 3674158 * self.env.size + 3674159
        self.assertEqual(keys.dtype, np.int64)
        np.testing.assert_array_equal(keys, [expected, expected])

    def test_complete_shared_graph_fresh_pose_labels_and_holdout(self):
        env, config = self.env, self.config
        roots = np.array([np.flatnonzero(env.depth == d)[0] for d in range(1, 15)])
        data = {"relative_roots": roots, "protected_states": np.array([0, 12345, 23456, 34567])}
        sampler = LocalGraphSampler(env, data, config, np.random.default_rng(23))
        words, parents, children = local_template(env)
        self.assertEqual((len(words), len(parents), children.shape), (34, 7, (7, 6)))
        poses = []
        for _ in range(2):
            batch = sampler.draw(14, 128)
            states = batch["states"]
            self.assertFalse(np.isin(states, data["protected_states"]).any())
            np.testing.assert_array_equal(env.distance(states[batch["pairs"][:, 0]], states[batch["pairs"][:, 1]]),
                                          batch["pair_distances"])
            np.testing.assert_array_equal(env.successors(states[batch["parents"]]), states[batch["children"]])
            np.testing.assert_array_equal(env.distance(states[batch["children"]], states[batch["goals"]][:, None]),
                                          batch["child_distances"])
            self.assertTrue((np.abs(batch["delta"]) == 1).all())
            packed = states.reshape(14, 35)
            np.testing.assert_array_equal(env.relative(packed[:, 0], packed[:, 1]), batch["relative_roots"])
            np.testing.assert_array_equal(env.relative(packed[:, :1], packed[:, 1:]),
                                          env.compose(batch["relative_roots"][:, None], words))
            poses.append(dict(zip(batch["relative_roots"].tolist(), packed[:, 0].tolist())))
        self.assertTrue(all(poses[0][r] != poses[1][r] for r in roots))
        self.assertEqual(sampler.coverage()["root_min_layer_passes"], 2)

    def test_build_and_train_keep_pure_q_and_relabel_old_states(self):
        env = self.env
        config = json.loads(json.dumps(self.config))
        config["data"].update(pairs_validation=28, pairs_test=28, decisions_validation=28,
                              decisions_test=28, relative_limit=28, monitor_groups=14, monitor_pairs=140)
        config["training"].update(steps=2, batch_size=32, graph_batch_size=14, eval_every=1)
        config["sampling"]["minimum_full_passes"] = .001
        config["evaluation"].update(decision_count=14, rollout_count=14)
        buckets = [np.flatnonzero(env.depth == d) for d in range(15)]
        reference = {}
        rng = np.random.default_rng(81)
        for split, bucket in (("validation", 1), ("test", 2)):
            rows = cube_pairs(env, rng, 28, bucket, buckets)
            reference[f"pairs_{split}"] = rows.copy()
            reference[f"pairs_{split}"][:, 2] = 99  # Never reuse old metric labels.
            reference[f"tasks_{split}"] = rows[:, :2]
            reference[f"parent_distances_{split}"] = rows[:, 2]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fresh = root / "fresh"
            fresh.mkdir()
            fresh_data, fresh_summary = build_local_graph(env, config, fresh)
            self.assertNotIn("tasks_reference_test", fresh_data)
            self.assertIsNone(fresh_summary["reference_source_commit"])
            ref = root / "reference"
            ref.mkdir()
            np.savez(ref / "data.npz", **reference)
            config["data"]["reference_source_commit"] = "test-reference"
            (ref / "summary.json").write_text(json.dumps({"source_commit": config["data"]["reference_source_commit"]}))
            config["data"]["reference_data"] = str(ref / "data.npz")
            data, summary = build_local_graph(env, config, root)
            self.assertEqual(summary["action_count"], 6)
            self.assertEqual(summary["graph_unique_nodes"], 34)
            np.testing.assert_array_equal(data["tasks_reference_test"][:14],
                                          reference["tasks_test"][data["reference_rollout_source_indices"]])
            self.assertLessEqual(data["pairs_reference_test"][:, 2].max(), 14)
            result = train(env, config, data, root, "cpu", "qtm-smoke")
            saved = torch.load(root / "last.pt", weights_only=False)
            self.assertTrue(all(key.startswith("network.") for key in saved["model"]))
            self.assertEqual(saved["actions"], list(env.actions))
            self.assertEqual(saved["cap"], 14)
            self.assertIn("root_layers", result["last_model_training_coverage"])
            self.assertIn("soft_rank_loss", result["initial_training_monitor"])
            reader = MapReader(Cube2(metric="QTM_URF_6"), root / "last.pt")
            action, next_state = reader.choose(env.step(0, "U"), 0)
            self.assertEqual(next_state, env.step(env.step(0, "U"), action))
            with self.assertRaises(ValueError):
                MapReader(Cube2(), root / "last.pt")
            repair = root / "repair"
            repair.mkdir()
            config["sampling"].update(complete_goal_distances=True,
                                      decision_replay={"fraction": .5, "capacity": 128})
            repaired = train(env, config, data, repair, "cpu", "repair-smoke", root / "best.pt")
            self.assertGreater(repaired["last_model_training_coverage"]["replayed_pose_groups"], 0)
            config["training"]["progress_margin_weight"] = 1.
            with self.assertRaisesRegex(ValueError, "Archived loss experiments"):
                train(env, config, data, root, "cpu", "retired-contract")

    def test_complete_distance_supervision_and_replay_only_wrong_training_graphs(self):
        env = self.env
        config = json.loads(json.dumps(self.config))
        config["sampling"].update(complete_goal_distances=True,
                                  decision_replay={"fraction": .5, "capacity": 2})
        data = {"relative_roots": np.array([np.flatnonzero(env.depth == 4)[0]]),
                "protected_states": np.array([0, 12345, 23456])}
        sampler = LocalGraphSampler(env, data, config, np.random.default_rng(103))
        first = sampler.draw(4, 32)
        pairs = set(map(tuple, np.sort(first["pairs"], axis=1)))
        candidates = np.column_stack((first["children"].ravel(), np.repeat(first["goals"], 6)))
        self.assertTrue(set(map(tuple, np.sort(candidates, axis=1))).issubset(pairs))
        np.testing.assert_array_equal(env.distance(first["states"][first["pairs"][:, 0]],
                                                   first["states"][first["pairs"][:, 1]]),
                                      first["pair_distances"])
        sampler.observe(first, -first["child_distances"].astype(float))
        self.assertEqual(len(sampler.replay), 2)
        retained = set(sampler.replay)
        second = sampler.draw(4, 32)
        posed = second["states"].reshape(4, 35)
        self.assertEqual(len(retained & set(map(tuple, posed[:, :2]))), 2)
        self.assertFalse(np.isin(second["states"], data["protected_states"]).any())
        self.assertEqual(sampler.coverage()["fresh_pose_groups"], 6)
        sampler.observe(second, -second["child_distances"].astype(float))
        self.assertEqual(set(sampler.replay), retained)  # Capacity cannot evict unresolved lessons.
        sampler.observe(second, second["child_distances"].astype(float))
        self.assertEqual(len(sampler.replay), 0)
        self.assertEqual(sampler.coverage()["replay_checks"], 4)
        self.assertEqual(sampler.coverage()["replay_corrected"], 2)


if __name__ == "__main__":
    unittest.main()
