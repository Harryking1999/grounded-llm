import json
from pathlib import Path
import tempfile
import unittest
import subprocess
import sys

import numpy as np
import torch
from torch.nn import functional as F

from experiments.cube2_qmap.src.environment import Cube2
from experiments.cube2_qmap.src.local_graph import LocalGraphSampler, build_local_graph, local_template
from experiments.puzzle_qmap.src.data import cube_pairs, pair_keys
from experiments.puzzle_qmap.src.model import MapReader
from experiments.puzzle_qmap.src.run import train
from experiments.puzzle_qmap.src.model import QMap, distance
from experiments.cube2_qmap.src.structure_study import build_template, objective, relation_values
from experiments.cube2_qmap.src.linf_dimension import construct


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

    def test_multigoal_template_has_exact_edges_and_disjoint_complete_poses(self):
        config = json.loads((Path(__file__).parents[1] / "configs/structure_checks.json").read_text())
        protected = np.array([0, 12345, 23456])
        data = build_template(self.env, config, protected)
        self.assertLessEqual(len(data["states"]), config["model"]["dimension"])
        self.assertEqual(len(np.unique(data["tasks"][:, 1])), config["data"]["goals"])
        used = set(protected)
        for poses in data["poses"].values():
            for row in poses:
                self.assertFalse(used.intersection(row.tolist()))
                used.update(row.tolist())
                np.testing.assert_array_equal(self.env.successors(row[data["tasks"][:, 0]]), row[data["children"]])
                np.testing.assert_array_equal(self.env.distance(row[:, None], row[None, :]), data["distances"])
        exact = torch.tensor(data["distances"], dtype=torch.float32)
        q = F.pad(exact, (0, 128 - len(exact)))[None]
        model = QMap(self.env.feature_dim, 128, 256, 14, "linf")
        pair, scores, parent, edges = relation_values(model, q, data, np.arange(len(data["pairs"])))
        np.testing.assert_array_equal(pair[0].numpy(), data["distances"][tuple(data["pairs"].T)])
        np.testing.assert_array_equal(scores[0].numpy(), data["truth"])
        torch.testing.assert_close(edges, torch.ones_like(edges))
        condition = {**config["conditions"][-1], "edge": 1., "path": 1.}
        q = q.repeat(2, 1, 1).requires_grad_()
        objective(model, q, data, config, condition, np.arange(len(data["pairs"]))).backward()
        self.assertTrue(torch.isfinite(q.grad).all())

    def test_group_norm_preserves_five_point_witness_and_spreads_gradient(self):
        ids = np.array([1, 1574641, 1224721, 629857, 0])
        exact = self.env.distance(ids[:, None], ids[None, :])
        q = torch.zeros(5, 16, 8)
        q[:, :5, 0] = torch.tensor(exact, dtype=torch.float32)
        q = q.flatten(1)
        torch.testing.assert_close(distance(q[:, None], q[None], "max_l1"), torch.tensor(exact).float())
        x = torch.tensor([[1., 2., 0., 0.]], requires_grad=True)
        distance(x, torch.zeros_like(x), "max_l1", 2).sum().backward()
        torch.testing.assert_close(x.grad, torch.tensor([[1., 1., 0., 0.]]))

    def test_129_incompatible_antipodes_require_more_than_128_linf_coordinates(self):
        config = json.loads((Path(__file__).parents[1] / "configs/linf_dimension_witness.json").read_text())
        result = construct(self.env, config, [0, 1, 12345])
        pairs = np.asarray(result["pairs"])
        self.assertFalse(np.isin(pairs, [0, 1, 12345]).any())
        self.assertEqual(len(np.unique(pairs)), 258)
        self.assertTrue((self.env.distance(pairs[:, 0], pairs[:, 1]) == 14).all())
        a, b = pairs.T
        # Both signs of a shared-coordinate witness would imply 28 <= one
        # of these cross sums. Each cross sum is <=26 for distinct pairs.
        same_sign = self.env.distance(a[:, None], b[None]) + self.env.distance(b[:, None], a[None])
        opposite_sign = self.env.distance(a[:, None], a[None]) + self.env.distance(b[:, None], b[None])
        off_diagonal = ~np.eye(len(pairs), dtype=bool)
        self.assertLessEqual(same_sign[off_diagonal].max(), 26)
        self.assertLessEqual(opposite_sign[off_diagonal].max(), 26)
        self.assertEqual(result["uniform_error_lower_bound_below_this_dimension"], .5)

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

    def test_qtm_square_and_far_goal_violate_an_l1_necessary_condition(self):
        # U^4 = identity. A fifth point exposes a global obstruction even
        # though the four-point local square itself embeds exactly in L1.
        cycle = np.array([1, 1574641, 1224721, 629857])
        np.testing.assert_array_equal(self.env.successors(cycle)[:, 0], np.roll(cycle, -1))
        np.testing.assert_array_equal(self.env.distance(cycle[:, None], cycle[None, :]),
                                      [[0, 1, 2, 1], [1, 0, 1, 2], [2, 1, 0, 1], [1, 2, 1, 0]])
        np.testing.assert_array_equal(self.env.distance(cycle, 0), [12, 11, 12, 11])
        ids = np.r_[cycle, 0]
        exact = self.env.distance(ids[:, None], ids[None, :]).astype(float)
        coefficients = np.outer([1, -1, 1, -1, 1], [1, -1, 1, -1, 1])
        self.assertEqual((coefficients * exact).sum() / 2, 2)
        # Every L1 coordinate is an integral of binary threshold cuts.
        # The inequality holds for all 32 cuts, hence for every dimension.
        for mask in range(32):
            x = np.array([(mask >> bit) & 1 for bit in range(5)])
            self.assertLessEqual((coefficients * np.abs(x[:, None] - x[None, :])).sum() / 2, 0)
        # A constructive alternative for this finite metric: coordinates are
        # distances to five anchors. This is an oracle feasibility proof only.
        np.testing.assert_array_equal(np.abs(exact[:, None, :] - exact[None, :, :]).max(-1), exact)

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
            config["training"]["progress_margin_weight"] = 1.
            with self.assertRaisesRegex(ValueError, "Archived loss experiments"):
                train(env, config, data, root, "cpu", "retired-contract")


if __name__ == "__main__":
    unittest.main()
