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
from experiments.puzzle_qmap.src.data import cube_pairs
from experiments.puzzle_qmap.src.model import MapReader, edge_difference_loss, progress_margin_loss
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

    def test_progress_supervision_corrects_the_actual_loop_order(self):
        parent = torch.tensor([3.584487], requires_grad=True)
        children = torch.tensor([[3.270956, 4.014374]], requires_grad=True)
        truth = torch.tensor([[4, 2]])
        valid = torch.ones_like(truth, dtype=torch.bool)
        loss = edge_difference_loss(parent, children, torch.tensor([3]), truth, valid)
        loss.backward()
        # Gradient descent raises the wrongly preferred retreat and lowers the
        # overlooked true shortest successor. Goal-specific offsets cancel.
        self.assertLess(float(children.grad[0, 0]), 0)
        self.assertGreater(float(children.grad[0, 1]), 0)
        shifted = edge_difference_loss(parent + 17, children + 17, torch.tensor([3]), truth, valid)
        torch.testing.assert_close(loss, shifted)
        perfect = edge_difference_loss(torch.tensor([3.]), truth.float(), torch.tensor([3]), truth, valid)
        self.assertEqual(float(perfect), 0)

    def test_margin_rejects_a_false_nonterminal_minimum_and_wrong_order(self):
        parent = torch.tensor([3.584487], requires_grad=True)
        children = torch.tensor([[3.270956, 4.014374]], requires_grad=True)
        truth, valid = torch.tensor([[4, 2]]), torch.tensor([[True, True]])
        loss = progress_margin_loss(parent, children, torch.tensor([3]), truth, valid, 1.)
        loss.backward()
        self.assertGreater(float(loss.detach()), 0)
        self.assertLess(float(children.grad[0, 0]), 0)
        self.assertGreater(float(children.grad[0, 1]), 0)
        exact = progress_margin_loss(torch.tensor([3.]), truth.float(), torch.tensor([3]), truth, valid, 1.)
        self.assertEqual(float(exact), 0)
        # At a maximum depth all actions may be shortest; descent still applies.
        all_good = progress_margin_loss(torch.tensor([4.]), torch.tensor([[3., 3.]]),
                                       torch.tensor([4]), torch.tensor([[3, 3]]), valid, 1.)
        self.assertEqual(float(all_good), 0)

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
            ref = root / "reference"
            ref.mkdir()
            np.savez(ref / "data.npz", **reference)
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
            edge = root / "edge"
            edge.mkdir()
            config["training"].update(distance_supervision="edge_difference", selection_metric="shortest_action_rate")
            continued = train(env, config, data, edge, "cpu", "progress-smoke", root / "best.pt")
            records = [json.loads(line) for line in (edge / "progress.jsonl").read_text().splitlines()]
            self.assertIn("validation_edge_difference_huber", records[-1])
            self.assertIn("edge_difference_huber", continued["initial_training_monitor"])
            self.assertIn("earliest maximum", continued["selection"])
            margin = root / "margin"
            margin.mkdir()
            config["training"].update(distance_supervision="absolute", progress_margin_weight=1., progress_margin=1.)
            train(env, config, data, margin, "cpu", "margin-smoke", root / "best.pt")
            record = json.loads((margin / "progress.jsonl").read_text().splitlines()[-1])
            self.assertGreaterEqual(record["progress_margin_loss"], 0)


if __name__ == "__main__":
    unittest.main()
