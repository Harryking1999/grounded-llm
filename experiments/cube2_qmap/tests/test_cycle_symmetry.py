import unittest
import tempfile
from pathlib import Path

import numpy as np
import torch

from experiments.cube2_qmap.src.cycle_supervision import cycle_pack
from experiments.cube2_qmap.src.cycle_trial import align_initialization, close_axis_mask
from experiments.cube2_qmap.src.environment import Cube2
from experiments.cube2_qmap.tests.test_full_state_joint import Ring
from experiments.puzzle_qmap.src.model import MODEL_TYPE, MapReader, QMap


class CycleSymmetryTests(unittest.TestCase):
    def test_output_relabeling_preserves_old_distances_before_projection(self):
        env = Cube2(metric="QTM_URF_6")
        config = {"dimension": 128, "hidden": 32, "metric": "linf"}
        plain = QMap(env.feature_dim, **config, cap=14)
        ids = np.random.default_rng(45).integers(env.size, size=40)
        features = torch.as_tensor(env.features(ids))
        old = plain.encode(features).detach()
        state, report = align_initialization({"model": plain.state_dict(), "model_config": config, "cap": 14},
                                              env, ids, "cpu")
        plain.load_state_dict(state)
        new = plain.encode(features).detach()
        torch.testing.assert_close(plain.distance(old[:20], old[20:]), plain.distance(new[:20], new[20:]))
        torch.testing.assert_close(new[:, 126], old[:, report["dominant_coordinate"]])

    def test_checkpoint_reader_and_protected_symmetry_orbits(self):
        env = Cube2(metric="QTM_URF_6")
        config = {"dimension": 128, "hidden": 32, "metric": "linf",
                  "symmetry": "cube_axis_permutation_average"}
        model = QMap(env.feature_dim, **config, cap=14)
        protected = np.zeros(env.size, dtype=bool)
        protected[123456] = True
        closed = close_axis_mask(env, protected)
        self.assertTrue(closed[env.axis_symmetry(123456, np.arange(6))].all())
        self.assertFalse(closed[0])
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "q.pt"
            torch.save({"model_type": MODEL_TYPE, "task": env.name, "actions": list(env.actions),
                        "model_config": config, "cap": 14, "model": model.state_dict()}, checkpoint)
            reader = MapReader(env, checkpoint)
            expected = model.encode(torch.as_tensor(env.features([0, 123456])))
            torch.testing.assert_close(reader.encode([0, 123456]), expected)

    def test_solved_center_face_symmetry_keeps_distinct_qs(self):
        torch.manual_seed(43)
        env = Cube2(metric="QTM_URF_6")
        model = QMap(192, 128, 32, 14, "linf", symmetry="cube_axis_permutation_average")
        one_turn = env.successors(0)[[0, 2, 4]]  # U, R, F
        q = model.encode(torch.as_tensor(env.features(one_turn)))
        e = model.encode(torch.as_tensor(env.features([0])))
        d = model.distance(q, e)
        torch.testing.assert_close(d, d[0].expand_as(d), atol=2e-6, rtol=0)
        self.assertGreater(float(model.distance(q[0], q[1]).detach()), 1e-5)
        ur = env.step(env.step(0, "U"), "R")
        rf = env.step(env.step(0, "R"), "F")
        two = model.encode(torch.as_tensor(env.features([ur, rf])))
        torch.testing.assert_close(model.distance(two[:1], e), model.distance(two[1:], e), atol=2e-6, rtol=0)

    def test_projection_preserves_pair_distances_without_identifying_states(self):
        torch.manual_seed(42)
        env = Cube2(metric="QTM_URF_6")
        model = QMap(192, 128, 32, 14, "linf", symmetry="cube_axis_permutation_average")
        plain = QMap(192, 128, 32, 14, "linf")
        model.load_state_dict(plain.state_dict(), strict=True)
        ids = np.random.default_rng(42).integers(env.size, size=40)
        bits = torch.as_tensor(env.features(ids))
        q = model.encode(bits)
        self.assertEqual(q.shape, (40, 128))
        self.assertEqual(set(model.state_dict()), set(plain.state_dict()))
        nonidentical = False
        for g in range(6):
            changed_bits = torch.as_tensor(env.features(env.axis_symmetry(ids, g)))
            torch.testing.assert_close(bits[:, model.axis_feature_gather[g]], changed_bits)
            changed = model.encode(changed_bits)
            torch.testing.assert_close(changed, q[:, model.axis_output_action[g]], atol=2e-6, rtol=0)
            torch.testing.assert_close(model.distance(changed[:20], changed[20:]),
                                       model.distance(q[:20], q[20:]), atol=2e-6, rtol=0)
            nonidentical |= bool((changed - q).abs().max() > 1e-5)
        self.assertTrue(nonidentical)
        model.distance(q[:20], q[20:]).mean().backward()
        self.assertGreater(sum(float(p.grad.abs().sum()) for p in model.parameters()), 0)

    def test_linked_packs_have_exact_branch_and_cross_goal_labels(self):
        env = Ring()
        env.size = 40
        env.depth = env.distance(np.arange(env.size), 0)
        heldout = np.zeros(env.size, dtype=bool)
        batch = cycle_pack(env, [[8, 9], [14, 15]], heldout, np.random.default_rng(1))
        self.assertEqual(len(batch["packs"]), 2)
        states = batch["states"]
        pairs = states[batch["pairs"]]
        np.testing.assert_array_equal(batch["labels"], env.distance(*pairs.T))
        children, goals = states[batch["children"]], states[batch["goals"]]
        np.testing.assert_array_equal(batch["truth"], env.distance(children, goals[:, None]))
        for pack in batch["packs"]:
            path = np.asarray(pack["exit_path"])
            np.testing.assert_array_equal(np.diff(env.depth[path].astype(int)), -np.ones(len(path) - 1))
            for parent in pack["cycle"]:
                self.assertTrue(any(goal == 0 and np.array_equal(row, env.successors(parent))
                                    for row, goal in zip(children, goals)))
        self.assertTrue(any(goal == 15 and np.array_equal(row, env.successors(8))
                            for row, goal in zip(children, goals)))
        heldout[5] = True
        protected = cycle_pack(env, [[8, 9], [14, 15]], heldout, np.random.default_rng(1))
        self.assertEqual(len(protected["packs"]), 1)
        self.assertFalse(heldout[protected["states"]].any())


if __name__ == "__main__":
    unittest.main()
