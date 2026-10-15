from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from experiments.hanoi_qmap.src.environment import Hanoi
from experiments.puzzle_qmap.src.evaluate import decisions, pair_metrics, rollout
from experiments.puzzle_qmap.src.model import MODEL_TYPE, MapReader, QMap, distance
from experiments.puzzle_qmap.src.post_eval import first_error
from experiments.cube2_qmap.src.metric_study import validation_gate


class MetricTest(unittest.TestCase):
    def test_fallback_uses_validation_only_and_requires_long_range_success(self):
        def result(rate, long_rate):
            return {"final": {"validation": {"rollout": {"q": {
                "success_rate": rate, "by_start_distance": {
                    "1": {"count": 1, "success_rate": 1.},
                    "7": {"count": 4, "success_rate": long_rate}}}}}}}
        gate = {"minimum_success_rate": .9, "minimum_long_success_rate": .8,
                "long_distance_minimum": 7, "require_not_worse_than_l1": True}
        control = result(.91, .85)
        self.assertTrue(validation_gate(control, result(.95, .9), gate)["passed"])
        self.assertFalse(validation_gate(control, result(.95, .5), gate)["passed"])
        self.assertFalse(validation_gate(control, result(.90, .9), gate)["passed"])

    def test_euclidean_norm_cannot_exactly_realize_a_unit_graph_square(self):
        # A Euclidean distance matrix has a positive semidefinite centered
        # Gram matrix. This graph square has a negative eigenvalue, in any
        # embedding dimension. L2 remains an approximate-ranking candidate.
        square = np.array([[0, 1, 2, 1], [1, 0, 1, 2],
                           [2, 1, 0, 1], [1, 2, 1, 0]], dtype=float)
        center = np.eye(4) - np.ones((4, 4)) / 4
        gram = -.5 * center @ square**2 @ center
        self.assertLess(np.linalg.eigvalsh(gram).min(), -.99)

    def test_norms_broadcast_and_have_finite_gradients_at_zero(self):
        a = torch.tensor([[1., 1., 1.], [3., 0., 0.]], requires_grad=True)
        goal = torch.zeros(3)
        torch.testing.assert_close(distance(a, goal, "linf"), torch.tensor([1., 3.]))
        torch.testing.assert_close(distance(a, goal, "l2"), torch.tensor([3.**.5, 3.]))
        for metric in ("l1", "linf", "l2"):
            x = torch.zeros(2, 3, requires_grad=True)
            distance(x, goal, metric).sum().backward()
            self.assertTrue(torch.isfinite(x.grad).all())
            torch.testing.assert_close(distance(a, goal, metric), distance(goal, a, metric))
        with self.assertRaises(ValueError):
            distance(a, goal, "other")

    def test_metric_does_not_change_initial_encoder_weights(self):
        weights = []
        for metric in ("l1", "linf", "l2"):
            torch.manual_seed(91)
            weights.append(QMap(9, 128, 256, 14, metric).state_dict())
        for key in weights[0]:
            self.assertTrue(key.startswith("network."))
            for other in weights[1:]:
                torch.testing.assert_close(weights[0][key], other[key], rtol=0, atol=0)

    def test_checkpoint_distance_is_restored_and_old_models_default_to_l1(self):
        env = Hanoi(2)
        with tempfile.TemporaryDirectory() as tmp:
            for metric in ("l1", "linf", "l2", "max_l1"):
                config = {"dimension": 4, "hidden": 8}
                if metric == "max_l1":
                    config["group_size"] = 2
                if metric != "l1":
                    config["metric"] = metric
                model = QMap(env.feature_dim, **config, cap=env.diameter)
                path = Path(tmp) / (metric + ".pt")
                torch.save({"model_type": MODEL_TYPE, "task": env.name, "actions": list(env.actions),
                            "model_config": config, "cap": env.diameter, "model": model.state_dict()}, path)
                reader = MapReader(env, path)
                self.assertEqual(reader.model.metric, metric)
                q = reader.encode([0, 1, 8])
                np.testing.assert_allclose(reader.scores([0, 1], [8, 8]),
                                           distance(q[:2], q[2], metric, config.get("group_size", 8)).numpy(), rtol=0, atol=0)

    def test_decisions_rollout_and_audit_use_the_models_distance(self):
        # A reversible path: 2 -- 0 -- 1 -- 3(goal). The readout choice differs
        # between sum and maximum although all encoder outputs are identical.
        graph = np.array([[1, 2], [0, 3], [0, 0], [1, 1]])
        true_depth = np.array([2, 1, 3, 0])
        env = SimpleNamespace(diameter=3,
            features=lambda ids: np.asarray(ids)[..., None],
            successors=lambda ids: graph[np.asarray(ids)],
            distance=lambda source, goal: true_depth[np.asarray(source)],
            baseline=lambda source, goal: np.zeros_like(source))
        table = torch.tensor([[1., 0., 1.], [1., 1., 1.], [2., 0., 0.], [0., 0., 0.]])
        tasks = np.array([[0, 3]])
        for metric, solved in (("l1", 0), ("linf", 1), ("l2", 1)):
            model = QMap(1, 3, 4, 10, metric)
            model.encode = lambda features: table[features[..., 0].long()]
            self.assertEqual(decisions(model, env, tasks, "cpu")["q"]["shortest_action_rate"], solved)
            self.assertEqual(rollout(model, env, tasks, "cpu", 2)["q"]["solved"], solved)
            self.assertEqual(first_error(model, env, tasks, "cpu")["all_steps_shortest"], solved)
            metrics = pair_metrics(model, env, np.array([[1, 3, 1], [2, 3, 3]]), "cpu")
            expected = torch.nn.functional.smooth_l1_loss(
                distance(table[[1, 2]], table[3], metric), torch.tensor([1., 3.]))
            self.assertAlmostEqual(metrics["huber"], float(expected), places=6)


if __name__ == "__main__":
    unittest.main()
