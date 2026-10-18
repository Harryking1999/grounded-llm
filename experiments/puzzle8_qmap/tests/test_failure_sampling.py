import copy
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch

from experiments.puzzle8_qmap.src.failure_sampling import FailureGraphData
from experiments.puzzle8_qmap.src.graph_data import GraphData
from experiments.puzzle8_qmap.src.model import QMap, MODEL_TYPE
from experiments.puzzle8_qmap.src.policy_audit import audit


class FailureReplayTest(unittest.TestCase):
    def test_training_only_exact_complete_stars_and_replay_resume(self):
        class Cycle:
            size, diameter, actions = 8, 4, ("left", "right")
            next = np.column_stack(((np.arange(8) + 1) % 8, (np.arange(8) - 1) % 8))
            delta = np.abs(np.arange(8)[:, None] - np.arange(8))
            full_distances = np.minimum(delta, 8 - delta)
            def distance(self, source, goal):
                return self.full_distances[np.asarray(goal), np.asarray(source)]
            def features(self, ids):
                return np.eye(81, dtype=np.float32)[ids]
        env = Cycle()
        pairs = np.column_stack((np.arange(8), np.zeros(8, dtype=int), env.distance(np.arange(8), 0)))
        data = dict(pairs_train=pairs, tasks_train=pairs[1:, :2], parent_distances_train=pairs[1:, 2],
                    pairs_validation=np.array([[2, 3, 1]]), pairs_test=np.array([[5, 7, 2]]))
        options = dict(descendant_steps=[0, 1, 2, 4], fractions=[.1, .3, .6],
                       route_fraction=.5, hard_fraction=.2, edge_fraction=.1)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)
            GraphData(env, data, options, np.random.default_rng(19), path)
            repair = dict(structure_cache=str(path / "graph_structure.npz"), fresh_goals_per_refresh=2,
                          errors_per_distance_degree=2, restoration_fraction=.35,
                          descendant_steps=[0, 1, 2, 4], graph_fraction=.5)
            def make():
                return FailureGraphData(env, data, options, np.random.default_rng(19), path, repair)
            sampler = make()
            model = QMap(dimension=8, hidden=16)
            reference = path / "reference.pt"
            torch.save({"task": "puzzle8", "model_type": MODEL_TYPE,
                        "config": {"model": {"dimension": 8, "hidden": 16}},
                        "model": model.state_dict()}, reference)
            repair.update(retention_fraction=.5, retention_checkpoint=str(reference),
                          retention_restore_checkpoint=str(reference))
            alternative = path / "alternative.pt"
            second = QMap(dimension=8, hidden=16)
            torch.save({"task": "puzzle8", "model_type": MODEL_TYPE,
                        "config": {"model": {"dimension": 8, "hidden": 16}},
                        "model": second.state_dict()}, alternative)
            masks = []
            features = torch.as_tensor(env.features(np.arange(8)))
            for refs in (str(reference), str(alternative), [str(reference), str(alternative)]):
                repair["retention_restore_checkpoint"] = refs
                with torch.no_grad():
                    masks.append(sampler.retained_states(model, features, 0, env.next,
                        env.next >= 0, env.distance(np.arange(8), 0)))
            for k in (0, 1):
                np.testing.assert_array_equal(masks[2][k], masks[0][k] | masks[1][k])
            _, state = audit(model, env, "cpu")
            sampler.mine(model, torch.as_tensor(env.features(np.arange(8))), state)
            for pool in (sampler.repair_middle, sampler.repair_restore):
                self.assertTrue(sampler.allowed[pool[:, 0]].all())
                self.assertFalse(np.isin(pool[:, 1], sampler.protected).any())
            pairs = sampler.pairs(1024)
            tasks, children, truth = sampler.tasks(1024)
            np.testing.assert_array_equal(pairs[:, 2], env.distance(pairs[:, 0], pairs[:, 1]))
            np.testing.assert_array_equal(children, env.next[tasks[:, 0]])
            np.testing.assert_array_equal(truth, env.distance(children, tasks[:, 1, None]))
            self.assertTrue(sampler.repair_seen["huber"])
            self.assertTrue(sampler.repair_seen["candidates"])
            self.assertGreater(sampler.retention_draws, 0)
            for pool in (sampler.retention_restore, sampler.retention_middle):
                if pool is not None:
                    self.assertTrue(sampler.allowed[pool[:, 0]].all())
                    self.assertFalse(np.isin(pool[:, 1], sampler.protected).any())
            state = copy.deepcopy(sampler.state_dict())
            wanted = sampler.tasks(128)
            restored = make()
            restored.load_state_dict(state)
            for actual, expected in zip(restored.tasks(128), wanted):
                np.testing.assert_array_equal(actual, expected)
            self.assertEqual(restored.report(), sampler.report())


if __name__ == "__main__":
    unittest.main()
