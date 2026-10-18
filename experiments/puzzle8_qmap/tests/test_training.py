import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
from experiments.task_text_baselines.src import puzzle8 as rules
from experiments.puzzle8_qmap.src.environment import Puzzle8
from experiments.puzzle8_qmap.src.data import build
from experiments.puzzle8_qmap.src.run import train
from experiments.puzzle8_qmap.src.model import MapReader
from experiments.puzzle8_qmap.src.model import QMap, soft_rank_loss
from torch.nn import functional as F


class PuzzleMapTest(unittest.TestCase):
    def test_interrupted_resume_keeps_absolute_steps_sampler_and_adam(self):
        class Cycle:
            size, diameter, name, actions = 8, 4, "puzzle8", ("left", "right")
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
                    pairs_validation=np.array([[2, 3, 1]]), pairs_test=np.array([[5, 7, 2]]),
                    tasks_validation=np.array([[2, 3]]), child_distances_validation=np.array([[0, 2]]),
                    tasks_restoration=pairs[1:, :2])
        config = dict(seed=18, model=dict(dimension=8, hidden=16),
                      training=dict(steps=4, batch_size=32, decision_batch_size=32, cpu_threads=1,
                                    learning_rate=.0001, rank_weight=1., rank_temperature=1., eval_every=1),
                      evaluation=dict(decision_count=1), continuation=dict(maximum_validation_mae_increase=.25),
                      graph_data=dict(descendant_steps=[0, 1, 2], fractions=[.1, .3, .6],
                                      route_fraction=.5, hard_fraction=.2, edge_fraction=.1))
        policy = dict(within_budget=7)
        state = dict(within=np.ones(8, dtype=bool), hard_parents=np.array([1]))
        module = "experiments.puzzle8_qmap.src.run."
        with tempfile.TemporaryDirectory() as temp, contextlib.ExitStack() as stack:
            root = Path(temp)
            for name, value in [("audit", (policy, state)), ("pair_metrics", dict(mae=1., huber=1., by_distance={})),
                                ("ranking_loss", 1.), ("decisions", dict(q=dict(shortest_action_rate=1.))),
                                ("task_indices", np.array([0])), ("intermediate_stage", {}),
                                ("cycle_ranking", []), ("evaluate", {})]:
                stack.enter_context(patch(module + name, return_value=value))
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            model = QMap(**config["model"])
            base_config = {key: value for key, value in config.items() if key != "graph_data"}
            torch.save(dict(model_type="shared_mlp_qmap", actions=list(env.actions), config=base_config,
                            model=model.state_dict(), optimizer=torch.optim.Adam(model.parameters()).state_dict(),
                            source_commit="base", step=100), root / "base.pt")
            # Use the actual model type, keeping the fixture tied to the public checkpoint contract.
            from experiments.puzzle8_qmap.src.model import MODEL_TYPE
            base = torch.load(root / "base.pt", weights_only=False)
            base["model_type"] = MODEL_TYPE
            torch.save(base, root / "base.pt")
            full, interrupted = root / "full", root / "interrupted"
            full.mkdir(); interrupted.mkdir()
            train(env, config, data, full, "cpu", "test", root / "base.pt")
            real_step = torch.optim.Adam.step
            calls = 0
            def interrupt(optimizer, *args, **kwargs):
                nonlocal calls
                calls += 1
                if calls == 3:
                    raise RuntimeError("simulated machine interruption")
                return real_step(optimizer, *args, **kwargs)
            with patch.object(torch.optim.Adam, "step", interrupt), self.assertRaises(RuntimeError):
                train(env, config, data, interrupted, "cpu", "test", root / "base.pt")
            train(env, config, data, interrupted, "cpu", "resumed", interrupted / "last.pt", resume=True)
            a, b = [torch.load(path / "last.pt", weights_only=False) for path in (full, interrupted)]
            self.assertEqual([json.loads(line)["step"] for line in (interrupted / "progress.jsonl").read_text().splitlines()], [1, 2, 3, 4])
            self.assertEqual(a["step"], b["step"])
            self.assertEqual(a["coverage"], b["coverage"])
            for key in a["model"]:
                torch.testing.assert_close(a["model"][key], b["model"][key], rtol=0, atol=0)

    def test_shared_state_encoding_keeps_original_objective_and_gradients(self):
        torch.manual_seed(18)
        model = QMap(dimension=8, hidden=16)
        cached_model = copy.deepcopy(model)
        features = torch.rand(17, 81)
        rows = torch.randint(17, (31, 2))
        tasks = torch.randint(17, (23, 2))
        children = torch.randint(-1, 17, (23, 4))
        children[:, 0] = 0
        valid = children >= 0
        distances = torch.randint(32, (31,)).float()
        labels = torch.randint(32, (23, 4))
        def objective(net, reuse):
            table = net.encode(features) if reuse else None
            def q(ids):
                return table[ids] if reuse else net.encode(features[ids])
            return (F.smooth_l1_loss(net.distance(q(rows[:, 0]), q(rows[:, 1])), distances)
                    + soft_rank_loss(net.distance(q(children.clamp_min(0)), q(tasks[:, 1])[:, None]),
                                     labels, valid, 1.))
        original, cached = objective(model, False), objective(cached_model, True)
        torch.testing.assert_close(original, cached)
        original.backward(); cached.backward()
        for a, b in zip(model.parameters(), cached_model.parameters()):
            torch.testing.assert_close(a.grad, b.grad, atol=1e-6, rtol=1e-4)

    def test_complete_graph_exact_labels_split_and_actual_update(self):
        env = Puzzle8()
        self.assertEqual(env.size, 181440)
        config = json.loads((Path(__file__).parents[1] / "configs/puzzle8.json").read_text())
        config["data"].update(intermediate_goals=[1, 1, 1], eval_per_depth=2)
        config["training"].update(steps=1, batch_size=64, decision_batch_size=64, eval_every=1, cpu_threads=1)
        config["evaluation"].update(decision_count=16, rollout_count=8)
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp)
            data, summary = build(env, config, out)
            self.assertEqual(summary["standard_goal_distance_coverage"], env.size)
            self.assertEqual(int(env.tables[0].max()), 31)
            ref = rules.distances(env.goal_state)
            np.testing.assert_array_equal(env.tables[0], [ref[tuple(map(int, state))] for state in env.positions])
            for goal in summary["goal_ids"]["validation"] + summary["goal_ids"]["test"]:
                ref = rules.distances(tuple(map(int, env.positions[goal])))
                ids = np.arange(0, env.size, 997)
                np.testing.assert_array_equal(env.distance(ids, goal), [ref[tuple(map(int, env.positions[i]))] for i in ids])
            train_pairs = data["pairs_train"]
            keys = lambda rows: np.minimum(rows[:, 0], rows[:, 1]).astype(np.int64) * env.size + np.maximum(rows[:, 0], rows[:, 1])
            train_keys = keys(train_pairs)
            for split in ("validation", "test"):
                self.assertFalse(np.intersect1d(train_keys, keys(data[f"pairs_{split}"])).size)
                tasks = data[f"tasks_{split}"]
                children = data[f"children_{split}"]
                packed = np.column_stack((children.ravel(), np.repeat(tasks[:, 1], 4)))
                self.assertFalse(np.intersect1d(train_keys, keys(packed[packed[:, 0] >= 0])).size)
            # A single real optimizer update and checkpoint roundtrip, not a full fit.
            for key in list(data):
                if key.endswith("_train"):
                    data[key] = data[key][:16]
            with contextlib.redirect_stdout(io.StringIO()):
                result = train(env, config, data, out, "cpu", "test")
            saved = torch.load(out / "best.pt", weights_only=False)
            self.assertEqual(saved["step"], 1)
            self.assertGreaterEqual(saved["coverage"]["pair_passes"], 1)
            reader = MapReader(env, out / "best.pt")
            for state in range(1, 20):
                action, child = reader.choose(state)
                expected = rules.step(tuple(map(int, env.positions[state])), action)
                self.assertEqual(tuple(env.positions[child]), expected)
            self.assertEqual(result["final"]["restoration_seen_states"]["oracle"]["solved"], 8)


if __name__ == "__main__":
    unittest.main()
