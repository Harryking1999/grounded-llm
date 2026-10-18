import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch
from experiments.task_text_baselines.src import puzzle8 as rules
from experiments.puzzle8_qmap.src.environment import Puzzle8
from experiments.puzzle8_qmap.src.data import build
from experiments.puzzle8_qmap.src.run import train
from experiments.puzzle8_qmap.src.model import MapReader


class PuzzleMapTest(unittest.TestCase):
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
