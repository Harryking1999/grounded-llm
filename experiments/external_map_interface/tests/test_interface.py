import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from experiments.external_map_interface.src.evaluate import run_episode
from experiments.external_map_interface.src.interface import graph_prompt
from experiments.external_map_interface.src.q_map import GraphQMap
from experiments.external_map_interface.src.transitions import GraphEnvironment


class DistanceInterfaceTest(unittest.TestCase):
    def test_candidate_uses_q_plus_v_and_actual_feedback(self):
        adjacency = np.array([[False, True, False], [True, False, True],
                              [False, True, False]])
        actions = np.array([[0, 1], [1, 0], [1, 2], [2, 1]])
        env = GraphEnvironment(adjacency, actions)
        q_map = GraphQMap(np.array([[0.], [1.], [3.]]),
                          np.array([[1.], [-1.], [2.], [-2.]]))
        prompt = graph_prompt(env, 0, 2, q_map)
        self.assertIn("learned_map_distance_to_goal=2.000000", prompt)
        self.assertNotIn("learned_map_distance", graph_prompt(env, 0, 2))
        seen = []

        def caller(text):
            seen.append(text)
            return json.dumps({"action_id": 0 if len(seen) == 1 else 2})

        result = run_episode(env, q_map, caller, 0, 2, 3)
        self.assertTrue(result["reached"])
        self.assertEqual([item["actual_next"] for item in result["trace"]], [1, 2])
        self.assertIn("Current node: 1", seen[1])

    def test_loader_ignores_true_distance_array(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "inputs.npz"
            np.savez(path, adjacency=np.array([[0, 1], [1, 0]]),
                     actions=np.array([[0, 1], [1, 0]]),
                     graph_distances=np.array([[0, 999], [999, 0]]))
            env = GraphEnvironment.load(path)
            self.assertEqual(env.execute(0, 0), 1)
            self.assertFalse(hasattr(env, "graph_distances"))


if __name__ == "__main__":
    unittest.main()
