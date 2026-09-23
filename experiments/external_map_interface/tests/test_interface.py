import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from experiments.external_map_interface.src.evaluate import run_episode, summarize
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

    def test_truncation_and_illegal_answer_remain_auditable(self):
        adjacency = np.array([[False, True], [True, False]])
        env = GraphEnvironment(adjacency, np.array([[0, 1], [1, 0]]))
        truncated = run_episode(env, None, lambda _: {
            "text": "<think>unfinished", "finish_reason": "length",
            "completion_tokens": 256}, 0, 1, 2)
        invalid = run_episode(env, None, lambda _: '{"action_id": 99}', 0, 1, 2)
        for record in (truncated, invalid):
            self.assertEqual(len(record["trace"]), 1)
            self.assertIn("prompt", record["trace"][0])
            self.assertIn("response", record["trace"][0])
            record["moves"] = sum("actual_next" in step for step in record["trace"])
            record["shortest_moves"] = 1
        self.assertEqual(truncated["failure"], "budget_truncated")
        self.assertEqual(invalid["failure"], "invalid_action")
        self.assertEqual(summarize([truncated, invalid])["reach_rate"], 0)
        self.assertEqual(summarize([truncated, invalid])["budget_truncated"], 1)
        self.assertEqual(summarize([truncated, invalid])["invalid_action_rate"], 0.5)


if __name__ == "__main__":
    unittest.main()
