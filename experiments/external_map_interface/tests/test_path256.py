import unittest
from collections import Counter
import json
from pathlib import Path

import numpy as np

from experiments.external_map_interface.src.evaluate_path256 import run_trial, summarize
from experiments.external_map_interface.src.interface import graph_prompt
from experiments.external_map_interface.src.q_map import GraphQMap
from experiments.external_map_interface.src.planner import parse_action_id, parse_final_action_id
from experiments.external_map_interface.src.transitions import GraphEnvironment, environment_from_suite
from experiments.qwen_path_blocks.src.prepare_path_suite import prepare


class Path256ContractTest(unittest.TestCase):
    def test_diverse_suite_has_independent_graph_and_balanced_lengths(self):
        root = Path(__file__).resolve().parents[3]
        config = json.loads((root / "experiments/external_map_interface/configs/path256_diverse_suite.json").read_text())
        suite = prepare(config)
        self.assertEqual(Counter(c["reference"]["length"] for c in suite["cases"]),
                         {4: 4, 6: 4, 8: 4, 10: 4})
        endpoints = [node for c in suite["cases"] for node in (c["start"], c["goal"])]
        self.assertEqual(len(endpoints), len(set(endpoints)))
        old_config = json.loads((root / "experiments/qwen_path_blocks/configs/path256_bidirectional_16k.json").read_text())
        old_suite = prepare(old_config)
        self.assertNotEqual(suite["cases"][0]["neighbors"], old_suite["cases"][0]["neighbors"])

    def test_unambiguous_fenced_json_is_recovered(self):
        self.assertEqual(parse_action_id('```json\n{"action_id": 28, "to_node": 251}\n```'), 28)
        with self.assertRaises(ValueError):
            parse_action_id('```json\n{"action_id": "28"}\n```')

    def test_final_action_json_recovers_verbose_instruct_answer(self):
        answer = ('Consider {"action_id": 26} first. Final choice:\n'
                  '```json\n{"action_id": 28}\n```\nThis is the selected move.')
        self.assertEqual(parse_final_action_id(answer), 28)
        with self.assertRaises(json.JSONDecodeError):
            parse_action_id(answer)
        with self.assertRaises(ValueError):
            parse_final_action_id('{"action_id": 26} then {"action_id": "28"}')
        with self.assertRaises(ValueError):
            parse_final_action_id('{"action_id": 26} then {"action_id":')

    def test_suite_order_and_revisit_rule_and_learned_score(self):
        suite = {"cases": [{"node_count": 3, "neighbors": {"0": [1], "1": [2, 0], "2": [1]},
                            "node_order": [2, 0, 1]}]}
        env = environment_from_suite(suite)
        self.assertTrue(np.array_equal(env.actions,
            np.array([[0, 1], [1, 0], [1, 2], [2, 1]])))
        case = {**suite["cases"][0], "start": 0, "goal": 2, "id": "path_undirected_256_00"}
        q_map = GraphQMap(np.array([[0.], [1.], [3.]]),
                          np.array([[1.], [-1.], [2.], [-2.]]))
        prompt = graph_prompt(env, 1, 2, q_map, case=case, path=[0, 1])
        self.assertIn("learned_map_distance_to_goal=0.000000", prompt)
        self.assertIn("Executed node sequence: 0, 1", prompt)
        self.assertIn("2: 1\n0: 1\n1: 2, 0", prompt)
        self.assertNotIn("action_id=1, to_node=0", prompt)
        self.assertNotIn("learned_map_distance", graph_prompt(env, 1, 2, case=case, path=[0, 1]).split("Legal actions")[1])

    def test_repeated_action_fails_and_each_call_has_its_own_seed(self):
        adjacency = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=bool)
        env = GraphEnvironment(adjacency, np.array([[0, 1], [1, 0], [1, 2], [2, 1]]))
        case = {"id": "path_undirected_256_00", "node_count": 3, "start": 0, "goal": 2,
                "neighbors": {"0": [1], "1": [0, 2], "2": [1]}, "node_order": [0, 1, 2]}
        seen = []

        def caller(prompt, *, seed, endpoint):
            seen.append(seed)
            return {"text": '{"action_id": 0}' if len(seen) == 1 else '{"action_id": 1}',
                    "finish_reason": "stop", "completion_tokens": 2}

        config = {"seed": 10, "replicates": 8, "pair_count": 16, "step_limit": 3}
        record = run_trial(case, 1, env, None, caller, config, "local")
        self.assertEqual(seen, [10, 138])
        self.assertEqual(record["failure"], "illegal_or_repeated_action")
        self.assertEqual(record["path"], [0, 1])
        self.assertEqual(summarize([record])["shortest"], 0)


if __name__ == "__main__":
    unittest.main()
