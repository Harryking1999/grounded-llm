import importlib.util
import json
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[3]
QWEN_SRC = ROOT / "experiments/qwen_path_blocks/src"
SOL_SRC = ROOT / "experiments/sol_dag_blocks/src"
for path in (str(QWEN_SRC), str(SOL_SRC)):
    if path not in sys.path:
        sys.path.insert(0, path)

import prepare_path_suite
from tasks import TASKS


class Path256SuiteTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = json.loads((ROOT / "experiments/qwen_path_blocks/configs/path256_bidirectional_16k.json").read_text())
        cls.suite = prepare_path_suite.prepare(cls.config)

    def test_fixed_contract_and_symmetric_graph(self):
        graph = self.config["graph"]
        self.assertEqual(len(self.suite["cases"]), graph["pair_count"])
        self.assertEqual(self.suite["planned_calls"], self.config["max_calls_per_model"])
        self.assertEqual({case["condition"] for case in self.suite["cases"]}, {"path_undirected_256"})
        for case in self.suite["cases"]:
            neighbors = {int(node): set(values) for node, values in case["neighbors"].items()}
            self.assertEqual(set(neighbors), set(range(graph["node_count"])))
            self.assertTrue(all(len(values) == graph["regular_degree"] for values in neighbors.values()))
            self.assertTrue(all(node in neighbors[other] for node, values in neighbors.items() for other in values))
            self.assertEqual(sorted(case["node_order"]), list(range(graph["node_count"])))

    def test_reference_paths_and_reverse_edges_are_accepted(self):
        task = TASKS["path_undirected_256"]
        for case in self.suite["cases"]:
            verdict = task.judge(case, {"path": case["reference"]["path"], "final_node": case["goal"]})
            self.assertTrue(verdict["pass"])
            self.assertGreaterEqual(verdict["shortest_moves"], self.config["graph"]["minimum_shortest_moves"])
        case = self.suite["cases"][0]
        left = case["start"]
        right = case["neighbors"][str(left)][0]
        reverse = dict(case, start=right, goal=left)
        verdict = task.judge(reverse, {"path": [{"from": right, "to": left}], "final_node": left})
        self.assertTrue(verdict["pass"])

    def test_prompt_names_the_actual_graph_semantics(self):
        prompt = TASKS["path_undirected_256"].prompt(self.suite["cases"][0])
        self.assertIn("256-node undirected graph", prompt)
        self.assertIn("Edges are bidirectional", prompt)


if __name__ == "__main__":
    unittest.main()
