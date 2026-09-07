import copy
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tasks import CONFIG, TASKS
from prepare import prepare
from run import evaluate, transport
from analyze import summarize


class StudyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.suite = prepare()

    def test_fixed_suite_witnesses_and_no_leakage(self):
        self.assertEqual(self.suite["planned_calls"], 384)
        self.assertEqual(self.suite["case_count"], 48)
        seen = set()
        for case in self.suite["cases"]:
            task = TASKS[case["condition"]]
            prompt = task.prompt(case)
            self.assertNotIn("construction_reference", prompt)
            self.assertNotIn("hidden_topological_order", prompt)
            if case["condition"].startswith("blocks"):
                mask = task.from_grid(case["grid"])
                key = task.normalized_key(mask)
                self.assertNotIn(key, seen)
                seen.add(key)
                actions = copy.deepcopy(case["construction_reference"])
                for action in actions:
                    mask = task.apply(mask, action)
                    action["board_after"] = task.to_rows(mask)
                verdict = task.judge(case, {"actions": actions, "final_status": "solved"})
                self.assertTrue(verdict["contract_pass"])
            else:
                self.assertGreaterEqual(case["reference"]["length"], CONFIG["path"]["minimum_shortest_moves"])
                rank = {n: i for i, n in enumerate(case["hidden_topological_order"])}
                self.assertEqual(set(case["node_order"]), set(range(32)))
                for a, vs in case["neighbors"].items():
                    for b in vs:
                        self.assertLess(rank[int(a)], rank[b])
                self.assertTrue(task.judge(case, {"path": case["reference"]["path"], "final_node": case["goal"]})["contract_pass"])

    def test_shapes_no_dominoes_and_four_unique_sz(self):
        shapes = CONFIG["blocks"]["shape_rows"]
        self.assertEqual(len(set(shapes)), 10)
        self.assertEqual([s.count("1") for s in shapes], [3] * 6 + [4] * 4)
        self.assertEqual(set(shapes[6:]), {"011/110", "10/11/01", "110/011", "01/11/10"})

    def test_empty_holes_repeat_bounds_and_state_reports(self):
        task = TASKS["blocks8"]
        action = {"shape_id": 0, "row": 0, "col": 0}
        mask = task.by_action[0, 0, 0]
        case = {"grid": "/".join(task.to_rows(mask))}
        # The 0-cell in the shape mask does not have to be filled.
        self.assertTrue(task.judge(case, {"actions": [action], "final_status": "solved"})["pass"])
        self.assertFalse(task.judge(case, {"actions": [action], "final_status": "solved"})["contract_pass"])
        repeated = task.judge(case, {"actions": [action, action]})
        self.assertEqual(repeated["illegal_action"]["reason"], "already_removed_overlap")
        with self.assertRaisesRegex(ValueError, "initial_empty_overlap"):
            task.apply(mask ^ 1, action, mask ^ 1)
        with self.assertRaisesRegex(ValueError, "out_of_bounds"):
            task.apply(mask, {"shape_id": 0, "row": 9, "col": 9})
        with self.assertRaisesRegex(ValueError, "noninteger"):
            task.apply(mask, {"shape_id": True, "row": 0, "col": 0})
        malformed = {**action, "board_after": [0] * 10}
        self.assertEqual(task.judge(case, {"actions": [malformed]})["state_reports_valid"], 0)

    def test_directed_edges_revisits_nonshortest_and_reports(self):
        task = TASKS["path_dag"]
        case = {"start": 0, "goal": 2, "neighbors": {"0": [1, 2], "1": [2], "2": []}}
        self.assertEqual(task.judge(case, {"path": [{"from": 0, "to": 1}, {"from": 1, "to": 2}]})["failure_type"], "non_shortest")
        self.assertFalse(task.judge(case, {"path": [{"from": 0, "to": 2}], "final_node": 1})["contract_pass"])
        reverse = {"start": 1, "goal": 2, "neighbors": case["neighbors"]}
        self.assertEqual(task.judge(reverse, {"path": [{"from": 1, "to": 0}]})["illegal_move"]["reason"], "nonexistent_directed_edge")
        cycle = {"start": 0, "goal": 2, "neighbors": {"0": [1, 2], "1": [0], "2": []}}
        self.assertEqual(task.judge(cycle, {"path": [{"from": 0, "to": 1}, {"from": 1, "to": 0}]})["illegal_move"]["reason"], "repeated_node")

    def test_truncation_retained_and_weak_json_separated(self):
        record = {"response_status": "incomplete", "response": {"incomplete_details": {"reason": "max_output_tokens"}}}
        self.assertTrue(transport.final_sample(record))
        case = self.suite["cases"][-1]
        raw = json.dumps({"path": case["reference"]["path"], "final_node": case["goal"]})
        verdict = evaluate(case, "```json\n" + raw + "\n```")
        self.assertTrue(verdict["pass"])
        self.assertFalse(verdict["contract_pass"])

    def test_summary_counts_inputs_once_and_truncation_in_pass8(self):
        case = self.suite["cases"][0]
        task = TASKS[case["condition"]]
        mask = task.from_grid(case["grid"])
        actions = copy.deepcopy(case["construction_reference"])
        for action in actions:
            mask = task.apply(mask, action)
            action["board_after"] = task.to_rows(mask)
        raw = json.dumps({"actions": actions, "final_status": "solved"})
        rows = [{"case_id": case["id"], "condition": case["condition"], "replicate": i,
                 "response_status": "completed", "raw_output": raw, "verdict": evaluate(case, raw)} for i in range(1, 8)]
        rows.append({"case_id": case["id"], "condition": case["condition"], "replicate": 8,
                     "response_status": "incomplete", "response": {"incomplete_details": {"reason": "max_output_tokens"}},
                     "verdict": {"pass": False, "failure_type": "budget_truncated"}})
        summary = summarize({"cases": rows, "source_commit": "test", "status": "completed", "api_config": CONFIG["api"]}, {"cases": [case]})["conditions"]["blocks8"]
        self.assertEqual(summary["sample_success_rate"], 7 / 8)
        self.assertEqual(summary["pass_at_8"], 1)
        self.assertEqual(summary["shapes"]["by_id"]["input_construction"]["total"], 8)
        self.assertEqual(summary["shapes"]["by_id"]["output_successful"]["total"], 56)


if __name__ == "__main__":
    unittest.main()
