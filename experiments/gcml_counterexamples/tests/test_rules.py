import itertools
import io
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import blocks
import path as path_task
from run import aggregate, prompt_for, read_response


class BlocksRules(unittest.TestCase):
    def test_solver_against_exhaustive_small_grid(self):
        # Independent brute-force removal search on every 2x3 binary board.
        def brute(occupied):
            if not occupied:
                return 0
            best = None
            for offsets in blocks.SHAPES:
                for r in range(2):
                    for c in range(3):
                        tile = {(r + dr, c + dc) for dr, dc in offsets}
                        if tile <= occupied:
                            child = brute(occupied - tile)
                            if child is not None:
                                best = min(best if best is not None else 100, child + 1)
            return best
        solve = blocks.solver()
        for flags in itertools.product([0, 1], repeat=6):
            occupied = {(i // 3, i % 3) for i, flag in enumerate(flags) if flag}
            mask = sum(1 << (r * 10 + c) for r, c in occupied)
            self.assertEqual(solve(mask)[0], brute(occupied))

    def test_alternative_decomposition_and_budget(self):
        mask = sum(1 << i for i in [0, 1, 2, 10, 11, 12])
        horizontal = [{"shape_id": 7, "row": r, "col": 0} for r in [0, 1]]
        vertical = [{"shape_id": 4, "row": 0, "col": c} for c in [0, 1, 2]]
        self.assertTrue(blocks.judge(blocks.to_grid(mask), horizontal, 3)["pass"])
        self.assertTrue(blocks.judge(blocks.to_grid(mask), vertical, 3)["pass"])
        self.assertFalse(blocks.judge(blocks.to_grid(mask), vertical, 2)["pass"])

    def test_illegal_action_stops_without_removing_cells(self):
        mask = (1 << 9) | (1 << 19)
        verdict = blocks.judge(blocks.to_grid(mask), [{"shape_id": 2, "row": 0, "col": 9}], 8)
        self.assertEqual(verdict["failure_type"], "illegal_action")
        self.assertEqual(verdict["remaining_cells"], 2)
        self.assertEqual(verdict["executed_actions"], 0)

    def test_legal_action_creates_dead_end(self):
        mask = sum(1 << i for i in [0, 1, 2, 3])
        verdict = blocks.judge(blocks.to_grid(mask), [{"shape_id": 7, "row": 0, "col": 0}], 2)
        self.assertTrue(verdict["legal"])
        self.assertEqual(verdict["failure_type"], "dead_end")
        self.assertEqual(verdict["first_irrecoverable_action"]["step"], 1)

    def test_uncapped_decomposition_can_exceed_construction_count(self):
        # Two horizontal triples can instead be cleared using three vertical pairs.
        mask = sum(1 << i for i in [0, 1, 2, 10, 11, 12])
        actions = [{"shape_id": 4, "row": 0, "col": c} for c in range(3)]
        verdict = blocks.judge(blocks.to_grid(mask), actions)
        self.assertTrue(verdict["pass"])
        self.assertIsNone(verdict["first_irrecoverable_action"])
        self.assertEqual(verdict["minimum_actions"], 2)
        self.assertEqual(verdict["action_efficiency_ratio"], 1.5)
        prompt = prompt_for({"condition": "blocks8", "grid": blocks.to_grid(mask), "budget": None})
        self.assertIn("There is no action-count limit", prompt)
        self.assertNotIn("at most", prompt)

    def test_original_pilot_verdicts_and_loss_points(self):
        root = Path(__file__).resolve().parents[3]
        original = root / "experiments/gcml_direct_api_blocks/runs/clawnode_luna_20260904_16/run.json"
        if not original.exists():
            self.skipTest("Local historical run not present")
        from run import parse_output
        cases = json.loads(original.read_text(encoding="utf-8-sig"))["cases"]
        for case in cases:
            output, _ = parse_output(case["raw_output"])
            verdict = blocks.judge(case["input_grid"], output["actions"], 8)
            self.assertEqual(verdict["pass"], case["verdict"]["pass"])
            if case["case"] in (14, 15):
                self.assertEqual(verdict["first_irrecoverable_action"]["step"], {14: 6, 15: 3}[case["case"]])


class PathRules(unittest.TestCase):
    def setUp(self):
        self.case = {"id": "gate", "condition": "path_gates1", "objective": "shortest",
                     "neighbors": {"0": [1, 2], "1": [0], "2": [0, 3], "3": [2]},
                     "start": 0, "goal": 3, "bits": 1, "initial_mask": 0,
                     "switches": [{"node": 1, "bit": 0}],
                     "gates": [{"edge": [0, 2], "bit": 0}], "step_limit": None}

    def test_revisits_need_product_state(self):
        reference = path_task.shortest(self.case)
        self.assertEqual(reference["length"], 4)
        self.assertEqual([s["node"] for s in reference["states"]], [0, 1, 0, 2, 3])
        self.assertTrue(path_task.judge(self.case, {"path": reference["path"]})["pass"])
        opened = {**self.case, "initial_mask": 1}
        self.assertEqual(path_task.shortest(opened)["length"], 2)

    def test_gate_checked_before_arrival_flip_and_both_directions(self):
        self.case["switches"] = [{"node": 2, "bit": 0}]
        with self.assertRaises(ValueError):
            path_task.transition(self.case, 0, 0, 2)
        self.assertEqual(path_task.transition(self.case, 0, 1, 2), (2, 0))
        with self.assertRaises(ValueError):
            path_task.transition(self.case, 2, 0, 0)

    def test_start_does_not_flip_and_reentry_does(self):
        self.case["switches"] = [{"node": 0, "bit": 0}]
        self.assertEqual(path_task.shortest(self.case)["length"], 4)

    def test_shortestness_and_reports_separate_from_execution(self):
        moves = [{"from": a, "to": b, "switches_after": [0]}
                 for a, b in zip([0, 1, 0, 1, 0, 1, 0, 2], [1, 0, 1, 0, 1, 0, 2, 3])]
        verdict = path_task.judge(self.case, {"path": moves})
        self.assertTrue(verdict["execution_pass"])
        self.assertFalse(verdict["pass"])
        self.assertEqual(verdict["failure_type"], "non_shortest")


class Statistics(unittest.TestCase):
    def test_stream_uses_final_response_and_detects_incomplete_stream(self):
        events = [{"type": "response.created", "response": {"status": "in_progress"}},
                  {"type": "response.output_text.delta", "delta": "partial"},
                  {"type": "response.completed", "response": {"status": "completed", "output": []}}]
        data = b"".join(("data: " + json.dumps(event) + "\n\n").encode() for event in events)
        response, counts = read_response(io.BytesIO(data), True)
        self.assertEqual(response["status"], "completed")
        self.assertEqual(counts["response.output_text.delta"], 1)
        data = ("data: " + json.dumps(events[0]) + "\n\n").encode()
        self.assertEqual(read_response(io.BytesIO(data), True)[0]["status"], "in_progress")

    def test_pass8_grouping_not_sample_average(self):
        cases = [{"id": "a", "replicates": 8}, {"id": "b", "replicates": 8}]
        rows = [{"case_id": c["id"], "condition": "blocks8", "response_status": "completed",
                 "verdict": {"pass": c["id"] == "a" and i == 0}} for c in cases for i in range(8)]
        report = aggregate(rows, cases)["blocks8"]
        self.assertEqual(report["sample_success_rate_all_attempts"], 1 / 16)
        self.assertEqual(report["pass_at_8"], .5)
        rows[-1]["api_error"] = "HTTP 500"
        self.assertEqual(aggregate(rows, cases)["blocks8"]["valid_pass8_case_groups"], 1)


if __name__ == "__main__":
    unittest.main()
