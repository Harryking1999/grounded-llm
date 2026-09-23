"""Continuation must retain failures and truncations without sampling for success."""
import contextlib
import copy
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "experiments/qwen_path_blocks/src"))
import run_api_pilot as runner


class ApiPilotTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = json.loads((ROOT / "experiments/qwen_path_blocks/configs/sol_path256_pilot.json").read_text())
        cls.cases = [
            {"id": f"case_{n}", "condition": "path_undirected_256", "node_count": 2,
             "start": 0, "goal": 1, "neighbors": {"0": [1], "1": [0]}, "node_order": [0, 1],
             "replicates": 8, "reference": {"length": 1, "path": [{"from": 0, "to": 1}]}}
            for n in range(2)]

    def record(self, case, correct=True, replicate=1):
        raw = json.dumps({"path": case["reference"]["path"] if correct else [], "final_node": 1})
        return {"case_id": case["id"], "replicate": replicate, "response_status": "completed",
                "request": runner.request_body(case, self.config["api"]), "raw_output": raw,
                "response": {"status": "completed"}, "elapsed_seconds": 0,
                "verdict": runner.evaluate(case, raw, runner.TASKS)}

    def test_resume_counts_wrong_answers_and_truncation_and_retries_only_service_error(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            config = copy.deepcopy(self.config)
            config.update(case_ids=[c["id"] for c in self.cases], target_samples=3,
                          suite=str(folder / "suite.json"), qwen_contract=str(folder / "qwen.json"))
            suite = {"config": {}, "cases": self.cases}
            (folder / "suite.json").write_text(json.dumps(suite))
            (folder / "qwen.json").write_text("{}")
            (folder / "config.json").write_text(json.dumps(config))
            prior = {"config": self.config, "cases": [self.record(self.cases[0]), self.record(self.cases[1], False)]}
            (folder / "prior.json").write_text(json.dumps(prior))
            error = {"case_id": "case_0", "replicate": 2, "http_status": 503, "api_error": "HTTPError",
                     "request": runner.request_body(self.cases[0], config["api"]), "elapsed_seconds": 0,
                     "verdict": {"pass": False, "failure_type": "api_error"}}
            truncated = self.record(self.cases[0], replicate=2)
            truncated.update(response_status="incomplete", response={"incomplete_details": {"reason": "max_output_tokens"}},
                             verdict={"pass": False, "failure_type": "incomplete_response"})
            argv = ["pilot", "--config", str(folder / "config.json"), "--out", str(folder / "out"),
                    "--continue-from", str(folder / "prior.json")]
            with patch.object(sys, "argv", argv), patch.dict(runner.os.environ, {"GND_API_KEY": "test-only"}), \
                 patch.object(runner, "validate_suite"), patch.object(runner, "sample", side_effect=[error, truncated]) as calls, \
                 contextlib.redirect_stdout(io.StringIO()):
                runner.main()
            result = json.loads((folder / "out/run.json").read_text())
            self.assertEqual(result["status"], "completed")
            self.assertEqual(result["cases"][:2], prior["cases"])
            self.assertEqual(len(result["cases"]), 3)
            self.assertEqual(sum(r["verdict"]["pass"] for r in result["cases"]), 1)
            self.assertEqual(len(result["prior_failed_attempts"]), 1)
            self.assertEqual(calls.call_count, 2)
            self.assertEqual([c.args[3] for c in calls.call_args_list], [2, 2])

    def test_changed_prompt_is_rejected_and_unknown_incomplete_is_not_retryable(self):
        record = self.record(self.cases[0])
        record["request"]["input"] += "Changed instructions"
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "prior.json"
            path.write_text(json.dumps({"config": self.config, "cases": [record]}))
            with self.assertRaisesRegex(ValueError, "changes a trial"):
                runner.retain_previous([str(path)], {c["id"]: c for c in self.cases},
                                       self.config["api"], {("case_0", 1)})
        record.update(response_status="incomplete", response={"incomplete_details": {"reason": "unknown"}})
        self.assertFalse(runner.retryable(record))
        self.assertFalse(runner.transport.final_sample(record))
        record.update(response_status="in_progress", stream_event_counts={"error": 1})
        self.assertTrue(runner.retryable(record))


if __name__ == "__main__":
    unittest.main()
