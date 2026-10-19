import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from experiments.blocks_distance_map.src.blocks12_eval import construct
from experiments.blocks_distance_map.src.blocks12_sealed import excluded_masks, final_evaluate, prepare, validate_cases


class SealedPreparationTest(unittest.TestCase):
    def test_final_comparison_uses_explicit_models_and_real_tiny_rollout(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)
            (path / "excluded.json").write_text("[]")
            config = {"boards": 1, "construction_placements": 1, "root_seed": 73,
                      "path_seed": 74, "intermediate_path_positions": [],
                      "excluded_training_roots": str(path / "excluded.json"),
                      "excluded_monitoring_cases": str(path / "excluded.json"),
                      "cpu_threads": 1, "oracle_cache_limit": 1000,
                      "oracle_seconds": 5, "policy": "pure_greedy"}
            prepare(config, path / "prepared", "prepared-source")
            def scorer(filename, device):
                return SimpleNamespace(identity={"checkpoint": str(filename)},
                                       model=SimpleNamespace(cap=19),
                                       scores=lambda masks, goal: np.zeros(len(masks)))
            with patch("experiments.blocks_distance_map.src.blocks12_sealed.QScorer",
                       side_effect=scorer) as loader:
                result = final_evaluate(path / "prepared", "candidate.pt", path / "result",
                                        "analysis-source", parent_checkpoint="parent.pt")
            self.assertEqual([call.args[0] for call in loader.call_args_list],
                             ["parent.pt", "candidate.pt"])
            self.assertEqual(result["prepared_source_commit"], "prepared-source")
            self.assertEqual(len(result["models"]["final_candidate"]["rows"]), 1)
            self.assertIsNone(result["models"]["final_candidate"]["sections"]["intermediate"]["success_rate"])
            self.assertEqual(json.loads((path / "result/summary.json").read_text()), result)

    def test_excludes_union_of_current_and_legacy_case_formats(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)
            candidates = construct(4, 12, 73)
            (path / "training.json").write_text(json.dumps([
                {**candidates[0], "initial": int(candidates[0]["initial"])}]))
            (path / "monitoring.json").write_text(json.dumps({"roots": candidates[1:2]}))
            (path / "clear1000.json").write_text(json.dumps({"cases": [
                {"source": candidates[2]["initial"], "goal": "0"}]}))
            monitoring = [str(path / "monitoring.json"), str(path / "clear1000.json")]
            self.assertEqual(excluded_masks(monitoring),
                             {int(r["initial"]) for r in candidates[1:3]})
            config = {"boards": 1, "construction_placements": 12, "root_seed": 73,
                      "path_seed": 74, "intermediate_path_positions": [],
                      "excluded_training_roots": str(path / "training.json"),
                      "excluded_monitoring_cases": monitoring}
            summary = prepare(config, path / "prepared", "committed-source")
            self.assertEqual(summary["excluded_monitoring_roots"], 2)
            self.assertEqual(summary["clear_tasks"], 1)
            self.assertEqual(summary["training_root_overlap"], 0)
            self.assertEqual(summary["monitoring_root_overlap"], 0)
            roots = json.loads((path / "prepared/cases.json").read_text())["roots"]
            self.assertEqual(roots[0]["initial"], candidates[3]["initial"])

    def test_excludes_both_sets_replays_every_goal_and_loads_no_model(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)
            candidates = construct(4, 12, 73)
            (path / "training.json").write_text(json.dumps(candidates[:1]))
            (path / "monitoring.json").write_text(json.dumps({"roots": candidates[1:2]}))
            config = {"boards": 2, "construction_placements": 12, "root_seed": 73,
                      "path_seed": 74, "intermediate_path_positions": [1, 4, 7, "last"],
                      "excluded_training_roots": str(path / "training.json"),
                      "excluded_monitoring_cases": str(path / "monitoring.json")}
            with patch("experiments.blocks_distance_map.src.blocks12_sealed.QScorer",
                       side_effect=AssertionError("Preparation loaded a model")):
                first = prepare(config, path / "first", "committed-source")
                prepare(config, path / "second", "committed-source")
            self.assertEqual(first["training_root_overlap"], 0)
            self.assertEqual(first["monitoring_root_overlap"], 0)
            self.assertEqual(first["boards"], 2)
            self.assertFalse(first["models_evaluated"])
            self.assertEqual((path / "first/cases.json").read_text(),
                             (path / "second/cases.json").read_text())
            cases = json.loads((path / "first/cases.json").read_text())
            self.assertEqual({r["initial"] for r in cases["roots"]},
                             {r["initial"] for r in candidates[2:]})
            validate_cases(cases["roots"], cases["rows"], 12)
            damaged = copy.deepcopy(cases["rows"])
            damaged[0]["goal"] = "1"
            with self.assertRaisesRegex(ValueError, "certificate"):
                validate_cases(cases["roots"], damaged, 12)


if __name__ == "__main__":
    unittest.main()
