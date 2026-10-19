import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from experiments.blocks_distance_map.src.clear_retention import prepare


class ClearRetentionTest(unittest.TestCase):
    def test_decision_pool_keeps_first_wrong_parent_without_oracle_action_selection(self):
        class Scorer:
            identity, cache = {"test": True}, {}
            def __init__(self, *args): pass
            def scores(self, masks, goal): return np.asarray([0., 1.])
        class Oracle:
            def __init__(self, *args): pass
            def distances(self, masks, goal): return np.asarray([-1, 9])
            def distance(self, source, goal): return 10
        with tempfile.TemporaryDirectory() as temp, patch(
                "experiments.blocks_distance_map.src.clear_retention.QScorer", Scorer), patch(
                "experiments.blocks_distance_map.src.clear_retention.successors", return_value=[(0, 1), (1, 2)]), patch(
                "experiments.blocks_distance_map.src.failure_diagnosis.RootOracle", Oracle), patch(
                "experiments.blocks_distance_map.src.full_graph_data.Root"):
            path = Path(temp)
            (path / "roots.json").write_text(json.dumps([{"initial": "3", "components": []}]))
            (path / "cases.json").write_text(json.dumps({"cases": []}))
            prepare({"graph_data": str(path), "baseline_cases": str(path / "cases.json"),
                     "parent_checkpoint": "mock", "retention_training_roots": 1,
                     "prioritize_training_decisions": True,
                     "training": {"clear_retention_pool": str(path / "pool.json")}})
            result = json.loads((path / "pool.json").read_text())
            self.assertEqual(result["successful_training_roots"], 0)
            self.assertEqual(result["rows"][0]["states"], ["3"])
            self.assertEqual(result["rows"][0]["priorities"], [4.])

    def test_only_successful_training_paths_and_reject_fixed_cases(self):
        class Scorer:
            identity, cache = {"test": True}, {}
            def __init__(self, *args): pass
            def scores(self, masks, goal): return np.zeros(len(masks))

        with tempfile.TemporaryDirectory() as temp, patch(
                "experiments.blocks_distance_map.src.clear_retention.QScorer", Scorer):
            path = Path(temp)
            (path / "roots.json").write_text(json.dumps([{"initial": "3"}, {"initial": "1"}]))
            (path / "cases.json").write_text(json.dumps({"cases": [{"source": "12"}]}))
            config = {"graph_data": str(path), "baseline_cases": str(path / "cases.json"),
                      "parent_checkpoint": "mock", "retention_training_roots": 2,
                      "training": {"clear_retention_pool": str(path / "retained.json")}}
            prepare(config)
            rows = json.loads((path / "retained.json").read_text())["rows"]
            self.assertEqual(rows, [{"board": 0, "initial": "3", "states": ["3"]}])
            (path / "cases.json").write_text(json.dumps({"cases": [{"source": "3"}]}))
            with self.assertRaisesRegex(ValueError, "evaluation root"):
                prepare(config)
