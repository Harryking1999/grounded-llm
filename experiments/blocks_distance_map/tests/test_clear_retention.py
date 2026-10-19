import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from experiments.blocks_distance_map.src.clear_retention import prepare


class ClearRetentionTest(unittest.TestCase):
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
