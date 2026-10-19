import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from experiments.blocks_distance_map.src.blocks12_eval import frozen_clear


class ClearingScorer:
    identity = {"checkpoint": "test"}

    def __init__(self, *args):
        pass

    def scores(self, children, goal):
        return np.asarray([child.bit_count() for child in children])


class FrozenClearTest(unittest.TestCase):
    def test_executes_to_empty_without_oracle_and_compares_identical_cases(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cases = [{"id": "a", "source": "3", "goal": "0", "true_distance": 1},
                     {"id": "b", "source": "12", "goal": "0", "true_distance": 1}]
            (root / "cases.json").write_text(json.dumps({"cases": cases}))
            (root / "reference.json").write_text(json.dumps({"rows": [
                {**cases[0], "outcome": "reached"}, {**cases[1], "outcome": "no_candidate"}]}))
            config = {"cases": str(root / "cases.json"), "case_count": 2,
                      "reference_results": str(root / "reference.json")}
            with patch("experiments.blocks_distance_map.src.blocks12_eval.QScorer", ClearingScorer), \
                    patch("experiments.blocks_distance_map.src.blocks12_eval.DistanceOracle",
                          side_effect=AssertionError("Oracle must not participate in execution")):
                result = frozen_clear(config, "test", root / "out", "test", "cpu")
                self.assertEqual(result["success"], 2)
                self.assertEqual(result["matched_reference"],
                                 {"old_success": 1, "new_success": 2, "lost": 0, "gained": 1})
                cases[0]["id"] = "different_question"
                (root / "cases.json").write_text(json.dumps({"cases": cases}))
                with self.assertRaisesRegex(ValueError, "identical ordered"):
                    frozen_clear(config, "test", root / "bad", "test", "cpu")


if __name__ == "__main__":
    unittest.main()
