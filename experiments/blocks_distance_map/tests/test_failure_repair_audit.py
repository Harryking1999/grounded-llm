import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from experiments.blocks_distance_map.src.failure_repair_audit import key, run, target_mass


class RepairAuditTest(unittest.TestCase):
    def test_larger_unreachable_label_reduces_target_mass_on_wrong_actions(self):
        star = {"true_distance": 13, "candidates": [{"truth": 12}, {"truth": -1}]}
        result = target_mass([star])
        self.assertAlmostEqual(result["mean"][0], 1 / (1 + np.exp(-1)))
        self.assertAlmostEqual(result["mean"][2], 1 / (1 + np.exp(-7)))

    def test_global_union_and_exact_old_error_membership(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            state = {"failure_repair": {"pairs_seen": {key(3, 1)}, "forks_seen": {key(3, 1)},
                     "candidates_seen": {key(1, 1), key(2, 1)}},
                     "ranking_gradient_counts": {"contexts_with_nonzero_score_gradient": 5}}
            for rank in (0, 1):
                torch.save(state, path / f"sampler_001000_rank{rank}.pt")
            failure = {"source": "3", "goal": "1", "true_distance": 1,
                       "candidates": [{"state": "1", "truth": 0}, {"state": "2", "truth": -1}]}
            section = {"rows": [{"success": False, "failure": failure}]}
            (path / "old.json").write_text(json.dumps(dict(training_long=section, monitor_long=section)))
            class Scores:
                def scores(self, masks, goal):
                    return np.asarray([0., 19.])
            with patch("experiments.blocks_distance_map.src.failure_repair_audit.QScorer", return_value=Scores()):
                run(path, 1000, path / "checkpoint.pt", path / "old.json", path / "audit.json")
            result = json.loads((path / "audit.json").read_text())
            self.assertEqual(result["global_distinct_repair_huber_relations"], 1)
            self.assertEqual(result["global_distinct_repair_candidate_relations"], 2)
            self.assertEqual(result["ranking_score_gradient_draws"]["contexts_with_nonzero_score_gradient"], 10)
            self.assertEqual(result["training_long"]["old_wrong_fork_now_shortest"], 1)


if __name__ == "__main__":
    unittest.main()
