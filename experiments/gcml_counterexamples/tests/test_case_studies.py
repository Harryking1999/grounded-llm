"""Checks for misleading shortest-path diagnoses, independent of saved examples."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from case_studies import path_diagnostics


def output(nodes):
    return {"path": [{"from": a, "to": b} for a, b in zip(nodes, nodes[1:])]}


class PathCaseDiagnosis(unittest.TestCase):
    def test_different_shortest_witness_is_not_a_loss(self):
        case = {"objective": "shortest", "start": 0, "goal": 4,
                "neighbors": {"0": [1, 2], "1": [0, 3], "2": [0, 3], "3": [1, 2, 4], "4": [3]}}
        tied = path_diagnostics(case, output([0, 2, 3, 4]))
        self.assertEqual(tied["reference"]["path"][0]["to"], 1)
        self.assertIsNone(tied["first_optimality_loss_step"])
        longer = path_diagnostics(case, output([0, 2, 0, 1, 3, 4]))
        self.assertEqual(longer["first_optimality_loss_step"], 2)
        self.assertEqual(longer["slack"], [0, 2, 0, 0, 0])

    def test_blocked_arrival_is_not_a_state_and_restart_does_not_flip(self):
        case = {"objective": "shortest", "start": 0, "goal": 3, "bits": 1, "initial_mask": 0,
                "neighbors": {"0": [1, 2], "1": [0], "2": [0, 3], "3": [2]},
                "switches": [{"node": 0, "bit": 0}], "gates": [{"edge": [2, 3], "bit": 0}]}
        result = path_diagnostics(case, output([0, 2, 3]))
        self.assertEqual(result["reference"]["length"], 4)
        self.assertEqual(result["states"], [{"node": 0, "mask": 0}, {"node": 2, "mask": 0}])
        self.assertEqual(result["focus_state"], {"node": 2, "mask": 0})
        self.assertEqual(result["blocked_gates"], [{"edge": [2, 3], "bit": 0}])
        self.assertEqual(result["recovery_suffix"]["length"], 3)


if __name__ == "__main__":
    unittest.main()
