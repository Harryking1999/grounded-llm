from types import SimpleNamespace
import unittest

import numpy as np
import torch

from experiments.hanoi_qmap.src.environment import Hanoi
from experiments.puzzle_qmap.src.post_eval import first_error


class DiagnosticTest(unittest.TestCase):
    def test_first_error_counts_goal_and_same_layer_mistakes(self):
        env = Hanoi(1)
        tasks = np.array([[0, 1], [0, 2], [1, 2]])
        exact_on_this_graph = SimpleNamespace(encode=lambda features: features)
        result = first_error(exact_on_this_graph, env, tasks, "cpu")
        self.assertEqual(result["all_steps_shortest"], 3)
        collapsed = SimpleNamespace(encode=lambda features: torch.zeros_like(features))
        result = first_error(collapsed, env, tasks, "cpu")
        self.assertEqual(result["all_steps_shortest"], 1)
        self.assertEqual(result["first_error_type"], {"same_layer": 2, "away": 0})


if __name__ == "__main__":
    unittest.main()
