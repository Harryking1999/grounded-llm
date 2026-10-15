"""Both optimal branches must be covered; inference cannot repair bad choices."""
import unittest

import numpy as np
import torch

from experiments.cube2_qmap.src.route_complete import FreeCoordinates, covered_rollout, optimal_closure


class Diamond:
    children = np.array([[1, 2], [0, 3], [0, 3], [1, 2]])
    distances = np.array([[0, 1, 1, 2], [1, 0, 2, 1], [1, 2, 0, 1], [2, 1, 1, 0]])

    def successors(self, ids):
        return self.children[ids]

    def distance(self, source, goal):
        return self.distances[source, goal]


class RouteTests(unittest.TestCase):
    def test_closure_covers_every_optimal_branch(self):
        env = Diamond()
        tasks = optimal_closure(env, [(3, 0)])
        self.assertEqual(set(map(tuple, tasks)), {(3, 0), (1, 0), (2, 0)})
        data = {"tasks": tasks, "children": env.successors(tasks[:, 0]), "distances": env.distances}
        result = covered_rollout(env.distances.astype(float), data, [(3, 0)])
        self.assertEqual(result["solved"], 1)
        self.assertEqual(result["unsupported"], 0)

    def test_unsupported_chosen_branch_counts_as_failure(self):
        env = Diamond()
        tasks = np.array([[3, 0], [1, 0]])  # Deliberately omit second branch.
        data = {"tasks": tasks, "children": env.successors(tasks[:, 0]), "distances": env.distances}
        q = np.array([[0.], [2.], [1.], [3.]])
        result = covered_rollout(q, data, [(3, 0)])
        self.assertEqual(result["solved"], 0)
        self.assertEqual(result["unsupported"], 1)
        self.assertIsNone(result["first_wrong_steps"][0])  # First action was optimal.

    def test_free_initialization_matches_encoder_outputs_and_learns(self):
        initial = torch.tensor([[5., 7., 9.], [6., 8., 10.]])
        model = FreeCoordinates(initial, 14.)
        torch.testing.assert_close(model(), initial, atol=1e-6, rtol=0)
        optimizer = torch.optim.Adam(model.parameters(), lr=.01)
        loss = (model()[0] - model()[1]).abs().amax()
        loss.backward()
        optimizer.step()
        self.assertFalse(torch.equal(model(), initial))


if __name__ == "__main__":
    unittest.main()
