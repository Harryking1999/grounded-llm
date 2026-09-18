import unittest

import numpy as np

from experiments.cml_map_scaling.src.core import (
    action_catalog, cosine_policy, evaluate_policy, geometry, local_update, gradient_update,
    random_graph, rankdata, sample_walks, shortest_distances, train_epoch,
)


class CoreTest(unittest.TestCase):
    def test_local_rule_only_updates_destination_and_action_from_old_error(self):
        q = np.array([[1., 2.], [5., 8.], [9., 10.]])
        v = np.array([[.5, 1.], [7., 9.]])
        q0, v0 = q.copy(), v.copy()
        error = q0[1] - q0[0] - v0[0]
        local_update(q, v, (0, 0, 1), .1, .01)
        np.testing.assert_allclose(q[1], q0[1] - .1 * error)
        np.testing.assert_allclose(v[0], v0[0] + .01 * error)
        np.testing.assert_array_equal(q[[0, 2]], q0[[0, 2]])
        np.testing.assert_array_equal(v[1], v0[1])

    def test_full_gradient_matches_numerical_loss_gradient(self):
        q = np.array([[1., 2.], [5., 8.], [9., 10.]])
        v = np.array([[.5, 1.], [7., 9.]])
        def loss():
            return .5 * np.sum((q[1] - q[0] - v[0]) ** 2)
        numerical = []
        for array in [q, v]:
            grad = np.zeros_like(array)
            for index in np.ndindex(array.shape):
                old = array[index]
                array[index] = old + 1e-5
                plus = loss()
                array[index] = old - 1e-5
                minus = loss()
                array[index] = old
                grad[index] = (plus - minus) / 2e-5
            numerical.append(grad)
        q0, v0 = q.copy(), v.copy()
        gradient_update(q, v, (0, 0, 1), .1, .01)
        np.testing.assert_allclose(q, q0 - .1 * numerical[0])
        np.testing.assert_allclose(v, v0 - .01 * numerical[1])

    def test_repeated_destination_updates_are_not_dropped(self):
        q = np.array([[0.], [1.], [2.]])
        v = np.zeros((2, 1))
        walks = np.array([[[0, 0, 1], [2, 1, 1]]])
        train_epoch(q, v, walks, [0], .1, .01)
        self.assertAlmostEqual(q[1, 0], 1.01)
        self.assertAlmostEqual(v[1, 0], -.011)

    def test_graph_and_walk_action_identity(self):
        adj = random_graph(32, 0)
        actions, outgoing = action_catalog(adj)
        distances = shortest_distances(adj)
        self.assertTrue(np.all(distances >= 0))
        self.assertTrue(np.array_equal(actions[::2], actions[1::2, ::-1]))
        walks = sample_walks(actions, outgoing, 20, 31, 8)
        np.testing.assert_array_equal(actions[walks[:, :, 1]], walks[:, :, [0, 2]])
        np.testing.assert_array_equal(walks[:, 1:, 0], walks[:, :-1, 2])

    def test_rank_ties(self):
        np.testing.assert_array_equal(rankdata([3, 1, 1, 2]), [3, .5, .5, 2])

    def test_goal_reach_and_cycle_are_measured_explicitly(self):
        adj = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=bool)
        actions, outgoing = action_catalog(adj)
        q = np.array([[0., 0.], [1., 0.], [2., 0.]])
        v = q[actions[:, 1]] - q[actions[:, 0]]
        distances = shortest_distances(adj)
        policy = cosine_policy(q, v, actions, outgoing)
        result, _ = evaluate_policy(policy, distances)
        self.assertEqual(result['reached'], 6)
        self.assertEqual(result['shortest'], 6)
        policy[1, 2] = 0  # 0 -> 1 -> 0 loop when the goal is 2.
        result, _ = evaluate_policy(policy, distances)
        self.assertEqual(result['cycle_or_step_cap'], 2)

    def test_exact_transition_fit_does_not_imply_distance_structure(self):
        adj = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=bool)
        actions, _ = action_catalog(adj)
        q = np.eye(3)
        v = q[actions[:, 1]] - q[actions[:, 0]]
        result, _ = geometry(q, v, actions, shortest_distances(adj), np.ones(4))
        self.assertEqual(result['transition_mse_all_actions'], 0)
        self.assertEqual(result['successor_retrieval_accuracy'], 1)
        self.assertGreater(result['latent_pair_min'], 0)
        self.assertIsNone(result['spearman'])


if __name__ == '__main__':
    unittest.main()
