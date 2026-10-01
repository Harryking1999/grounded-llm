"""Check predicted successors and local action numbering against the environment."""

import unittest

import numpy as np

from experiments.external_map_interface.src.q_map import GraphQMap
from experiments.external_map_interface.src.transitions import GraphEnvironment
from experiments.flamingo_map_reader.src.graph import batch_maps, graph_step


def line_graph():
    adjacency = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=bool)
    actions = np.array([[0, 1], [1, 0], [1, 2], [2, 1]])
    environment = GraphEnvironment(adjacency, actions)
    qmap = GraphQMap(
        q=np.array([[0., 0.], [1., 0.], [2., 0.]]),
        v=np.array([[1., 0.], [-1., 0.], [1., 0.], [-1., 0.]]),
    )
    return environment, qmap


class GraphStepTest(unittest.TestCase):
    def test_local_ids_follow_shuffled_predicted_successors(self):
        environment, qmap = line_graph()
        step = graph_step(environment, qmap, 1, 2, rng=np.random.default_rng(7))
        self.assertFalse(step.done)
        for local_id, action_id in enumerate(step.candidate_actions, 1):
            vector = step.map_batch.vectors[0, local_id + 1].numpy()
            np.testing.assert_allclose(vector, qmap.q[1] + qmap.v[action_id])
            self.assertEqual(step.execute(environment, local_id),
                             (action_id, int(environment.actions[action_id, 1])))
        self.assertEqual(step.candidate_destinations[step.map_minimal_candidates[0] - 1], 2)

    def test_done_uses_real_state_even_when_actions_remain(self):
        environment, qmap = line_graph()
        terminal = graph_step(environment, qmap, 2, 2)
        self.assertTrue(terminal.done)
        self.assertEqual(len(terminal.candidate_actions), 1)
        batched = batch_maps([graph_step(environment, qmap, 1, 2), terminal])
        self.assertEqual(tuple(batched.vectors.shape), (2, 4, 2))
        self.assertFalse(batched.valid[1, 3])


if __name__ == "__main__":
    unittest.main()
