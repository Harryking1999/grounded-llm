from pathlib import Path
import tempfile
import unittest
import numpy as np
from experiments.puzzle8_qmap.src.graph_data import StreamEpoch, GraphData


class GraphDataTest(unittest.TestCase):
    def test_large_space_stream_is_a_complete_permutation(self):
        for size in [1, 2, 3, 17, 257, 1031]:
            sampler = StreamEpoch(size, np.random.default_rng(18))
            for _ in range(2):
                rows = np.r_[sampler.draw(size // 3), sampler.draw(size - size // 3)]
                np.testing.assert_array_equal(np.sort(rows), np.arange(size))
            self.assertEqual(sampler.passes, 2)

    def test_all_endpoints_exact_full_stars_and_complete_downstream(self):
        class Cycle:
            size, diameter = 8, 4
            next = np.column_stack(((np.arange(8) + 1) % 8, (np.arange(8) - 1) % 8))
            delta = np.abs(np.arange(8)[:, None] - np.arange(8))
            full_distances = np.minimum(delta, 8 - delta)

            def distance(self, source, goal):
                return self.full_distances[np.asarray(goal), np.asarray(source)]

        env = Cycle()
        pairs = np.column_stack((np.arange(8), np.zeros(8, dtype=int), env.distance(np.arange(8), 0)))
        data = dict(pairs_train=pairs, tasks_train=pairs[1:, :2], parent_distances_train=pairs[1:, 2],
                    pairs_validation=np.array([[2, 3, 1]]), pairs_test=np.array([[5, 7, 2]]))
        options = dict(descendant_steps=[0, 1, 2, 4], fractions=[.1, .3, .6],
                       route_fraction=.5, hard_fraction=.2, edge_fraction=.1)
        with tempfile.TemporaryDirectory() as temp:
            sampler = GraphData(env, data, options, np.random.default_rng(19), Path(temp))
            supported = set(map(tuple, sampler.routes))
            for source, goal in sampler.routes:
                for child in env.next[source]:
                    if (child != goal and sampler.allowed[child] and
                            env.distance(child, goal) == env.distance(source, goal) - 1):
                        self.assertIn((child, goal), supported)
            sampler.refresh(np.array([1, 5]))
            rows = sampler.pairs(2048)
            np.testing.assert_array_equal(rows[:, 2], env.distance(rows[:, 0], rows[:, 1]))
            tasks, children, truth = sampler.tasks(2048)
            np.testing.assert_array_equal(children, env.next[tasks[:, 0]])
            np.testing.assert_array_equal(truth, env.distance(children, tasks[:, 1, None]))
            self.assertEqual(sampler.gp.size, len(sampler.sources) ** 2)
            self.assertEqual(sampler.gr.size, len(sampler.parents) * len(sampler.sources))
            self.assertFalse(np.isin(sampler.routes, sampler.protected).any())
            self.assertEqual(sampler.report()["observed_broad_goals"], len(sampler.sources))


if __name__ == "__main__":
    unittest.main()
