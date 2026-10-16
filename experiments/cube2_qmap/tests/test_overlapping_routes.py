"""Physical overlap and equal exposure are essential to the causal comparison."""
import unittest

import numpy as np

from experiments.cube2_qmap.src.overlapping_routes import make_schedule, novel_relations, pool_rows
from experiments.puzzle_qmap.src.data import pair_keys


class Cyclic:
    size = 101

    def compose(self, left, right):
        return (np.asarray(left) + np.asarray(right)) % self.size

    def relative(self, source, goal):
        return (np.asarray(goal) - np.asarray(source)) % self.size

    def distance(self, source, goal):
        delta = self.relative(source, goal)
        return np.minimum(delta, self.size - delta)

    def successors(self, ids):
        ids = np.asarray(ids)
        return np.stack(((ids - 1) % self.size, (ids + 1) % self.size), axis=-1)


class OverlapTests(unittest.TestCase):
    def test_shared_states_are_physical_and_pools_form_a_connected_chain(self):
        env = Cyclic()
        base = {"states": np.arange(5), "poses_fit": np.arange(20, 25)[None],
                "tasks": np.array([[1, 0], [2, 4], [3, 0]])}
        config = {"data": {"seed": 17, "collections": 3, "minimum_shared_states": 3,
                          "minimum_new_states": 1, "minimum_new_goals": 1, "maximum_pose_attempts": 256}}
        forbidden = np.zeros(env.size, dtype=bool)
        forbidden[70:80] = True
        rows = pool_rows(env, base, config, forbidden, "overlap")
        np.testing.assert_array_equal(rows[0], base["poses_fit"][0])
        for i in range(1, len(rows)):
            self.assertFalse(forbidden[rows[i]].any())
            self.assertGreaterEqual(max(len(np.intersect1d(rows[i], row)) for row in rows[:i]), 3)
            self.assertGreaterEqual((~np.isin(rows[i], rows[:i])).sum(), 1)
        disjoint = pool_rows(env, base, config, forbidden, "disjoint")
        self.assertEqual(len(np.unique(disjoint)), disjoint.size)
        self.assertFalse(forbidden[disjoint].any())

    def test_joint_and_sequential_have_exactly_equal_collection_exposure(self):
        for mode in ("joint", "sequential"):
            schedule = make_schedule(8, 17, mode, 23)
            np.testing.assert_array_equal(np.bincount(schedule), np.full(8, 17))
        joint = make_schedule(8, 17, "joint", 23).reshape(-1, 8)
        for round_ in joint:
            np.testing.assert_array_equal(np.sort(round_), np.arange(8))

    def test_novel_relations_use_seen_inputs_but_no_training_pair_label(self):
        env = Cyclic()
        rows = np.array([np.arange(5), np.arange(10, 15)])
        base = {"pairs": np.column_stack(np.triu_indices(5, 1)),
                "tasks": np.array([[1, 0], [2, 0], [3, 0], [1, 4], [2, 4], [3, 4]])}
        config = {"data": {"seed": 17}, "evaluation": {"cross_distance_levels": [7, 8, 9],
                  "cross_pairs_per_level": 2, "cross_decisions_per_level": 2}}
        result = novel_relations(env, base, rows, config)
        train = np.concatenate([pair_keys(row[base["pairs"]], env.size) for row in rows])
        physical = result["states"][result["pairs"]]
        self.assertFalse(np.isin(pair_keys(physical, env.size), train).any())
        tasks = result["states"][result["tasks"]]
        children = result["states"][result["children"]]
        np.testing.assert_array_equal(children, env.successors(tasks[:, 0]))
        for col in range(children.shape[1]):
            keys = pair_keys(np.column_stack((children[:, col], tasks[:, 1])), env.size)
            self.assertFalse(np.isin(keys, train).any())


if __name__ == "__main__":
    unittest.main()
