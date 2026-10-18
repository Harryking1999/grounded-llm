from collections import deque
import unittest
import numpy as np
from experiments.puzzle8_qmap.src.environment import Puzzle8
from experiments.puzzle8_qmap.src.data_sampling import local_relations
from experiments.task_text_baselines.src import puzzle8 as rules


class LocalDataTest(unittest.TestCase):
    def test_exact_local_distances_complete_candidates_and_protected_goals(self):
        env = Puzzle8()
        protected = np.array([123, 4567])
        pairs, tasks, children, labels = local_relations(env, protected)
        self.assertFalse(np.isin(pairs[:, :2], protected).any())
        self.assertFalse(np.isin(tasks, protected).any())
        self.assertFalse(np.isin(children[children >= 0], protected).any())
        np.testing.assert_array_equal(children, env.next[tasks[:, 0]])
        rng = np.random.default_rng(18)
        for i in rng.choice(len(tasks), 30, replace=False):
            goal = tuple(map(int, env.positions[tasks[i, 1]]))
            truth, queue = {goal: 0}, deque([goal])
            while queue:
                s = queue.popleft()
                if truth[s] == 3:
                    continue
                for _, c in rules.successors(s):
                    if c not in truth:
                        truth[c] = truth[s] + 1
                        queue.append(c)
            self.assertEqual(truth[tuple(map(int, env.positions[tasks[i, 0]]))], 2)
            for c, distance in zip(children[i], labels[i]):
                if c >= 0:
                    self.assertEqual(truth[tuple(map(int, env.positions[c]))], distance)
                else:
                    self.assertEqual(distance, 32)
        for i in rng.choice(len(pairs), 30, replace=False):
            a, b, distance = pairs[i]
            self.assertEqual(distance == 1, b in env.next[a])
            if distance == 2:
                self.assertTrue(any(b in env.next[c] for c in env.next[a] if c >= 0))


if __name__ == "__main__":
    unittest.main()
