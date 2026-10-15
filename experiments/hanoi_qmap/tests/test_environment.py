import tempfile
from pathlib import Path
import unittest

import numpy as np

from experiments.hanoi_qmap.src.environment import Hanoi
from experiments.puzzle_qmap.src.data import build, pair_keys


class HanoiTest(unittest.TestCase):
    def test_exhaustive_graph_and_solutions(self):
        env = Hanoi()
        self.assertEqual(int(env.distances.max()), 7)
        self.assertEqual(int(env.distance(0, 26)), 7)
        self.assertEqual(int((env.next >= 0).sum()), 78)
        for source in range(27):
            for goal in range(27):
                state = source
                actions = env.solve(source, goal)
                self.assertEqual(len(actions), env.distance(source, goal))
                for action in actions:
                    state = env.step(state, action)
                self.assertEqual(state, goal)
            for action, child in enumerate(env.next[source]):
                if child < 0:
                    with self.assertRaises(ValueError):
                        env.step(source, env.actions[action])
                else:
                    self.assertIn(source, env.next[child])

    def test_invalid_input_and_encoding(self):
        env = Hanoi()
        for invalid in (-1, 27, 1.5, True):
            with self.assertRaises(ValueError):
                env.step(invalid, "A>B")
        for state in range(27):
            self.assertEqual(env.state_id(env.positions[state]), state)
        with self.assertRaises(ValueError):
            env.step(0, "B>A")
        with self.assertRaises(ValueError):
            env.state_id([0, 1, 3])

    def test_labels_from_all_losses_do_not_cross_pair_split(self):
        env = Hanoi()
        with tempfile.TemporaryDirectory() as tmp:
            data, _ = build(env, {"seed": 20261015, "data": {}}, Path(tmp))
        train = set(pair_keys(data["pairs_train"], env.size))
        for split in ("validation", "test"):
            self.assertFalse(train & set(pair_keys(data[f"pairs_{split}"], env.size)))
        for (parent, goal), children in zip(data["tasks_train"], data["children_train"]):
            for child in children[children >= 0]:
                self.assertIn(min(child, goal) * 27 + max(child, goal), train)
        for src, _, dst in data["transitions_train"]:
            self.assertIn(min(src, dst) * 27 + max(src, dst), train)


if __name__ == "__main__":
    unittest.main()
