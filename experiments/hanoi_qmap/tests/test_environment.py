import tempfile
import json
from pathlib import Path
import unittest

import numpy as np

from experiments.hanoi_qmap.src.environment import Hanoi
from experiments.puzzle_qmap.src.data import build, pair_keys, hanoi_orbit_keys


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
        config = json.loads((Path(__file__).parents[1] / "configs/study.json").read_text())
        with tempfile.TemporaryDirectory() as tmp:
            data, summary = build(env, config, Path(tmp))
        train = set(pair_keys(data["pairs_train"], env.size))
        for split in ("validation", "test"):
            self.assertFalse(train & set(pair_keys(data[f"pairs_{split}"], env.size)))
            self.assertFalse(set(hanoi_orbit_keys(env, data["pairs_train"])) &
                             set(hanoi_orbit_keys(env, data[f"pairs_{split}"])))
        for (parent, goal), children in zip(data["tasks_train"], data["children_train"]):
            for child in children[children >= 0]:
                self.assertIn(min(child, goal) * 27 + max(child, goal), train)
        self.assertFalse(any(key.startswith("transitions") for key in data))
        self.assertGreater(summary["augmentation"]["pairs"]["changed_rows"], 0)
        self.assertGreater(len(data["pairs_train"]), len(data["pairs_train_base"]))

    def test_six_peg_symmetries_preserve_every_edge_and_distance(self):
        env = Hanoi()
        for transform in range(6):
            renamed = env.symmetry(np.arange(27), transform)
            self.assertEqual(len(np.unique(renamed)), 27)
            np.testing.assert_array_equal(env.distances, env.distances[renamed[:, None], renamed[None, :]])
            for state in range(27):
                for action, child in enumerate(env.next[state]):
                    renamed_action = env.symmetry_actions(action, transform)
                    expected = -1 if child < 0 else env.symmetry(child, transform)
                    self.assertEqual(env.next[renamed[state], renamed_action], expected)

    def test_four_five_six_disks_exact_graph_and_long_solutions(self):
        rng = np.random.default_rng(16)
        for disks in (4, 5, 6):
            env = Hanoi(disks)
            self.assertEqual(env.size, 3**disks)
            self.assertEqual(int(env.distances.max()), 2**disks - 1)
            self.assertEqual(len(env.solve(0)), 2**disks - 1)
            self.assertEqual(env.features([0]).shape, (1, 3 * disks))
            self.assertTrue((env.distances >= 0).all())
            for source, goal in rng.integers(env.size, size=(30, 2)):
                current = int(source)
                actions = env.solve(current, int(goal))
                self.assertEqual(len(actions), env.distance(source, goal))
                for action in actions:
                    current = env.step(current, action)
                self.assertEqual(current, goal)
            states = np.arange(env.size)
            renamed = env.symmetry(states, 4)
            np.testing.assert_array_equal(env.distances, env.distances[renamed[:, None], renamed[None, :]])


if __name__ == "__main__":
    unittest.main()
