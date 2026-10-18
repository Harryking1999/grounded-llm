import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from experiments.puzzle8_qmap.src.environment import Puzzle8
from experiments.puzzle8_qmap.src.data import build
from experiments.puzzle8_qmap.src.branch_data import BranchData
from experiments.task_text_baselines.src import puzzle8 as rules


class BranchDataTest(unittest.TestCase):
    def test_exact_forks_downstream_closure_and_heldout_relations(self):
        env = Puzzle8()
        config = json.loads((Path(__file__).parents[1] / "configs/puzzle8.json").read_text())
        config["data"].update(intermediate_goals=[1, 1, 1], eval_per_depth=2)
        options = dict(new_goals=3, seed=18, fork_roots=124, hard_roots=16,
                       fractions=[.3, .3, .4], route_fraction=.1, edge_fraction=.1)
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp)
            data, _ = build(env, config, out)
            sampler = BranchData(env, data, options, np.random.default_rng(19), out)
            protected = sampler.protected
            self.assertFalse(np.isin(sampler.goals, protected).any())
            self.assertFalse(np.isin(sampler.edges[:, :2], protected).any())
            expected_edges = (env.next >= 0) & ~np.isin(np.arange(env.size), protected)[:, None]
            expected_edges &= ~np.isin(env.next, protected)
            self.assertEqual(len(sampler.edges), int(expected_edges.sum()))
            supported = set(map(tuple, sampler.routes))
            for goal, roots in zip(sampler.route_goals, sampler.roots):
                for root in roots:
                    for state in np.r_[root, env.next[root]]:
                        if state >= 0 and state != goal and sampler.allowed[state]:
                            self.assertIn((state, goal), supported)
            for state, goal in sampler.routes:
                children = env.next[state]
                truth = env.distance(np.maximum(children, 0), goal)
                good = children[(children >= 0) & (truth == env.distance(state, goal) - 1)]
                for child in good:
                    if child != goal and sampler.allowed[child]:
                        self.assertIn((child, goal), supported)
            # Independent rule BFS, including cross-branch candidate labels.
            for goal in sampler.goals:
                reference = rules.distances(tuple(map(int, env.positions[goal])))
                sources = np.arange(0, env.size, 997)
                np.testing.assert_array_equal(env.distance(sources, goal),
                    [reference[tuple(map(int, env.positions[s]))] for s in sources])
            sampler.refresh(np.arange(1, 200))
            pairs = sampler.pairs(4096)
            tasks, children, labels = sampler.tasks(4096)
            np.testing.assert_array_equal(children, env.next[tasks[:, 0]])
            known = np.isin(pairs[:, 1], list(env.goal_columns))
            np.testing.assert_array_equal(pairs[known, 2], env.distance(pairs[known, 0], pairs[known, 1]))
            self.assertTrue((pairs[~known, 2] == 1).all())
            self.assertTrue((env.next[pairs[~known, 0]] == pairs[~known, 1, None]).any(-1).all())
            np.testing.assert_array_equal(labels[children >= 0],
                env.distance(np.maximum(children, 0), tasks[:, 1, None])[children >= 0])
            # Protect every heldout parent/candidate relation in either direction.
            key = lambda x: np.minimum(x[:, 0], x[:, 1]).astype(np.int64) * env.size + np.maximum(x[:, 0], x[:, 1])
            train = np.r_[key(pairs[:, :2]), key(np.column_stack((children[children >= 0],
                                  np.broadcast_to(tasks[:, 1, None], children.shape)[children >= 0])))]
            for split in ("validation", "test"):
                test = data[f"tasks_{split}"]
                candidates = data[f"children_{split}"]
                heldout = np.r_[key(data[f"pairs_{split}"][:, :2]),
                    key(np.column_stack((candidates[candidates >= 0],
                        np.broadcast_to(test[:, 1, None], candidates.shape)[candidates >= 0])))]
                self.assertFalse(np.intersect1d(train, heldout).size)


if __name__ == "__main__":
    unittest.main()
