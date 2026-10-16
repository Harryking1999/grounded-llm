"""Coverage and leakage checks for the relational split, on an exact small graph."""
import unittest

import numpy as np
import torch

from experiments.cube2_qmap.src.full_state_joint import (
    FullStateSampler, RelationGuard, batch_losses, fixed_batch, sample_tasks, task_pairs)
from experiments.puzzle_qmap.src.model import QMap


class Ring:
    size, diameter, actions = 41, 20, ("+", "-")

    def compose(self, a, b):
        return (np.asarray(a) + np.asarray(b)) % self.size

    def successors(self, states):
        states = np.asarray(states)
        return np.stack(((states + 1) % self.size, (states - 1) % self.size), -1)

    def distance(self, a, b):
        delta = (np.asarray(a) - np.asarray(b)) % self.size
        return np.minimum(delta, self.size - delta)

    def features(self, states):
        return np.eye(self.size, dtype=np.float32)[states]


class FullStateTests(unittest.TestCase):
    def setUp(self):
        self.env = Ring()
        self.layers = [np.flatnonzero(self.env.distance(np.arange(self.env.size), 0) == d)
                       for d in range(self.env.diameter + 1)]
        self.config = {"data": {"seed": 5, "target_distance_levels": list(range(1, 21))},
                       "training": {"parent_batch_size": 8, "temperature": 1.0}}

    def test_unordered_guard_blocks_candidate_labels_and_rank_tasks(self):
        heldout = np.array([[3, 10]])
        guard = RelationGuard(self.env.size, task_pairs(self.env, heldout))
        self.assertTrue(guard.contains(np.array([[10, 3]]))[0])
        self.assertTrue(guard.task_blocked(self.env, np.array([[4, 10]]))[0])
        self.assertFalse(guard.contains(np.array([[0, 1]]))[0])
        sampler = FullStateSampler(self.env, self.config, guard, self.layers)
        for parents in sampler.batches():
            batch = sampler.draw(parents)
            self.assertFalse(guard.contains(batch["states"][batch["pairs"]]).any())
            np.testing.assert_array_equal(batch["labels"], self.env.distance(*batch["states"][batch["pairs"]].T))
            physical_children = batch["states"][batch["children"]]
            goals = batch["states"][batch["goals"]]
            np.testing.assert_array_equal(batch["truth"], self.env.distance(physical_children, goals[:, None]))

    def test_every_parent_and_action_edge_are_supervised_each_epoch(self):
        sampler = FullStateSampler(self.env, self.config, RelationGuard(self.env.size, np.empty((0, 2))), self.layers)
        for epoch in range(1, 3):
            visited, edges = [], []
            for parents in sampler.batches():
                batch = sampler.draw(parents)
                visited.extend(parents)
                edge_count = len(parents) * len(self.env.actions)
                physical = batch["states"][batch["pairs"][:edge_count]]
                np.testing.assert_array_equal(physical[:, 0], np.repeat(parents, len(self.env.actions)))
                np.testing.assert_array_equal(physical[:, 1], self.env.successors(parents).ravel())
                np.testing.assert_array_equal(batch["labels"][:edge_count], np.ones(edge_count))
                edges.extend(map(tuple, physical))
            np.testing.assert_array_equal(np.sort(visited), np.arange(self.env.size))
            self.assertEqual(len(set(edges)), self.env.size * len(self.env.actions))
            self.assertEqual(sampler.coverage()["parent_exposures"], epoch * self.env.size)
            self.assertEqual(sampler.coverage()["directed_edges_covered"], self.env.size * len(self.env.actions))

    def test_stratified_holdout_excludes_existing_candidate_relations(self):
        guard = RelationGuard(self.env.size, task_pairs(self.env, np.array([[3, 10], [20, 30]])))
        tasks = sample_tasks(self.env, [4, 5, 6], 5, np.random.default_rng(7), self.layers, guard)
        self.assertFalse(guard.task_blocked(self.env, tasks).any())
        np.testing.assert_array_equal(self.env.distance(*tasks.T), np.repeat([4, 5, 6], 5))

    def test_fixed_replay_calibrates_all_candidates_and_backpropagates(self):
        tasks = np.array([[3, 10], [20, 30]])
        batch = fixed_batch(self.env, tasks)
        model = QMap(self.env.size, dimension=8, hidden=16, cap=self.env.diameter, metric="linf")
        h, r = batch_losses(model, torch.as_tensor(self.env.features(batch["states"])), batch, self.config)
        (h + r).backward()
        self.assertEqual(len(batch["pairs"]), len(tasks) * (1 + len(self.env.actions)))
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters()))
        self.assertGreater(sum(float(p.grad.abs().sum()) for p in model.parameters()), 0)


if __name__ == "__main__":
    unittest.main()
