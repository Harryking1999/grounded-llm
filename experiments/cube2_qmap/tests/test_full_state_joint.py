"""Coverage and leakage checks for the relational split, on an exact small graph."""
import unittest
import copy

import numpy as np
import torch

from experiments.cube2_qmap.src.full_state_joint import (
    FullStateSampler, RelationGuard, batch_losses, fixed_batch, restore_sampler, sample_tasks, task_pairs)
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

    def test_new_and_legacy_resume_preserve_next_epoch_samples(self):
        guard = RelationGuard(self.env.size, np.empty((0, 2)))
        original = FullStateSampler(self.env, self.config, guard, self.layers)
        for _ in range(2):
            for parents in original.batches():
                original.draw(parents)
        saved = {"epoch": 2, "step": 12, "coverage": original.coverage()}
        for legacy in (False, True):
            checkpoint = copy.deepcopy(saved)
            if not legacy:
                checkpoint["sampler"] = original.state_dict()
            restored = FullStateSampler(self.env, self.config, guard, self.layers)
            self.assertEqual(restore_sampler(restored, checkpoint), (2, 12))
            expected = FullStateSampler(self.env, self.config, guard, self.layers)
            expected.load_state_dict(original.state_dict())
            for a, b in zip(expected.batches(), restored.batches()):
                np.testing.assert_array_equal(a, b)
                aa, bb = expected.draw(a), restored.draw(b)
                for key in aa:
                    np.testing.assert_array_equal(aa[key], bb[key])

    def test_mid_epoch_checkpoint_is_rejected(self):
        sampler = FullStateSampler(self.env, self.config, RelationGuard(self.env.size, np.empty((0, 2))), self.layers)
        sampler.draw(np.arange(8))
        with self.assertRaisesRegex(ValueError, "completed epoch"):
            restore_sampler(sampler, {"epoch": 1, "step": 1, "coverage": sampler.coverage()})


if __name__ == "__main__":
    unittest.main()
