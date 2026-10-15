import unittest
import numpy as np

from experiments.puzzle_qmap.src.sampling import StratifiedEpochSampler


class StratifiedEpochTest(unittest.TestCase):
    def test_equal_quotas_and_full_epochs_in_each_layer(self):
        labels = np.repeat([1, 3, 11], 7)
        sampler = StratifiedEpochSampler(labels, np.random.default_rng(5))
        first = np.concatenate([sampler.draw(3) for _ in range(7)])
        self.assertEqual(set(first), set(range(21)))
        self.assertEqual(sampler.passes, 1)
        second = np.concatenate([sampler.draw(3) for _ in range(7)])
        self.assertEqual(set(second), set(range(21)))
        self.assertEqual(sampler.passes, 2)

    def test_remainder_rotates_and_small_layers_do_not_starve(self):
        labels = np.repeat([1, 2, 3], [1, 4, 40])
        sampler = StratifiedEpochSampler(labels, np.random.default_rng(6))
        all_draws = []
        for _ in range(24):
            ids = sampler.draw(5)
            counts = np.bincount(labels[ids], minlength=4)[1:]
            self.assertLessEqual(np.ptp(counts), 1)
            all_draws.extend(ids)
        np.testing.assert_array_equal(np.bincount(labels[all_draws])[1:], [40, 40, 40])
        self.assertEqual(set(all_draws), set(range(len(labels))))
        self.assertEqual(sampler.passes, 1)
        self.assertEqual(sampler.coverage()["1"]["passes"], 40)
        self.assertEqual(sampler.steps_for_passes(5, 2), 48)

    def test_batch_smaller_than_number_of_layers(self):
        labels = np.arange(1, 12)
        sampler = StratifiedEpochSampler(labels, np.random.default_rng(7))
        ids = np.concatenate([sampler.draw(2) for _ in range(11)])
        np.testing.assert_array_equal(np.bincount(ids), np.full(11, 2))


if __name__ == "__main__":
    unittest.main()
