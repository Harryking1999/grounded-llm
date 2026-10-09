import math
import unittest
from itertools import permutations

from experiments.flamingo_map_reader.src.analyze_blocks_ranking import ndcg


class MinmaxNdcgTest(unittest.TestCase):
    def test_relative_distances_and_exact_random_expectation(self):
        for cutoff in (1, 10):
            scores = []
            for order in permutations((1, 2, 3)):
                ranks = {candidate: i for i, candidate in enumerate(order)}
                value, chance = ndcg([8, 9, 10], ranks, cutoff, 'minmax_distance')
                shifted = ndcg([1, 2, 3], ranks, cutoff, 'minmax_distance')[0]
                scaled = ndcg([16, 18, 20], ranks, cutoff, 'minmax_distance')[0]
                self.assertAlmostEqual(value, shifted)
                self.assertAlmostEqual(value, scaled)
                scores.append(value)
            self.assertAlmostEqual(sum(scores) / len(scores), chance)
        reverse = {3: 0, 2: 1, 1: 2}
        self.assertEqual(ndcg([8, 9, 10], reverse, 1, 'minmax_distance')[0], 0)
        ideal = 1 + .5 / math.log2(3)
        self.assertAlmostEqual(ndcg([8, 9, 10], reverse, 10, 'minmax_distance')[0],
                               (.5 / math.log2(3) + .5) / ideal)

    def test_ties_single_candidate_and_invalid_answers(self):
        self.assertEqual(ndcg([], None, gain='minmax_distance'), (None, None))
        self.assertEqual(ndcg([8], {1: 0}, gain='minmax_distance'), (1, 1))
        self.assertEqual(ndcg([8, 8], {1: 0, 2: 1}, gain='minmax_distance'), (1, 1))
        self.assertEqual(ndcg([8, 9], None, gain='minmax_distance')[0], 0)
        self.assertAlmostEqual(ndcg([8, 9, 10], {1: 0, 3: 0}, 1,
                                   'minmax_distance')[0], .5)


if __name__ == '__main__':
    unittest.main()
