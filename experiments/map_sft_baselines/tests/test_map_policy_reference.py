import random
import unittest
from experiments.map_sft_baselines.src.map_policy_reference import choose_candidate


class SamplingTests(unittest.TestCase):
    def test_top_three_uniform_does_not_include_fourth(self):
        counts={i:0 for i in range(1,5)};rng=random.Random(41)
        for _ in range(6000):counts[choose_candidate([4.,1.,3.,2.],3,rng)]+=1
        self.assertEqual(counts[1],0)
        for i in (2,3,4):self.assertTrue(1800<counts[i]<2200)

    def test_ties_and_small_candidate_sets(self):
        self.assertEqual(choose_candidate([1.,1.,2.],1,random.Random(1)),1)
        self.assertEqual(choose_candidate([8.],3,random.Random(1)),1)
        with self.assertRaises(ValueError):choose_candidate([],3,random.Random(1))


if __name__=='__main__':unittest.main()
