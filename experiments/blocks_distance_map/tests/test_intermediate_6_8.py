import unittest

from experiments.blocks_distance_map.src.blocks12_eval import construct
from experiments.blocks_distance_map.src.intermediate_6_8 import make_cases
from experiments.blocks_distance_map.src.oracle import DistanceOracle


class IntermediateBandTest(unittest.TestCase):
    def test_fixed_cases_are_exact_and_exclude_easy_tasks(self):
        roots = construct(3, 12, 41)
        inputs = [{"board": i, "source": r["initial"]} for i, r in enumerate(roots)]
        first = make_cases(inputs, 19)
        self.assertEqual(first, make_cases(inputs, 19))
        self.assertEqual([r["true_distance"] for r in first], [6, 7, 8])
        oracle = DistanceOracle()
        for row in first:
            self.assertTrue(int(row["goal"]))
            self.assertEqual(oracle.distance(int(row["source"]), int(row["goal"])), row["true_distance"])
            self.assertEqual(row["source"], inputs[row["board"]]["source"])


if __name__ == "__main__":
    unittest.main()
