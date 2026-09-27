"""Check that the frozen suite is partitioned without duplicate trial seeds."""

import json
from pathlib import Path
import unittest

from experiments.external_map_interface.src.evaluate_path256_continuous_batch import (
    shard_cases, trial_seed,
)


CONFIG = Path(__file__).resolve().parents[1] / "configs" / "path256_continuous_five_graphs.json"


class BatchIdentityTest(unittest.TestCase):
    def test_every_case_in_exactly_one_shard(self):
        config = json.loads(CONFIG.read_text(encoding="utf-8"))
        shards = [shard_cases(config, index, 2) for index in range(2)]
        self.assertEqual(sorted(shards[0] + shards[1]), config["case_indices"])
        self.assertFalse(set(shards[0]) & set(shards[1]))

    def test_every_frozen_sample_has_a_unique_seed(self):
        config = json.loads(CONFIG.read_text(encoding="utf-8"))
        seeds = [trial_seed(config, graph, case, replicate)
                 for graph in range(config["graph_count"])
                 for case in config["case_indices"]
                 for replicate in range(config["replicates"])]
        self.assertEqual(len(seeds), 1920)
        self.assertEqual(len(set(seeds)), len(seeds))
        self.assertEqual(seeds[0], config["seed"])


if __name__ == "__main__":
    unittest.main()
