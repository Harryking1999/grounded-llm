import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from experiments.hanoi_qmap.src.data import build, pair_keys
from experiments.hanoi_qmap.src.environment import Hanoi
from experiments.hanoi_qmap.src.run import train
from experiments.hanoi_qmap.src.tower_data import TowerData


class TowerCoverageTest(unittest.TestCase):
    def test_promotions_preserve_split_history_and_close_all_tower_branches(self):
        env = Hanoi(3)
        base = json.loads((Path(__file__).parents[1] / "configs/hanoi8.json").read_text())
        base["task"] = "hanoi3"
        base["rules"]["disks"] = 3
        config = copy.deepcopy(base)
        config["data"]["require_all_tower_relations"] = True
        config["model"] = dict(dimension=8, hidden=16)
        config["training"].update(steps=1, batch_size=16, decision_batch_size=8, eval_every=1,
                                  cpu_threads=1, min_steps=0, retain_initial=True)
        config["sampling"].update(tower_fraction=.4, minimum_broad_passes=0.)
        config["evaluation"].update(decision_count=16, rollout_count=16, full_tower_policy=True)
        config["selection"] = dict(kind="tower_restoration_with_relation_guard", maximum_validation_mae_increase=.25)
        with tempfile.TemporaryDirectory() as temp:
            old, out = Path(temp) / "old", Path(temp) / "new"
            old.mkdir(); out.mkdir()
            prior, _ = build(env, base, old)
            data, summary = build(env, config, out)
            train_keys = pair_keys(data["pairs_train"], env.size)
            self.assertFalse(np.setdiff1d(pair_keys(prior["pairs_train"], env.size), train_keys).size)
            for split in ("validation", "test"):
                self.assertFalse(np.setdiff1d(pair_keys(data[f"pairs_{split}"], env.size),
                                               pair_keys(prior[f"pairs_{split}"], env.size)).size)
                self.assertFalse(np.intersect1d(pair_keys(data[f"pairs_{split}"], env.size), train_keys).size)
            self.assertEqual(set(summary["tower_decision_coverage"].values()), {env.size - 1})
            sampler = TowerData(env, data, np.random.default_rng(18))
            supported = set(map(tuple, data["tasks_train"][sampler.tasks]))
            for source, goal in supported:
                children = env.next[source]
                for child in children[children >= 0]:
                    if child != goal:
                        self.assertIn((child, goal), supported)
                for transform in range(6):
                    self.assertIn((int(env.symmetry(source, transform)), int(env.symmetry(goal, transform))), supported)
            with contextlib.redirect_stdout(io.StringIO()):
                result = train(env, config, data, out, "cpu", "test")
            report = result["final"]["all_state_towers"]
            self.assertEqual(report["nonterminal_tasks"], 3 * (env.size - 1))
            actual = sum(sum(group["tower_starts_solved"].values()) for group in report["by_goal"].values())
            self.assertEqual(actual, result["final"]["standard_towers"]["q"]["solved"])


if __name__ == "__main__":
    unittest.main()
