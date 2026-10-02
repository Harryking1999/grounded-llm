"""The value calibration is one frozen scalar per task and condition."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from experiments.flamingo_map_reader.src.prepare_kv_scale_followup import build_followup


class ScaleFollowupTest(unittest.TestCase):
    def test_one_scale_shared_by_trajectory_and_counterfactual_conditions(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            data, measures = root / "data", root / "measure"
            data.mkdir()
            measures.mkdir()
            base = {"task": "graph", "study": "small_pilot",
                    "map": {"memory_mode": "address_key_state_value"},
                    "training": {"epochs": 16},
                    "checkpoint": {"every_epoch_fraction": 0.1}}
            records = [{"split": "train", "graph_id": "graph_00"}]
            (data / "graph_addressed_kv.config.json").write_text(json.dumps(base))
            (data / "graph_addressed_kv.manifest.json").write_text(
                json.dumps({"config": base, "records": records}))
            (data / "graph_joint.manifest.json").write_text(
                json.dumps({"config": {}, "records": records}))
            for mode, magnitude in (("joint", 0.4),
                                    ("address_key_state_value", 0.1)):
                name = "joint" if mode == "joint" else "addressed_kv"
                (measures / f"graph_{name}.json").write_text(json.dumps({
                    "task": "graph", "memory_mode": mode, "train_cases": 16,
                    "value_scale": 1.0, "median_ungated": magnitude}))
            contract = {"train_cases_per_task": 16, "scale_rule": "test",
                        "max_steps_per_condition": 256,
                        "checkpoint_every_epoch_fraction": 1.0,
                        "conditions": ["calibrated_trajectory",
                                       "calibrated_counterfactual_first_turn",
                                       "calibrated_pairwise_counterfactual"]}
            scale, prepared = build_followup("graph", data, measures, contract)
            self.assertEqual(scale, 4.0)
            self.assertEqual(len(prepared), 3)
            for _, config, manifest in prepared:
                self.assertEqual(config["map"]["value_scale"], 4.0)
                self.assertEqual(config["training"]["max_steps"], 256)
                self.assertEqual(config, manifest["config"])
            self.assertEqual(prepared[0][1]["training"]["supervision_mode"], "trajectory")
            self.assertEqual(prepared[1][1]["training"]["supervision_mode"],
                             "counterfactual_first_turn")
            self.assertEqual(prepared[2][1]["training"]["readout_style"], "pairwise")


if __name__ == "__main__":
    unittest.main()
