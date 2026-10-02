"""Freeze taskwise value scales and paired trajectory/readout pilot contracts."""

from argparse import ArgumentParser
from copy import deepcopy
import json
import math
from pathlib import Path


def build_followup(task, data_root, measure_root, contract):
    joint = json.loads((measure_root / f"{task}_joint.json").read_text(encoding="utf-8"))
    addressed = json.loads((measure_root / f"{task}_addressed_kv.json").read_text(encoding="utf-8"))
    for mode, report in (("joint", joint), ("address_key_state_value", addressed)):
        if (report["task"] != task or report["memory_mode"] != mode or
                report["train_cases"] != contract["train_cases_per_task"] or
                report["value_scale"] != 1.0):
            raise ValueError(f"Unexpected {task} {mode} calibration source")
    reference = joint["median_ungated"]
    candidate = addressed["median_ungated"]
    scale = reference / candidate
    if not all(math.isfinite(value) and value > 0 for value in (reference, candidate, scale)):
        raise ValueError("Map branch calibration needs finite positive measurements")
    base_config = json.loads((data_root / f"{task}_addressed_kv.config.json").read_text())
    base_manifest = json.loads((data_root / f"{task}_addressed_kv.manifest.json").read_text())
    joint_manifest = json.loads((data_root / f"{task}_joint.manifest.json").read_text())
    if (base_config != base_manifest["config"] or
            base_manifest["records"] != joint_manifest["records"]):
        raise ValueError("Paired pilot records differ")
    prepared = []
    for condition in contract["conditions"]:
        config = deepcopy(base_config)
        config["study"] += "_" + condition
        config["map"]["value_scale"] = scale
        config["training"]["max_steps"] = contract["max_steps_per_condition"]
        config["checkpoint"]["every_epoch_fraction"] = contract["checkpoint_every_epoch_fraction"]
        config["training"]["supervision_mode"] = (
            "counterfactual_first_turn" if condition == "calibrated_counterfactual_first_turn"
            else "trajectory")
        config["scale_calibration"] = {
            "rule": contract["scale_rule"],
            "joint_median_ungated": reference,
            "addressed_median_ungated": candidate,
            "train_cases": addressed["train_cases"],
        }
        manifest = dict(base_manifest, config=config)
        prepared.append((condition, config, manifest))
    return scale, prepared


def main():
    parser = ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--pilot-data-root", type=Path, required=True)
    parser.add_argument("--measure-root", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    args = parser.parse_args()
    if args.out_root.exists():
        raise FileExistsError(args.out_root)
    contract = json.loads(args.contract.read_text(encoding="utf-8"))
    prepared = []
    scales = {}
    for task in ("graph", "blocks"):
        scales[task], conditions = build_followup(task, args.pilot_data_root,
                                                   args.measure_root, contract)
        prepared.extend((task, *entry) for entry in conditions)
    args.out_root.mkdir(parents=True)
    for task, condition, config, manifest in prepared:
        prefix = args.out_root / f"{task}_{condition}"
        prefix.with_suffix(".config.json").write_text(
            json.dumps(config, indent=2) + "\n", encoding="utf-8")
        prefix.with_suffix(".manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"taskwise_value_scales": scales,
                      "conditions": len(prepared)}))


if __name__ == "__main__":
    main()
