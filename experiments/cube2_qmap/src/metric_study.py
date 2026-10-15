"""Matched pure-Q norm comparison, with validation-only Linf -> L2 fallback."""
import argparse
import copy
import json
from pathlib import Path
import subprocess
import sys


def validation_gate(control, candidate, contract):
    reference = control["final"]["validation"]["rollout"]["q"]
    result = candidate["final"]["validation"]["rollout"]["q"]
    long_rows = [v for k, v in result["by_start_distance"].items()
                 if int(k) >= contract["long_distance_minimum"]]
    count = sum(v["count"] for v in long_rows)
    long_rate = sum(v["count"] * v["success_rate"] for v in long_rows) / count if count else 0.
    passed = (result["success_rate"] >= contract["minimum_success_rate"] and
              long_rate >= contract["minimum_long_success_rate"] and
              (not contract["require_not_worse_than_l1"] or
               result["success_rate"] >= reference["success_rate"]))
    return {"passed": passed, "validation_success_rate": result["success_rate"],
            "validation_long_count": count, "validation_long_success_rate": long_rate,
            "l1_validation_success_rate": reference["success_rate"], "test_used": False}


def summarize(root, names, device, source_commit):
    import numpy as np
    import torch
    from experiments.puzzle_qmap.src.evaluate import evaluate
    from experiments.puzzle_qmap.src.model import MapReader
    from experiments.puzzle_qmap.src.tasks import make_environment

    torch.set_num_threads(4)
    result = {"source_commit": source_commit, "conditions": {}}
    for name in names:
        run = root / name
        summary = json.loads((run / "summary.json").read_text())
        config = json.loads((run / "config.json").read_text())
        data = dict(np.load(run / "data.npz", allow_pickle=False))
        env = make_environment(config["task"], run / "oracle_depth.npy")
        last = evaluate(MapReader(env, run / "last.pt", device).model, env, data, config, device)
        rows = [json.loads(line) for line in (run / "progress.jsonl").read_text().splitlines()]
        final = rows[-1]
        result["conditions"][name] = {
            "best_step": summary["best_step"], "last_step": summary["last_step"],
            "seconds": summary["training_seconds"], "stop_reason": summary["stop_reason"],
            "initial_training": summary["initial_training_monitor"],
            "last_training": final["training_monitor"],
            "initial_validation_objective": summary["initial_validation_objective"],
            "last_validation_objective": final["validation_objective"],
            "best": summary["final"], "last": last,
            "best_coverage": summary["best_model_training_coverage"],
            "last_coverage": summary["last_model_training_coverage"]}
    (root / "study_summary.json").write_text(json.dumps(result, indent=2) + "\n")


def plot_losses(root, names):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(10, 6), constrained_layout=True)
    for name in names:
        run = root / name
        rows = [json.loads(line) for line in (run / "progress.jsonl").read_text().splitlines()]
        summary = json.loads((run / "summary.json").read_text())
        initial = summary["initial_training_monitor"]
        x = [0] + [r["step"] / 1000 for r in rows]
        series = [
            [initial["pairs"]["huber"]] + [r["training_monitor"]["pairs"]["huber"] for r in rows],
            [summary["initial_validation_pairs"]["huber"]] + [r["validation_huber"] for r in rows],
            [initial["soft_rank_loss"]] + [r["training_monitor"]["soft_rank_loss"] for r in rows],
            [summary["initial_validation_objective"] - summary["initial_validation_pairs"]["huber"]] +
            [r["validation_soft_rank_loss"] for r in rows]]
        for ax, values in zip(axes.flat, series):
            ax.plot(x, values, label=name)
    for ax, title in zip(axes.flat, ("Fixed train: Huber", "Validation: Huber",
                                     "Fixed train: full soft CE", "Validation: full soft CE")):
        ax.set(title=title, xlabel="Updates (thousands)", ylabel="Loss")
        ax.grid(alpha=.2)
        ax.legend()
    fig.savefig(root / "losses.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--prepared-run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--devices", nargs=2, default=["cuda:0", "cuda:1"])
    parser.add_argument("--plot", action="store_true", help="Optional local plot; needs matplotlib")
    args = parser.parse_args()
    contract = json.loads(args.contract.read_text())
    prepared = args.prepared_run.resolve()
    parent = json.loads((prepared / "summary.json").read_text())
    if parent["source_commit"] != contract["prepared_source_commit"]:
        raise ValueError("Wrong prepared run")
    base = json.loads((prepared / "config.json").read_text())
    root = args.out.resolve()
    root.mkdir(parents=True, exist_ok=False)
    (root / "contract.json").write_text(json.dumps(contract, indent=2) + "\n")
    runtime = {"source_commit": args.source_commit, "prepared_run": str(prepared),
               "initialization": "identical fresh random weights; no checkpoint continuation", "jobs": {}}
    logs = []

    def launch(name, device):
        config = copy.deepcopy(base)
        config["seed"] = contract["seed"]
        config["model"] = {**contract["model"], "metric": name}
        config["training"].update(contract["training"])
        config["sampling"].update(contract["sampling"])
        path = root / (name + ".config.json")
        path.write_text(json.dumps(config, indent=2) + "\n")
        log = (root / (name + ".log")).open("w")
        logs.append(log)
        command = [sys.executable, "-u", "-m", "experiments.puzzle_qmap.src.run",
                   "--config", str(path), "--prepared-run", str(prepared), "--out", str(root / name),
                   "--source-commit", args.source_commit, "--device", device]
        child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        runtime["jobs"][name] = {"pid": child.pid, "device": device}
        (root / "runtime.json").write_text(json.dumps(runtime, indent=2) + "\n")
        return child

    children = {name: launch(name, device) for name, device in zip(contract["initial_conditions"], args.devices)}
    codes = {name: child.wait() for name, child in children.items()}
    (root / "completion.json").write_text(json.dumps(codes, indent=2) + "\n")
    if any(codes.values()):
        raise RuntimeError(f"Training process failed: {codes}")
    summaries = {name: json.loads((root / name / "summary.json").read_text()) for name in children}
    gate = validation_gate(summaries["l1"], summaries["linf"], contract["validation_gate"])
    (root / "fallback_decision.json").write_text(json.dumps(gate, indent=2) + "\n")
    names = list(children)
    if not gate["passed"]:
        fallback = contract["fallback"]
        names.append(fallback)
        codes[fallback] = launch(fallback, args.devices[1]).wait()
        (root / "completion.json").write_text(json.dumps(codes, indent=2) + "\n")
        if codes[fallback]:
            raise RuntimeError("Fallback training process failed")
    for log in logs:
        log.close()
    summarize(root, names, args.devices[0], args.source_commit)
    if args.plot:
        plot_losses(root, names)
    (root / "complete.json").write_text(json.dumps({"conditions": names, "source_commit": args.source_commit}) + "\n")


if __name__ == "__main__":
    main()
