"""Plot recorded fixed-set losses, without smoothing or further optimization."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments.puzzle_qmap.src.sampling import task_indices
from experiments.puzzle_qmap.src.tasks import make_environment


def plot(run, output):
    config = json.loads((run / "config.json").read_text())
    summary = json.loads((run / "summary.json").read_text())
    rows = [json.loads(line) for line in (run / "progress.jsonl").read_text().splitlines()]
    initial = summary["initial_training_monitor"]
    step = np.r_[0, [r["step"] for r in rows]] / 1000
    train_huber = np.r_[initial["pairs"]["huber"], [r["training_monitor"]["pairs"]["huber"] for r in rows]]
    train_rank = np.r_[initial["soft_rank_loss"], [r["training_monitor"]["soft_rank_loss"] for r in rows]]
    val_huber = np.r_[summary["initial_validation_pairs"]["huber"], [r["validation_huber"] for r in rows]]
    val_total = np.r_[summary["initial_validation_objective"], [r["validation_objective"] for r in rows]]
    weight = config["training"]["rank_weight"]
    val_rank = (val_total - val_huber) / weight
    data = dict(np.load(run / "data.npz", allow_pickle=False))
    env = make_environment(config["task"], run / "oracle_depth.npy")
    idx = task_indices(env, data, "validation", config["evaluation"]["decision_count"], config)
    logits = -data["child_distances_validation"][idx].astype(float) / config["training"]["rank_temperature"]
    logits -= logits.max(axis=1, keepdims=True)
    log_prob = logits - np.log(np.exp(logits).sum(axis=1, keepdims=True))
    entropy = float(-(np.exp(log_prob) * log_prob).sum(axis=1).mean())
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), constrained_layout=True)
    lines = ((train_huber, val_huber, "Distance Huber"),
             (train_rank, val_rank, "Full soft ranking CE"),
             (train_huber + weight * train_rank, val_total, "Total loss"))
    for ax, (a, b, title) in zip(axes, lines):
        ax.plot(step, a, color="#2166ac", label="Fixed training monitor", linewidth=1.7)
        ax.plot(step, b, color="#b35806", label="Validation", linewidth=1.7)
        ax.axvline(summary["best_step"] / 1000, color="0.5", linestyle=":", linewidth=1)
        ax.set(xlabel="Updates (thousands)", ylabel="Loss", title=title, xlim=(0, step[-1]))
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="0.9", linewidth=.6)
    axes[1].axhline(entropy, color="#b35806", linestyle="--", linewidth=1, label="Validation target entropy")
    axes[1].axhline(initial["rank_target_entropy"], color="#2166ac", linestyle="--", linewidth=1,
                    label="Training target entropy")
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].legend(frameon=False, fontsize=7, loc="center left")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    plt.close(fig)
    return entropy


def plot_progress(control, treatment, output):
    fig, axes = plt.subplots(2, 2, figsize=(10, 6), constrained_layout=True)
    for run, color, label in ((control, "#2166ac", "Absolute-distance control"),
                              (treatment, "#b35806", "Edge-difference supervision")):
        summary = json.loads((run / "summary.json").read_text())
        rows = [json.loads(line) for line in (run / "progress.jsonl").read_text().splitlines()]
        step = np.array([row["step"] for row in rows]) / 1000
        series = ([row["validation_edge_difference_huber"] for row in rows],
                  [row["validation_soft_rank_loss"] for row in rows],
                  np.array([row["validation_shortest_action_rate"] for row in rows]) * 100,
                  [row["validation_huber"] for row in rows])
        for ax, values in zip(axes.ravel(), series):
            ax.plot(step, values, color=color, label=label, linewidth=1.7)
        axes[1, 0].axvline(summary["best_step"] / 1000, color=color, linestyle=":", linewidth=1)
    for ax, title, ylabel in zip(axes.ravel(),
        ("Validation edge-difference Huber", "Validation full soft ranking CE",
         "Validation shortest-action rate", "Validation absolute-distance Huber"),
        ("Loss", "Loss", "Percent", "Loss")):
        ax.set(xlabel="Additional updates (thousands)", ylabel=ylabel, title=title)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="0.9", linewidth=.6)
    axes[0, 0].legend(frameon=False, fontsize=8)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--control", type=Path, help="Plot a matched progress-supervision comparison")
    args = parser.parse_args()
    if args.control is None:
        print(json.dumps({"validation_target_entropy": plot(args.run, args.output)}))
    else:
        plot_progress(args.control, args.run, args.output)
