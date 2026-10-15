"""Plot recorded geometry checks locally; not a training dependency."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--structure-run", type=Path, required=True)
    parser.add_argument("--coordinate-run", type=Path, required=True)
    parser.add_argument("--shifted-run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def rows(root, name):
        return [json.loads(x) for x in (root / name / "progress.jsonl").read_text().splitlines()]

    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.2), constrained_layout=True)
    finite = [(args.structure_run, "linf_joint_fixed", "Pair/rank loss", "#0072B2"),
              (args.coordinate_run, "linf_anchor_teacher_fixed", "Coordinate target", "#D55E00"),
              (args.shifted_run, "linf_anchor_teacher_shifted", "Translated target", "#009E73")]
    for root, name, label, color in finite:
        records = rows(root, name)
        x = [r["step"] / 1000 for r in records]
        axes[0, 0].plot(x, [r["fit"]["max_error"] for r in records], label=label, color=color)
        axes[0, 1].plot(x, [r["validation"]["mae"] for r in records], label=label, color=color)
    axes[0, 0].set(title="Finite set: maximum distance error", ylabel="Steps of error", yscale="log")
    axes[0, 1].set(title="New poses: distance MAE", ylabel="Steps of error")
    for col, intervention, metric, title in (
        (0, "linf_augmented_edge", "edge_mae", "New poses: edge-distance error"),
        (1, "linf_augmented_symmetry", "common_left_gap_mean", "New poses: symmetry gap")):
        for name, label, color in (("linf_augmented_control", "Control", "#0072B2"),
                                   (intervention, "Added constraint", "#D55E00")):
            records = rows(args.structure_run, name)
            axes[1, col].plot([r["step"] / 1000 for r in records],
                              [r["validation"][metric] for r in records], label=label, color=color)
        axes[1, col].set(title=title, ylabel="Steps of error")
    for ax in axes.flat:
        ax.set_xlabel("Updates (thousands)")
        ax.grid(alpha=.18)
        ax.legend(frameon=False)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
