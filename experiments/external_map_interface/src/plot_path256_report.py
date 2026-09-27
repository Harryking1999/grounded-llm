"""Draw report figures from the compact five-graph results; no model calls."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


ROOT = Path(__file__).resolve().parents[3]
RESULTS = ROOT / "experiments/external_map_interface/results"
STYLES = [
    ("instruct", "Instruct", "#4477AA", "o"),
    ("instruct_distance", "Instruct＋地图", "#EE7733", "s"),
    ("thinking", "Thinking", "#AA3377", "^"),
]


def setup():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["SimHei", "Microsoft YaHei", "Noto Sans CJK SC", "DejaVu Sans"],
        "font.size": 12,
        "axes.labelsize": 13,
        "axes.titlesize": 14,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#A4ADB5",
        "text.color": "#26323D",
        "axes.labelcolor": "#26323D",
        "xtick.color": "#52616E",
        "ytick.color": "#52616E",
        "legend.frameon": False,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    })


def save(fig, name, output, exports):
    fig.savefig(str(output / (name + ".png")), dpi=300, facecolor="white")
    for extension in ["svg", "pdf"]:
        fig.savefig(str(exports / (name + "." + extension)), facecolor="white")
    plt.close(fig)


def line_figure(title, xlabel, ticks, xlim):
    fig, ax = plt.subplots(figsize=(9.4, 5.6))
    fig.subplots_adjust(left=.11, right=.94, bottom=.13, top=.78)
    fig.suptitle(title, x=.11, ha="left", y=.97, fontsize=17)
    ax.set(xlabel=xlabel, ylabel="最短解成功概率", ylim=(0, 105), xlim=xlim)
    ax.set_xticks(ticks)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, pos: "{:.0f}%".format(value)))
    ax.grid(axis="y", color="#E5E9ED", linewidth=.8)
    ax.set_axisbelow(True)
    return fig, ax


def pass_curve(data, output, exports):
    fig, ax = line_figure("不同采样次数的 pass@k", "采样次数 k", [1, 8, 16], (0, 18))
    for condition, label, color, marker in STYLES:
        values = [100 * data["conditions"][condition]["pass_at"][str(k)] for k in [1, 8, 16]]
        ax.plot([1, 8, 16], values, color=color, marker=marker, markersize=8,
                linewidth=2.3, label=label)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.03), ncol=3, borderaxespad=0)
    save(fig, "path256_five_graphs_pass", output, exports)


def length_curve(data, output, exports):
    lengths = [4, 6, 8, 10]
    fig, ax = line_figure("不同路径长度的 pass@1", "题目的最短路长度（步）", lengths, (3.65, 10.85))
    for condition, label, color, marker in STYLES:
        values = [100 * data["by_shortest_length"][str(n)][condition]["pass_at"]["1"] for n in lengths]
        ax.plot(lengths, values, color=color, marker=marker, markersize=8,
                linewidth=2.3, label=label)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.03), ncol=3, borderaxespad=0)
    save(fig, "path256_five_graphs_length", output, exports)


def cost_figure(data, output, exports):
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.5))
    fig.subplots_adjust(left=.17, right=.97, bottom=.17, top=.83, wspace=.55)
    conditions = [data["conditions"][condition] for condition, _, _, _ in STYLES]
    values = [
        [c["inference_cost"]["mean_output_tokens_per_decision"] for c in conditions],
        [100 * c["failures"].get("budget_truncated", 0) / c["trials"] for c in conditions],
    ]
    for ax, nums, title, unit, limit in zip(axes, values,
            ["每次决策平均输出", "发生截断的轨迹比例"],
            ["输出 tokens", "轨迹比例"], [8500, 80]):
        for i, ((_, label, color, _), value) in enumerate(zip(STYLES, nums)):
            ax.barh(2-i, value, color=color, height=.48)
        ax.set_yticks([2, 1, 0])
        ax.set_yticklabels([s[1] for s in STYLES])
        ax.set(xlim=(0, limit), ylim=(-.65, 2.65), xlabel=unit, title=title)
        ax.grid(axis="x", color="#E5E9ED", linewidth=.8)
        ax.set_axisbelow(True)
        if unit == "轨迹比例":
            ax.xaxis.set_major_formatter(FuncFormatter(lambda value, pos: "{:.0f}%".format(value)))
    save(fig, "path256_five_graphs_cost", output, exports)


def case_figure(data, output, exports):
    cases = data["case_studies"]["cases"]
    fig = plt.figure(figsize=(11.4, 6.2))
    fig.suptitle("三个案例的最短解次数与局部分叉", x=.04, ha="left", y=.97, fontsize=17)
    ax = fig.add_axes([.18, .13, .30, .68])
    labels = ["地图增益\nGraph 04 · 题 00", "输出截断，无增益\nGraph 04 · 题 13", "局部排序错误\nGraph 01 · 题 17"]
    for index, case in enumerate(cases):
        for condition_index, (condition, label, color, _) in enumerate(STYLES[:2]):
            y = 2-index + (.16 if condition_index == 0 else -.16)
            count = case["counts"][condition]["shortest"]
            ax.barh(y, count, height=.26, color=color, label=label if index == 0 else None)
            if count == 0:
                ax.plot(0, y, "|", color=color, markersize=12, markeredgewidth=2)
            ax.text(count+.35, y, "{}/16".format(count), va="center", color=color, fontsize=12)
    ax.set_yticks([2, 1, 0])
    ax.set_yticklabels(labels, fontsize=11)
    ax.set(xlim=(0, 16), ylim=(-.55, 2.55), xlabel="完成最短解的采样次数")
    ax.set_xticks([0, 4, 8, 12, 16])
    ax.grid(axis="x", color="#E5E9ED", linewidth=.8)
    ax.set_axisbelow(True)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.03), ncol=2, fontsize=11, borderaxespad=0)

    branch = fig.add_axes([.54, .13, .43, .68])
    branch.set(xlim=(0, 1), ylim=(0, 1))
    branch.axis("off")
    branch.text(.04, .98, "Graph 01 · 题 17 的一次实际分叉", fontsize=13, va="top")
    branch.text(.04, .85, "共同前缀 251 → 218 → 63 → 196；目标 152", fontsize=10.5)
    branch.text(.08, .47, "196", ha="center", va="center", fontsize=14,
                bbox={"boxstyle": "circle,pad=.5", "facecolor": "#F0F3F5", "edgecolor": "#A4ADB5"})
    for y, candidate, style, explanation in [
            (.65, cases[2]["candidates"][0], STYLES[0], "Instruct 选择"),
            (.20, cases[2]["candidates"][1], STYLES[1], "地图组选择（较小预测距离）")]:
        _, _, color, _ = style
        branch.annotate("", xy=(.40, y+.06), xytext=(.17, .47),
                        arrowprops={"arrowstyle": "->", "color": color, "lw": 2})
        branch.text(.43, y+.12, "节点 {}".format(candidate["to_node"]), fontsize=14, color=color)
        branch.text(.43, y+.035, explanation, fontsize=10.5, color=color)
        branch.text(.43, y-.09, "地图距离 {:.3f}\n实际还需 {} 步".format(
            candidate["learned_distance"], candidate["true_remaining_moves_posthoc"]), fontsize=11, linespacing=1.6)
    save(fig, "path256_five_graphs_cases", output, exports)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=RESULTS / "path256_five_graphs.json")
    parser.add_argument("--output", type=Path, default=RESULTS)
    parser.add_argument("--exports", type=Path, default=ROOT / "runs/external_map_interface/report_figures")
    args = parser.parse_args()
    data = json.loads(args.summary.read_text(encoding="utf-8"))
    args.output.mkdir(parents=True, exist_ok=True)
    args.exports.mkdir(parents=True, exist_ok=True)
    setup()
    for plot in [pass_curve, length_curve, cost_figure, case_figure]:
        plot(data, args.output, args.exports)
    print("Saved four report PNGs and editable SVG/PDF exports.")


if __name__ == "__main__":
    main()
