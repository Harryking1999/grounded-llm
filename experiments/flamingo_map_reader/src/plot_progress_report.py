"""Reporting figures from the frozen, complete evaluation summaries.

Quantitative panels: task completion, readout learning, then within-trajectory
behaviour. No smoothing, synthetic points, cross-run pooling or seed error bars.
PNG is for the report; SVG and CSV preserve editable plots and source values.
"""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter, MaxNLocator

RESULTS = Path(__file__).resolve().parents[1] / "results"
BLUE, ORANGE, GRAY = "#326B9B", "#C8753F", "#7D8790"


def load(name):
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))


def setup():
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Microsoft YaHei", "DejaVu Sans"],
        "font.size": 10, "axes.titlesize": 12, "axes.labelsize": 10,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "#AFB6BC", "axes.labelcolor": "#28333D",
        "text.color": "#28333D", "axes.unicode_minus": False,
        "legend.frameon": False, "svg.fonttype": "none", "pdf.fonttype": 42,
        "lines.linewidth": 2, "savefig.facecolor": "white",
    })


def tidy(ax, xlabel, percent=False):
    ax.set_xlabel(xlabel)
    ax.grid(axis="y", color="#E8ECEF", linewidth=.7)
    ax.set_axisbelow(True)
    if percent:
        ax.yaxis.set_major_formatter(PercentFormatter(1))


def save(fig, name, rows, out, vector):
    fig.savefig(out / f"{name}.png", dpi=220, bbox_inches="tight")
    fig.savefig(vector / f"{name}.svg", bbox_inches="tight")
    with (vector / f"{name}.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=RESULTS / "figures")
    parser.add_argument("--vector-output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    args.vector_output.mkdir(parents=True, exist_ok=True)
    setup()
    blocks, old = load("blocks_results_summary.json"), load("long_trajectory_summary.json")
    baseline = load("learning_curve_baselines.json")["tasks"]["path"]["reference"]
    path = []
    for epoch, checkpoint in [(1, "checkpoint-24192"), (2, "checkpoint-48384"), (3, "final")]:
        roll = old["evaluation_batteries"][f"path/{checkpoint}/validation_rollout_map"]
        ref = old["evaluation_batteries"][f"path/{checkpoint}/validation_reference_map"]
        total, zero = roll["rollout"], roll["groups"]["initial_goal"]["rollout"]
        n = total["attempts"] - zero["attempts"]
        assert n == 500 and roll["completion"] == ref["completion"] == "complete"
        audit = ref["reachability_audit"]
        path.append(dict(epoch=epoch, cases=n,
            reached=(total["reached"]-zero["reached"])/n,
            shortest=(total["reached_shortest"]-zero["reached_shortest"])/n,
            reference_shortest=audit["action_environment_shortest"]/audit["current_reachable_turns"]))
    fig, axs = plt.subplots(1, 2, figsize=(10.4, 3.7), layout="constrained")
    xx = [r["epoch"] for r in path]
    for key, label, color in [("reached", "到达目标", BLUE), ("shortest", "到达且最短", ORANGE)]:
        yy = [r[key] for r in path]
        axs[0].plot(xx, yy, "o-", label=label, color=color)
        axs[0].annotate(f"{yy[-1]:.1%}", (3, yy[-1]), xytext=(-8, 8), textcoords="offset points", ha="right")
    axs[0].set_title("a  寻路：自主执行（500 个非零步任务）", loc="left")
    axs[0].set_ylim(0, 1.02); axs[0].legend(loc="center left")
    axs[1].plot(xx, [r["reference_shortest"] for r in path], "o-", color=BLUE, label="模型：reference")
    axs[1].axhline(baseline["chance_shortest"], color=GRAY, linestyle=":", label="同状态均匀随机")
    axs[1].axhline(baseline["greedy_shortest"], color=ORANGE, linestyle="--", label="同状态地图贪心")
    axs[1].set_ylim(.45, .95); axs[1].set_title("b  单步最短路动作率", loc="left")
    axs[1].legend(loc="lower right", fontsize=9)
    for ax in axs:
        tidy(ax, "训练 epoch", True); ax.set_xticks([1, 2, 3]); ax.set_xlim(.85, 3.15)
    save(fig, "progress_path_training", path, args.output, args.vector_output)

    rows = []
    for step, cp in sorted(blocks["checkpoints"].items(), key=lambda item: int(item[0])):
        r, f = (cp["modes"][mode] for mode in ("test_reference", "test_rollout"))
        assert r["cases"] == f["cases"] == 510
        row = dict(epoch=cp["epoch"], cases=f["cases"], reached=f["reached"]/510, stopped=f["success"]/510)
        for tag, mode in [("R", r), ("F", f)]:
            a = mode["all_decisions"]
            row[f"safe_{tag}"] = mode["safe"] / mode["solvable_decisions"]
            row[f"map_{tag}"] = a["map_minimum"] / a["decisions"]
            for metric in ("inverse_rank", "minmax_distance"):
                for k in (1, 10):
                    name = f"ndcg_{metric}_at_{k}"
                    row[f"{name}_{tag}"] = a[name]
            row[f"distance_random_{tag}"] = a["normalized_distance_baselines"]["ndcg_minmax_distance_at_10"]["random"]
        rows.append(row)
    assert len(rows) == 40
    xx = [r["epoch"] for r in rows]
    fig, axs = plt.subplots(1, 2, figsize=(10.4, 3.9), layout="constrained")
    for key, label, color in [("reached", "实际到达", BLUE), ("stopped", "到达且正确停止", ORANGE)]:
        axs[0].plot(xx, [r[key] for r in rows], color=color, label=label)
    greedy = blocks["map_greedy_comparison"]["results"]["current"]["all"]["rates"]["reached"]
    axs[0].axhline(greedy, color=GRAY, linestyle="--", label=f"同题地图贪心 {greedy:.1%}")
    best = max(rows, key=lambda r:r["reached"])
    axs[0].plot(best["epoch"], best["reached"], "o", color=BLUE)
    axs[0].annotate(f"19.5 epoch：{best['reached']:.1%}", (best["epoch"],best["reached"]),
                    xytext=(-12, 18), textcoords="offset points", ha="right", fontsize=10)
    axs[0].set_ylim(0,.78); axs[0].set_title("a  积木：完整任务（同一批 510 题）", loc="left")
    axs[0].legend(loc="upper left", bbox_to_anchor=(.01, .80), fontsize=9)
    for tag, label, color in [("R", "示范历史 R", BLUE), ("F", "自身历史 F", ORANGE)]:
        axs[1].plot(xx, [r[f"safe_{tag}"] for r in rows], color=color, label=f"可达率 · {label}")
        axs[1].plot(xx, [r[f"map_{tag}"] for r in rows], color=color, linestyle="--", label=f"地图最优率 · {label}")
    axs[1].set_ylim(0,1); axs[1].set_title("b  单步动作：保住可达与选中地图最优", loc="left")
    axs[1].legend(loc="lower right", fontsize=8.5)
    for ax in axs: tidy(ax, "训练 epoch", True); ax.set_xlim(.5,20); ax.set_xticks([.5,5,10,15,20])
    save(fig, "progress_blocks_training", rows, args.output, args.vector_output)

    fig, axs = plt.subplots(1,2,figsize=(10.4,3.8),layout="constrained")
    for ax, metric, title in zip(axs,["inverse_rank","minmax_distance"],["a  NDCG@10：位置倒数", "b  NDCG@10：Q-map 距离归一化"]):
        for tag,label,color in [("R","示范历史 R",BLUE),("F","自身历史 F",ORANGE)]:
            ax.plot(xx,[r[f"ndcg_{metric}_at_10_{tag}"] for r in rows],color=color,label=label)
        if metric == "minmax_distance":
            ax.plot(xx,[r["distance_random_F"] for r in rows],color=GRAY,linestyle=":",label="F 同状态随机排序")
        ax.set_title(title,loc="left"); ax.set_ylim(0,1); ax.set_xlim(.5,20); ax.set_xticks([.5,5,10,15,20])
        tidy(ax,"训练 epoch"); ax.legend(loc="lower right",fontsize=9)
    save(fig,"progress_blocks_ranking",rows,args.output,args.vector_output)

    steps = load("final_trajectory_step_analysis.json")["tasks"]["path"]
    fig, axs = plt.subplots(1,2,figsize=(10.4,3.8),layout="constrained")
    source=[]
    for mode,label,color in [("reference","示范历史 R",BLUE),("rollout","自身历史 F",ORANGE)]:
        rr=steps["modes"][mode]["by_step"]
        axs[0].plot([x["step"] for x in rr],[x["shortest"]/x["solvable"] if x["solvable"] else float("nan") for x in rr],color=color,label=label)
        for x in rr: source.append(dict(series=mode,step=x["step"],numerator=x["shortest"],denominator=x["solvable"]))
    for mode,label,color,style in [("rollout","模型自主执行",ORANGE,"-"),("greedy","地图贪心",GRAY,"--")]:
        rr=steps["progress"][mode]
        axs[1].plot([x["step"] for x in rr],[x["reached"]/x["denominator"] for x in rr],color=color,linestyle=style,label=label)
        for x in rr: source.append(dict(series=f"cumulative_{mode}",step=x["step"],numerator=x["reached"],denominator=x["denominator"]))
    for ax in axs:
        tidy(ax,"决策轮次 / 已执行动作数",True); ax.set_ylim(0,1.04); ax.legend(loc="lower right",fontsize=9)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True,nbins=7))
    axs[0].set_title("a  寻路：各轮最短路动作率",loc="left")
    axs[1].set_title("b  累计到达（固定分母 500 题）",loc="left")
    save(fig,"progress_path_steps",source,args.output,args.vector_output)
    print("Created four report figures from complete saved summaries.")


if __name__ == "__main__":
    main()
