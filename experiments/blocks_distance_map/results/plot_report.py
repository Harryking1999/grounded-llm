"""Render the human-facing figures from the compact exploratory summary.

Requires matplotlib; the training code does not depend on it.
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib import font_manager


HERE = Path(__file__).resolve().parent
DATA = json.loads((HERE / "summary.json").read_text(encoding="utf-8"))
FONT = Path("C:/Windows/Fonts/msyh.ttc")
if FONT.exists():
    font_manager.fontManager.addfont(str(FONT))
    plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(FONT)).get_name()
plt.rcParams.update({"axes.unicode_minus": False, "figure.dpi": 120,
                     "savefig.dpi": 180, "font.size": 11})
BLUE, ORANGE, GREEN, RED = "#27617D", "#D98B28", "#278167", "#C65049"


def get(run):
    return next(row for row in DATA["conditions"] if row["run"] == run)


def make_outcomes():
    rows = [
        ("A 稀疏监督 · 棋盘编码器", get("sparse_encoder_a3d520a")),
        ("A 地标监督 · 查表 Q", get("dense_direction_a43b352")),
        ("A 地标监督 · 棋盘编码器", get("encoder_c40e5ba")),
        ("B 独立训练 · 棋盘编码器", get("second_board_encoder_7e9e153")),
        ("B 直接使用 A 的编码器", get("encoder_c40e5ba/zero_shot_second_board_rollout")),
    ]
    fig, ax = plt.subplots(figsize=(10.5, 5.3), layout="constrained")
    for i, (_, row) in enumerate(rows):
        y = len(rows) - 1 - i
        one = row["one_step_solvable"] * 100
        rollout = row["rollout_solved"] / row["rollout_starts"] * 100
        ax.barh(y + .18, one, height=.32, color=BLUE,
                label="单步可解率" if i == 0 else None)
        ax.barh(y - .18, rollout, height=.32, color=ORANGE,
                label="逐步清空率" if i == 0 else None)
    ax.set_yticks(range(len(rows)), [label for label, _ in rows[::-1]])
    ax.set_xlim(0, 100)
    ax.set_ylim(-.6, 4.6)
    ax.set_xticks([0, 20, 40, 60, 80, 100], ["0%", "20%", "40%", "60%", "80%", "100%"])
    ax.grid(axis="x", color="#E5EBEE")
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0, pad=8)
    ax.set_title("单步选择与完整清空", loc="left", fontsize=16, pad=12)
    ax.legend(loc="upper center", bbox_to_anchor=(.58, 1.03), ncol=2, frameon=False)
    fig.savefig(HERE / "qmap_outcomes.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def draw_board(ax, rows):
    matrix = [[int(ch) for ch in row] for row in rows]
    ax.imshow(matrix, cmap=ListedColormap(["#F1F4F5", "#27617D"]), vmin=0, vmax=1)
    ax.set_xticks([i - .5 for i in range(len(rows[0]) + 1)], minor=True)
    ax.set_yticks([i - .5 for i in range(len(rows) + 1)], minor=True)
    ax.grid(which="minor", color="white", lw=2)
    ax.tick_params(which="both", bottom=False, left=False,
                   labelbottom=False, labelleft=False)
    ax.spines[:].set_visible(False)


def make_case(filename, case, model_names, values, title):
    fig = plt.figure(figsize=(10.8, 4.7), layout="constrained")
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 2.25])
    left = fig.add_subplot(gs[0, 0])
    right = fig.add_subplot(gs[0, 1])
    draw_board(left, case["board_grid"])
    left.set_title("当前棋盘", fontsize=12, pad=12)
    x = [0, 1]
    width = .32
    good = [pair[0] for pair in values]
    bad = [pair[1] for pair in values]
    right.bar([i - width / 2 for i in x], good, width,
              color=GREEN, label="可解后继")
    right.bar([i + width / 2 for i in x], bad, width,
              color=RED, label="死局后继")
    right.set_xticks(x, model_names)
    right.set_ylabel("Q 距离")
    right.set_ylim(0, max(good + bad) * 1.20)
    right.grid(axis="y", color="#E5EBEE")
    right.set_axisbelow(True)
    right.spines[["top", "right"]].set_visible(False)
    right.legend(loc="upper center", ncol=2, frameon=False)
    fig.suptitle(title, fontsize=16, x=.04, ha="left")
    fig.savefig(HERE / filename, bbox_inches="tight", facecolor="white")
    plt.close(fig)


make_outcomes()
case = DATA["case_sparse_vs_landmark"]
make_case("qmap_case_supervision.png", case, ["稀疏监督", "加入地标监督"],
          [(case["good_sparse_score"], case["bad_sparse_score"]),
           (case["good_landmark_score"], case["bad_landmark_score"])],
          "棋盘 A：候选评分")
case = DATA["case_cross_board_transfer"]
make_case("qmap_case_transfer.png", case, ["A 模型直接迁移", "在 B 上独立训练"],
          [(case["good_transfer_score"], case["bad_transfer_score"]),
           (case["good_trained_score"], case["bad_trained_score"])],
          "棋盘 B：候选评分")
