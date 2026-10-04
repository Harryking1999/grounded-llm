"""Render the human-facing figures from the compact exploratory summary.

Requires matplotlib; the training code does not depend on it.
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib import font_manager
from matplotlib.patches import Rectangle


HERE = Path(__file__).resolve().parent
DATA = json.loads((HERE / "summary.json").read_text(encoding="utf-8"))
FONT = Path("C:/Windows/Fonts/msyh.ttc")
if FONT.exists():
    font_manager.fontManager.addfont(str(FONT))
    plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(FONT)).get_name()
plt.rcParams.update({"axes.unicode_minus": False, "figure.dpi": 120,
                     "savefig.dpi": 180, "font.size": 11})
BLUE, ORANGE = "#27617D", "#D98B28"


def get(run):
    return next(row for row in DATA["conditions"] if row["run"] == run)


def make_outcomes():
    rows = [
        ("A · 稀疏监督，网络算坐标", get("sparse_encoder_a3d520a")),
        ("A · 增加地标，逐状态存坐标", get("dense_direction_a43b352")),
        ("A · 增加地标，网络算坐标", get("encoder_c40e5ba")),
        ("B · 在 B 上重新训练网络", get("second_board_encoder_7e9e153")),
        ("B · 直接使用 A 的网络", get("encoder_c40e5ba/zero_shot_second_board_rollout")),
    ]
    fig, ax = plt.subplots(figsize=(10.5, 5.3), layout="constrained")
    for i, (_, row) in enumerate(rows):
        y = len(rows) - 1 - i
        one = row["one_step_solvable"] * 100
        rollout = row["rollout_solved"] / row["rollout_starts"] * 100
        ax.barh(y + .18, one, height=.32, color=BLUE,
                label="单步选对" if i == 0 else None)
        ax.barh(y - .18, rollout, height=.32, color=ORANGE,
                label="完整清空" if i == 0 else None)
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


def draw_board(ax, rows, origin, removed=()):
    matrix = [[int(ch) for ch in row] for row in rows]
    ax.imshow(matrix, cmap=ListedColormap(["#F1F4F5", BLUE]), vmin=0, vmax=1)
    for r, c in removed:
        assert matrix[r][c] == 1, "A highlighted removal must cover an occupied cell"
        ax.add_patch(Rectangle((c - .5, r - .5), 1, 1, facecolor=ORANGE,
                               edgecolor="#946021", linewidth=.4, hatch="///"))
    ax.set_xticks([i - .5 for i in range(len(rows[0]) + 1)], minor=True)
    ax.set_yticks([i - .5 for i in range(len(rows) + 1)], minor=True)
    ax.grid(which="minor", color="white", lw=2)
    ax.set_xticks(range(len(rows[0])), range(origin[1] + 1, origin[1] + len(rows[0]) + 1))
    ax.set_yticks(range(len(rows)), range(origin[0] + 1, origin[0] + len(rows) + 1))
    ax.tick_params(which="both", length=0, labelsize=9, colors="#65747B")
    ax.spines[:].set_visible(False)


def make_case(filename, case, right_action_name):
    # The same crop and coordinate frame is used in all four panels.
    rows = case["board_grid"]
    height = len(rows) / len(rows[0]) * 7 + 1.4
    fig, axes = plt.subplots(2, 2, figsize=(8.4, height), layout="constrained")
    for col, (key, label) in enumerate([
            ("good_action", "左：移除三格拐角"),
            ("bad_action", f"右：移除{right_action_name}")]):
        action = case[key]
        removed = action["removed_cells_cropped"]
        expected = [list(row) for row in rows]
        for r, c in removed:
            expected[r][c] = "0"
        assert ["".join(row) for row in expected] == action["after_grid_cropped"]
        draw_board(axes[0, col], rows, case["crop_origin"], removed)
        draw_board(axes[1, col], action["after_grid_cropped"], case["crop_origin"])
        axes[0, col].set_title(label, fontsize=13, pad=10)
        axes[1, col].set_title("移除后", fontsize=12, pad=10)
    fig.suptitle(f"棋盘 {case['board']} · 两个合法动作", fontsize=16)
    fig.savefig(HERE / filename, bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    make_outcomes()
    make_case("qmap_case_supervision.png", DATA["case_sparse_vs_landmark"], "三格竖条")
    make_case("qmap_case_transfer.png", DATA["case_cross_board_transfer"], "三格横条")
