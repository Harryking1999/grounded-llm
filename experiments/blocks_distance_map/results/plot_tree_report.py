"""Draw goal-conditioned decision-tree examples from the frozen tree Q-map."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import torch

from experiments.blocks_distance_map.src.model import BoardEncoder, board_bits, distance
from experiments.blocks_distance_map.src.oracle import DistanceOracle, successors


BLUE = "#39768A"
RED = "#C96560"
GREEN = "#18876E"
GOLD = "#D99B3D"
INK = "#202D34"


def set_style(font_file):
    font_manager.fontManager.addfont(str(font_file))
    family = font_manager.FontProperties(fname=str(font_file)).get_name()
    plt.rcParams.update({
        "font.family": family, "font.size": 11, "axes.unicode_minus": False,
        "savefig.dpi": 240,
    })


def save(fig, out, stem):
    fig.savefig(out / f"{stem}.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def load_model(checkpoint):
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model = BoardEncoder(saved["config"]["model"]["state_dim"],
                         saved["config"]["model"]["encoder_hidden_dim"],
                         saved["cap"], saved["metric"])
    model.load_state_dict(saved["model"])
    model.eval()
    return model, saved


def greedy_path(model, metric, source, goal, limit=12):
    path = [source]
    with torch.no_grad():
        goal_q = model.encode(board_bits([goal]))
        while path[-1] != goal and len(path) <= limit:
            children = [child for _, child in successors(path[-1])]
            if not children:
                break
            scores = distance(model.encode(board_bits(children)),
                              goal_q.expand(len(children), -1), metric)
            path.append(children[int(torch.argmin(scores))])
    if path[-1] != goal:
        raise AssertionError("Selected example does not reach its goal")
    return path


def complete_graph(source, max_states):
    seen, pending, edges = {source}, [source], set()
    while pending:
        state = pending.pop()
        for _, child in successors(state):
            edges.add((state, child))
            if child not in seen:
                seen.add(child)
                if len(seen) > max_states:
                    raise ValueError("Complete graph exceeds limit")
                pending.append(child)
    return seen, edges


def nearby_graph(sources, depth, paths):
    seen, frontier, edges = set(sources), set(sources), set()
    for _ in range(depth):
        new = set()
        for state in frontier:
            for _, child in successors(state):
                edges.add((state, child))
                if child not in seen:
                    new.add(child)
        seen.update(new)
        frontier = new
    for path in paths:
        seen.update(path)
        edges.update(zip(path[:-1], path[1:]))
    return seen, edges


def geometry(model, states, edges):
    masks = sorted(states, key=lambda state: (-state.bit_count(), state))
    ids = {state: i for i, state in enumerate(masks)}
    with torch.no_grad():
        q = model.encode(board_bits(masks)).numpy()
    centered = q - q.mean(axis=0)
    u, singular, _ = np.linalg.svd(centered, full_matrices=False)
    xy = u[:, :2] * singular[:2]
    edge_ids = np.asarray([(ids[a], ids[b]) for a, b in sorted(edges)], dtype=np.int32)
    return masks, ids, xy, edge_ids, (singular[:2] ** 2 / (singular ** 2).sum()).tolist()


def reachability(masks, goal, oracle):
    return np.asarray([oracle.distance(state, goal) >= 0 for state in masks])


def legend(fig):
    handles = [
        Line2D([], [], marker="o", ls="none", markerfacecolor=BLUE,
               markeredgecolor="white", markersize=8, label="可达当前目标"),
        Line2D([], [], marker="x", ls="none", color=RED,
               markersize=8, label="不可达当前目标"),
        Line2D([], [], color=GREEN, lw=2.5, marker="o",
               markersize=4, label="Q 的实际路径"),
        Line2D([], [], marker="s", ls="none", color=INK, markersize=8, label="起点"),
        Line2D([], [], marker="*", ls="none", color=GOLD, markersize=12, label="目标"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False,
               fontsize=9, bbox_to_anchor=(.5, .005))


def draw_graph(ax, masks, ids, xy, edges, labels, path, source, goal):
    ax.add_collection(LineCollection(xy[edges], colors="#91A7B0",
                                     linewidths=.42, alpha=.15, zorder=1))
    ax.scatter(xy[~labels, 0], xy[~labels, 1], marker="x", s=18,
               linewidths=.75, color=RED, alpha=.47, zorder=2)
    ax.scatter(xy[labels, 0], xy[labels, 1], s=37, linewidths=.35,
               facecolor=BLUE, edgecolor="white", alpha=.94, zorder=3)
    positions = [ids[state] for state in path]
    ax.plot(xy[positions, 0], xy[positions, 1], color=GREEN,
            linewidth=2.7, marker="o", markersize=4, zorder=4)
    ax.scatter(*xy[ids[source]], marker="s", s=125, color=INK, zorder=5)
    ax.scatter(*xy[ids[goal]], marker="*", s=250, color=GOLD,
               edgecolor=INK, linewidth=.6, zorder=6)
    ax.set_aspect("equal", adjustable="box")
    ax.margins(.12)
    ax.axis("off")


def task_pair(data, index):
    task = data["task_rows"][index]
    if not data["task_outcomes"]["tree"][index]:
        raise ValueError(f"Task {index} is not solved by the frozen Q")
    return int(task["source"]), int(task["goal"])


def plot_single(data, model, metric, out, task_index, max_states):
    source, goal = task_pair(data, task_index)
    states, edges = complete_graph(source, max_states)
    path = greedy_path(model, metric, source, goal)
    masks, ids, xy, edge_ids, variance = geometry(model, states, edges)
    labels = reachability(masks, goal, DistanceOracle(cache_limit=500_000, seconds=300))
    fig, ax = plt.subplots(figsize=(9.1, 6.7))
    draw_graph(ax, masks, ids, xy, edge_ids, labels, path, source, goal)
    legend(fig)
    fig.subplots_adjust(bottom=.1)
    save(fig, out, "tree_qmap_projection_full")
    return {"task_index": task_index, "states": len(masks), "edges": len(edge_ids),
            "reachable": int(labels.sum()), "path_steps": len(path)-1,
            "pca_variance": variance}


def plot_pair(data, model, metric, out, pair, name, depth):
    tasks = [task_pair(data, i) for i in pair]
    paths = [greedy_path(model, metric, source, goal) for source, goal in tasks]
    states, edges = nearby_graph([source for source, _ in tasks], depth, paths)
    masks, ids, xy, edge_ids, variance = geometry(model, states, edges)
    oracle = DistanceOracle(cache_limit=2_000_000, seconds=300)
    labels = [reachability(masks, goal, oracle) for _, goal in tasks]
    fig, axes = plt.subplots(1, 2, figsize=(14.6, 6.3))
    for ax, mask, path, (source, goal), panel in zip(
            axes, labels, paths, tasks, ("A", "B")):
        draw_graph(ax, masks, ids, xy, edge_ids, mask, path, source, goal)
        ax.text(.015, .98, panel, transform=ax.transAxes, va="top", ha="left",
                fontsize=15, fontweight="bold", color=INK)
    # Identical PCA coordinates and axis limits make the two panels directly comparable.
    limits = (min(ax.get_xlim()[0] for ax in axes), max(ax.get_xlim()[1] for ax in axes),
              min(ax.get_ylim()[0] for ax in axes), max(ax.get_ylim()[1] for ax in axes))
    for ax in axes:
        ax.set_xlim(limits[:2])
        ax.set_ylim(limits[2:])
    legend(fig)
    fig.subplots_adjust(left=.015, right=.985, bottom=.105, top=.98, wspace=.03)
    save(fig, out, f"tree_qmap_projection_{name}")
    return {"task_indices": list(pair), "states": len(masks), "edges": len(edge_ids),
            "reachable": [int(mask.sum()) for mask in labels],
            "label_changes": int(np.count_nonzero(labels[0] != labels[1])),
            "path_steps": [len(path)-1 for path in paths], "pca_variance": variance}


def schematic(out):
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.2))
    panels = [("清空目标", "000000", True), ("保留左端格", "100000", False)]
    boxes = {"S": (.50, .83, "111111"), "A": (.23, .51, "000111"),
             "B": (.77, .51, "100011")}
    for ax, (title, target, a_reaches) in zip(axes, panels):
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        ax.set_title(title, pad=7, fontsize=13)
        positions = {key: (x, y) for key, (x, y, _) in boxes.items()}
        positions["G"] = (.5, .17)
        arrows = [("S", "A", "#9AAAB1", "-"),
                  ("S", "B", "#9AAAB1", "-"),
                  ("A", "G", GREEN if a_reaches else RED,
                   "-" if a_reaches else "--"),
                  ("B", "G", RED if a_reaches else GREEN,
                   "--" if a_reaches else "-")]
        for start, end, color, line in arrows:
            x1, y1 = positions[start]
            x2, y2 = positions[end]
            ax.add_patch(FancyArrowPatch((x1, y1-.08), (x2, y2+.08),
                         arrowstyle="-|>", mutation_scale=13, lw=1.6,
                         linestyle=line, color=color, zorder=1))
        for name, (x, y, bits) in boxes.items():
            color = INK if name == "S" else (
                BLUE if (name == "A") == a_reaches else RED)
            ax.add_patch(FancyBboxPatch((x-.155, y-.073), .31, .146,
                         boxstyle="round,pad=0.009,rounding_size=.02",
                         facecolor="white", edgecolor=color, linewidth=1.7, zorder=2))
            ax.text(x, y, f"{name}  {bits}", ha="center", va="center",
                    fontfamily="DejaVu Sans Mono", fontsize=11, color=INK, zorder=3)
        x, y = positions["G"]
        ax.add_patch(FancyBboxPatch((x-.155, y-.073), .31, .146,
                     boxstyle="round,pad=0.009,rounding_size=.02",
                     facecolor="#FFF8ED", edgecolor=GOLD, linewidth=1.7, zorder=2))
        ax.text(x, y, f"G  {target}", ha="center", va="center",
                fontfamily="DejaVu Sans Mono", fontsize=11, color=INK, zorder=3)
    fig.subplots_adjust(wspace=.11)
    save(fig, out, "tree_qmap_supervision_example")


def plot_distance_relation(dataset_path, model, metric, out):
    dataset = np.load(dataset_path)
    pairs = dataset["test_ood_board"]
    pair_board = dataset["pair_board"]
    needed = np.unique(pairs[:, :2])
    states = dataset["states"]
    with torch.no_grad():
        embeddings = torch.cat([
            model.encode(board_bits([int(states[i]) for i in needed[j:j+2048]]))
            for j in range(0, len(needed), 2048)
        ])
        source_ids = torch.from_numpy(np.searchsorted(needed, pairs[:, 0]))
        target_ids = torch.from_numpy(np.searchsorted(needed, pairs[:, 1]))
        predicted = distance(embeddings[source_ids], embeddings[target_ids],
                             metric).numpy()
    groups = [(str(step), pairs[:, 2] == step) for step in range(1, 8)]
    groups.append(("8+", pairs[:, 2] >= 8))
    groups.append(("不可达", pairs[:, 2] < 0))
    fig, ax = plt.subplots(figsize=(9.0, 4.7))
    rng = np.random.default_rng(20261009)
    records = []
    board_count = int(pair_board.max()) + 1
    for label, selected in groups:
        values = predicted[selected]
        boards = pair_board[selected]
        sums = np.bincount(boards, weights=values, minlength=board_count)
        counts = np.bincount(boards, minlength=board_count)
        sampled = rng.integers(0, board_count, size=(2000, board_count))
        boot = sums[sampled].sum(axis=1) / counts[sampled].sum(axis=1)
        low, high = np.quantile(boot, [.025, .975])
        records.append({"true_distance": label, "count": len(values),
                        "mean": float(values.mean()),
                        "ci95_low": float(low), "ci95_high": float(high)})
    x = np.arange(1, 9)
    y = np.asarray([row["mean"] for row in records[:8]])
    low = np.asarray([row["ci95_low"] for row in records[:8]])
    high = np.asarray([row["ci95_high"] for row in records[:8]])
    ax.plot(x, y, marker="o", markersize=6, linewidth=2.6, color=BLUE)
    ax.fill_between(x, low, high, color=BLUE, alpha=.23)
    ax.errorbar(x, y, yerr=[y-low, high-y], fmt="none", color=BLUE,
                elinewidth=1.8, capsize=4)
    unreachable = records[-1]
    ax.errorbar([9], [unreachable["mean"]],
                yerr=[[unreachable["mean"]-unreachable["ci95_low"]],
                      [unreachable["ci95_high"]-unreachable["mean"]]],
                fmt="o", color=RED, markersize=7, elinewidth=2, capsize=4)
    ax.set_xticks(range(1, len(groups)+1), [label for label, _ in groups])
    ax.set_ylabel("Q 预测的有向距离")
    ax.set_xlabel("真实最短步数")
    ax.set_xlim(.45, len(groups)+.55)
    ax.set_ylim(0, max(10, unreachable["ci95_high"] + 1))
    ax.grid(axis="y", color="#E4EAEC", linewidth=.7)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, out, "tree_qmap_distance_relation")
    return {"split": "test_ood_board", "pairs": int(len(pairs)), "groups": records}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--font-file", type=Path, required=True)
    parser.add_argument("--full-task", type=int, default=595)
    parser.add_argument("--goal-switch-tasks", type=int, nargs=2, default=(371, 373))
    parser.add_argument("--start-switch-tasks", type=int, nargs=2, default=(537, 539))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    set_style(args.font_file)
    data = json.loads(args.comparison.read_text(encoding="utf-8"))
    model, saved = load_model(args.checkpoint)
    schematic(args.out)
    distance_record = plot_distance_relation(args.dataset, model, saved["metric"], args.out)
    figures = {
        "complete_isolated_goal": plot_single(data, model, saved["metric"], args.out,
                                               args.full_task, 4096),
        "same_start_two_goals": plot_pair(data, model, saved["metric"], args.out,
                                          args.goal_switch_tasks, "goal_switch", 2),
        "same_goal_two_starts": plot_pair(data, model, saved["metric"], args.out,
                                          args.start_switch_tasks, "start_switch", 2),
    }
    record = {
        "checkpoint_source_commit": saved["source_commit"],
        "evaluation_analysis_commit": data["analysis_commit"],
        "fresh_boards": data["boards"], "tasks": data["tasks"],
        "tree_outcomes": data["results"]["tree"],
        "distance_relation": distance_record,
        "figures": figures,
    }
    (args.out / "tree_qmap_figure_data.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(figures, ensure_ascii=False))


if __name__ == "__main__":
    main()
