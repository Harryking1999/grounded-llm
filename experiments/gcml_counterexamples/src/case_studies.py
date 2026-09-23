"""Offline Luna case replay, exact counterfactuals, figures and a Chinese report.

Uses retained completed answers, never calls a model. Selection is explicitly
post hoc. Shortest-path deviations are diagnosed by distance, not by whether a
route differs from an arbitrarily chosen BFS witness.
"""
import argparse
from collections import Counter, defaultdict
import html
import json
from pathlib import Path

import blocks
import path as paths
from run import evaluate, parse_output

ROOT = Path(__file__).resolve().parents[3]
CONDITIONS = {
    "blocks8": "原始 8 块积木", "blocks12": "生成 12 块积木",
    "path_shortest": "32 节点原图最短路", "path_gates1": "32 节点单开关最短路",
    "path_gates2": "32 节点双开关最短路",
}
DARK, ORANGE, RED, BLUE, GREEN = "#25354a", "#e47742", "#c83c4b", "#2467a4", "#168064"


def read_json(filename):
    return json.loads(filename.read_text(encoding="utf-8"))


def action_text(action):
    return f"shape {action['shape_id']} @ ({action['row']},{action['col']})"


def action_mask(action):
    return blocks.PLACEMENTS[blocks.BY_ACTION[tuple(action[k] for k in ("shape_id", "row", "col"))]][0]


def coordinates(mask):
    return [list(divmod(i, blocks.SIZE)) for i in blocks.cells(mask)]


def bit_text(case, mask):
    return "[" + ",".join(str((mask >> b) & 1) for b in range(case.get("bits", 0))) + "]"


def route_text(start, moves):
    return " → ".join(str(n) for n in [start] + [m["to"] for m in moves])


def path_diagnostics(case, output):
    """Exact state distances on the legal prefix; invalid moves never advance it."""
    verdict = paths.judge(case, output)
    start = {"node": case["start"], "mask": case.get("initial_mask", 0)}
    states = [start] + [{k: t[k] for k in ("node", "mask")} for t in verdict["trace"]]
    references = [paths.shortest({**case, "start": s["node"], "initial_mask": s["mask"]}) for s in states]
    if any(r is None for r in references):
        raise ValueError("This report expects reachable goals throughout the legal prefix")
    distance = [r["length"] for r in references]
    slack = [1 + b - a for a, b in zip(distance, distance[1:])]
    assert all(x >= 0 for x in slack)
    if verdict["execution_pass"]:
        assert sum(slack) == verdict["excess_moves"]
    first = next((i for i, value in enumerate(slack, 1) if value > 0), None)
    illegal = verdict["illegal_move"]
    focus_step = illegal["step"] if illegal else first
    focus_index = focus_step - 1 if focus_step else 0
    current = states[focus_index]
    failed_move = output["path"][focus_index] if focus_step else None
    blocked = [gate for gate in case.get("gates", [])
               if failed_move and gate["edge"] == sorted([failed_move["from"], failed_move["to"]])
               and not current["mask"] & (1 << gate["bit"])]
    recovery = references[focus_index]
    # The suffix starts in the actual current mask; starting there does not flip a switch.
    assert paths.judge({**case, "start": current["node"], "initial_mask": current["mask"]},
                       {"path": recovery["path"]})["pass"]
    result = {"states": states, "distance": distance, "slack": slack,
              "first_optimality_loss_step": first, "focus_step": focus_step,
              "focus_state": current, "focus_move": failed_move, "blocked_gates": blocked,
              "reference": references[0], "recovery_suffix": recovery}
    if illegal:
        opened = case.get("initial_mask", 0)
        for gate in blocked:
            opened |= 1 << gate["bit"]
        # This changes one initial condition and replays the SAME recorded route.
        # It is not a newly observed model response or a causal attribution.
        replay = paths.judge({**case, "initial_mask": opened}, output)
        result["counterfactual_initial_mask"] = opened
        result["counterfactual_same_route"] = {k: replay[k] for k in
            ("execution_pass", "optimal", "shortest_moves", "illegal_move")}
    return result


def block_diagnostics(case, output, verdict, selection):
    initial = blocks.from_grid(case["grid"])
    loss, illegal = verdict["first_irrecoverable_action"], verdict["illegal_action"]
    mode = selection["focus"]
    if mode == "trap":
        assert loss and loss["kind"] == "untileable"
        before, after = blocks.from_grid(loss["before_grid"]), blocks.from_grid(loss["after_grid"])
        action, step = loss["action"], loss["step"]
        assert blocks.apply(before, action) == after
    elif mode == "illegal":
        assert illegal and not loss
        before = blocks.from_grid(verdict["remaining_grid"])
        after, action, step = before, illegal["action"], illegal["step"]
        try:
            blocks.apply(before, action)
        except ValueError:
            pass
        else:
            raise AssertionError("Selected illegal action was legal")
    else:
        assert mode == "stop" and verdict["failure_type"] == "stopped_early" and not loss
        before = after = blocks.from_grid(verdict["remaining_grid"])
        action, step = None, verdict["executed_actions"]
    solve = blocks.solver()
    minimum, ids = solve(before)
    assert minimum is not None
    supported = 0
    for index in blocks.legal_indices(after):
        supported |= blocks.PLACEMENTS[index][0]
    unsupported = after & ~supported
    if mode == "trap":
        assert solve(after)[0] is None
    preferred = selection.get("alternate_first")
    if preferred:
        next_state = blocks.apply(before, preferred)
        count, ids = solve(next_state)
        assert count is not None
        suffix = [preferred] + [blocks.PLACEMENTS[i][1] for i in ids]
    else:
        relevant = action_mask(action) if action else before
        ids = sorted(ids, key=lambda i: ((blocks.PLACEMENTS[i][0] & unsupported).bit_count(),
                                         (blocks.PLACEMENTS[i][0] & relevant).bit_count()), reverse=True)
        suffix = [blocks.PLACEMENTS[i][1] for i in ids]
    assert blocks.judge(blocks.to_grid(before), suffix)["pass"]
    prefix_count = step if mode == "stop" else step - 1
    repaired_plan = output["actions"][:prefix_count] + suffix
    assert blocks.judge(case["grid"], repaired_plan)["pass"]
    illegal_detail = None
    if illegal:
        current = blocks.from_grid(verdict["remaining_grid"])
        overlap = action_mask(illegal["action"]) & ~current
        removed = overlap & initial
        removal_steps = []
        for t in verdict["trace"]:
            intersection = action_mask(t["action"]) & removed
            if intersection:
                removal_steps.append({"step": t["step"], "action": t["action"], "cells": coordinates(intersection)})
        illegal_detail = {**illegal, "initially_empty_cells": coordinates(overlap & ~initial),
                          "previously_removed_cells": coordinates(removed), "removal_steps": removal_steps}
    return {"focus_step": step, "focus_action": action, "before_grid": blocks.to_grid(before),
            "after_grid": blocks.to_grid(after), "unsupported_cells": coordinates(unsupported),
            "minimum_before": minimum, "alternative_suffix": suffix,
            "repaired_full_plan": repaired_plan, "illegal_detail": illegal_detail}


def load_evidence(config):
    suite, compact = read_json(ROOT / config["suite"]), read_json(ROOT / config["summary"])
    cases = {c["id"]: c for c in suite["cases"]}
    records, by_case, sources = {}, defaultdict(list), []
    for source in compact["run_sources"]:
        filename = ROOT / source["path"].replace("\\", "/")
        payload = read_json(filename)
        assert payload["source_commit"] == source["source_commit"]
        sources.append({**source, "path": filename.relative_to(ROOT).as_posix()})
        for record in payload["cases"]:
            key = record["case_id"], record["replicate"]
            assert key not in records
            assert record["response_status"] == "completed" and not record.get("api_error")
            # Recheck every retained answer, not just the chosen illustrations.
            assert evaluate(cases[key[0]], record["raw_output"]) == record["verdict"], key
            records[key] = record
            by_case[key[0]].append(record)
    expected = {(c["id"], r) for c in cases.values() for r in range(1, c["replicates"] + 1)}
    assert records.keys() == expected and len(records) == compact["completed_answers"]
    population = {}
    for condition in CONDITIONS:
        chosen = [r for r in records.values() if r["condition"] == condition]
        groups = {r["case_id"] for r in chosen}
        success = sum(r["verdict"]["pass"] for r in chosen)
        counts = dict(Counter(r["verdict"].get("failure_type") or "success" for r in chosen))
        assert success == compact["summary"][condition]["successes"]
        assert counts == compact["summary"][condition]["failure_counts"]
        population[condition] = {"answers": len(chosen), "instances": len(groups), "successes": success,
            "arrival": sum(r["verdict"].get("execution_pass", r["verdict"].get("solved", False)) for r in chosen),
            "solved_instances": sum(any(r["verdict"]["pass"] for r in by_case[c]) for c in groups),
            "failed_instances": sum(any(not r["verdict"]["pass"] for r in by_case[c]) for c in groups),
            "failure_counts": counts}
    selected = []
    for selection in config["selections"]:
        key = selection["case_id"], selection["replicate"]
        record, case = records[key], cases[key[0]]
        verdict, output = record["verdict"], parse_output(record["raw_output"])[0]
        assert not verdict["pass"]
        is_block = case["condition"].startswith("blocks")
        assert case.get("budget" if is_block else "step_limit") is None
        sequence = "actions" if is_block else "path"
        # Pattern equality uses actions/edges, not rationale or auxiliary reports.
        fields = ("shape_id", "row", "col") if is_block else ("from", "to")
        pattern = lambda obj: [tuple(m[k] for k in fields) for m in obj[sequence]]
        same = [r["replicate"] for r in by_case[key[0]]
                if pattern(parse_output(r["raw_output"])[0]) == pattern(output)]
        details = (block_diagnostics(case, output, verdict, selection) if is_block
                   else path_diagnostics(case, output))
        selected.append({**selection, "case": case, "model_output": output, "verdict": verdict,
                         "case_successes": sum(r["verdict"]["pass"] for r in by_case[key[0]]),
                         "case_samples": len(by_case[key[0]]), "same_plan_replicates": sorted(same),
                         "details": details})
    return {"sources": sources, "population": population, "studies": selected,
            "selection": "Post hoc illustrations; variants of the same instance are not independent cases.",
            "verification": {"replayed_answers": len(records), "selected_answers": len(selected),
                             "distinct_selected_instances": len({s["case_id"] for s in selected})}}


def plot_modules(chinese=False):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.titlesize": 11,
                         "axes.labelsize": 10, "savefig.facecolor": "white"})
    if chinese:
        from matplotlib import font_manager
        installed = {f.name for f in font_manager.fontManager.ttflist}
        choices = [f for f in ("Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "WenQuanYi Zen Hei") if f in installed]
        if not choices:
            raise RuntimeError("Chinese plot labels require a CJK font, e.g. Microsoft YaHei or Noto Sans CJK SC")
        plt.rcParams["font.family"] = choices[0]
    return plt


def draw_board(ax, mask, title, selected=None, marked=(), bad=(), tile_plan=None):
    from matplotlib.patches import Rectangle
    import matplotlib.pyplot as plt
    highlight = action_mask(selected) if selected else 0
    colors, numbers = {}, {}
    if tile_plan:
        palette = plt.get_cmap("tab20")
        for number, action in enumerate(tile_plan, 1):
            for i in blocks.cells(action_mask(action)):
                colors[i], numbers[i] = palette((number - 1) % 20), number
    for r in range(10):
        for c in range(10):
            i = r * 10 + c
            color = colors.get(i, ORANGE if highlight & (1 << i) else DARK if mask & (1 << i) else "white")
            ax.add_patch(Rectangle((c, r), 1, 1, facecolor=color, edgecolor="#d1d8df", linewidth=.7))
            if i in numbers:
                ax.text(c + .5, r + .5, str(numbers[i]), ha="center", va="center", fontsize=8)
    for r, c in marked:
        ax.add_patch(Rectangle((c + .06, r + .06), .88, .88, fill=False, edgecolor=RED, linewidth=2.3))
    for r, c in bad:
        ax.add_patch(Rectangle((c + .05, r + .05), .9, .9, facecolor="#fff0f0", edgecolor=RED, linewidth=1.8))
        ax.plot([c + .2, c + .8], [r + .2, r + .8], color=RED, lw=2)
        ax.plot([c + .2, c + .8], [r + .8, r + .2], color=RED, lw=2)
    ax.set(xlim=(0, 10), ylim=(10, 0), aspect="equal")
    ax.set_xticks([c + .5 for c in range(10)], range(10), fontsize=9)
    ax.set_yticks([r + .5 for r in range(10)], range(10), fontsize=9)
    ax.set_title(title, pad=12)
    if selected:
        ax.set_xlabel(action_text(selected), labelpad=8)


def render_blocks(study, target):
    plt = plot_modules()
    d, v = study["details"], study["verdict"]
    before, after = blocks.from_grid(d["before_grid"]), blocks.from_grid(d["after_grid"])
    suffix = d["alternative_suffix"]
    fig, axes = plt.subplots(1, 4, figsize=(16, 5.3))
    if study["focus"] == "trap":
        draw_board(axes[0], before, f"Before step {d['focus_step']}\n{d['minimum_before']} moves can finish")
        draw_board(axes[1], before, "Luna: legal removal", selected=d["focus_action"])
        draw_board(axes[2], after, "After: completion impossible", marked=d["unsupported_cells"])
        draw_board(axes[3], before, f"Alternative: valid {len(suffix)}-move suffix", selected=suffix[0])
        caption = "Orange = selected removal. Red outline = occupied cell with no legal covering shape."
    elif study["focus"] == "illegal":
        detail = d["illegal_detail"]
        bad = detail["initially_empty_cells"] + detail["previously_removed_cells"]
        draw_board(axes[0], before, f"Before illegal step {d['focus_step']}\n{d['minimum_before']} moves can finish")
        draw_board(axes[1], before, "Luna: overlaps an empty cell", selected=d["focus_action"], bad=bad)
        draw_board(axes[2], after, "Rejected: state unchanged", bad=bad)
        draw_board(axes[3], before, f"Alternative: valid {len(suffix)}-move suffix", selected=suffix[0])
        kind = "previously removed" if detail["previously_removed_cells"] else "empty in the initial board"
        caption = f"Red X = {bad}, {kind}. Invalid moves do not remove any cells."
    elif v["executed_actions"]:
        draw_board(axes[0], blocks.from_grid(study["case"]["grid"]), "Initial board")
        draw_board(axes[1], before, f"After {v['executed_actions']} legal moves\nLuna reports: {v['reported_status']}",
                   marked=coordinates(before))
        draw_board(axes[2], before, f"A valid {len(suffix)}-move completion remains", selected=suffix[0])
        draw_board(axes[3], 0, "After the verified suffix: solved")
        caption = "No action-count cap. Red outline = leftover cells, not an irrecoverable dead end."
    else:
        draw_board(axes[0], before, "Luna outputs zero moves\nand claims no solution")
        draw_board(axes[1], before, f"Counterexample: {len(suffix)} disjoint pieces", tile_plan=suffix)
        draw_board(axes[2], before, "First piece refutes\nthe forced-choice argument", selected=suffix[0])
        draw_board(axes[3], blocks.apply(before, suffix[0]), f"After that piece\n{len(suffix) - 1} more moves clear the board")
        caption = "Numbers in the colored tiling identify the verified removal order. No rotations or extra shapes used."
    fig.suptitle(f"{study['label']} | {study['case_id']} | sample {study['replicate']} | instance success {study['case_successes']}/8",
                 fontsize=14, y=.98)
    fig.text(.5, .035, caption + "  Coordinates: zero-based (row, column).", ha="center", fontsize=10, color=DARK)
    fig.subplots_adjust(left=.028, right=.994, top=.80, bottom=.19, wspace=.16)
    fig.savefig(target, dpi=170)
    plt.close(fig)


def graph_layout(case):
    import networkx as nx
    graph = nx.Graph()
    graph.add_nodes_from(range(len(case["neighbors"])))
    graph.add_edges_from((int(n), m) for n, neighbors in case["neighbors"].items() for m in neighbors)
    # One deterministic display layout for all original and gated conditions.
    pos = nx.kamada_kawai_layout(graph)
    # In this fixed graph, the automatic layout puts nodes 3, 29 and 30 on
    # unrelated edges (0,1), (21,22) and (24,27). Move only their display y
    # coordinates so those edges cannot be misread as passing through a node.
    for node, dy in {3: .15, 29: -.12, 30: -.20}.items():
        pos[node][1] += dy
    return graph, pos


def draw_graph(ax, graph, pos, case, study):
    """One real pre-decision snapshot; no future route is drawn as already executed."""
    import networkx as nx
    from matplotlib.lines import Line2D
    d, v = study["details"], study["verdict"]
    current, mask = d["focus_state"]["node"], d["focus_state"]["mask"]
    closed = {tuple(g["edge"]) for g in case.get("gates", []) if not mask & (1 << g["bit"])}
    edges = list(graph.edges())
    nx.draw_networkx_edges(graph, pos, ax=ax, edgelist=[e for e in edges if tuple(sorted(e)) not in closed],
                           edge_color="#c3cbd4", width=1.5)
    nx.draw_networkx_edges(graph, pos, ax=ax, edgelist=[e for e in edges if tuple(sorted(e)) in closed],
                           edge_color="#db9299", style="dashed", width=1.8)
    special = {case["start"], case["goal"]} | {s["node"] for s in case.get("switches", [])}
    ordinary = [n for n in graph.nodes if n not in special]
    nx.draw_networkx_nodes(graph, pos, ax=ax, nodelist=ordinary, node_color="#f8fafc", edgecolors="#a7b5c3",
                           node_size=330, linewidths=1.2)
    nx.draw_networkx_labels(graph, pos, ax=ax, labels={n: str(n) for n in ordinary}, font_size=9, font_color=DARK)
    for node, label, color in [(case["start"], "S", BLUE), (case["goal"], "G", GREEN)]:
        nx.draw_networkx_nodes(graph, pos, ax=ax, nodelist=[node], node_color=color, edgecolors="white",
                               node_size=720, linewidths=1.5)
        nx.draw_networkx_labels(graph, pos, ax=ax, labels={node: f"{label}\n{node}"}, font_size=10,
                                font_color="white", font_weight="bold")
    for switch in case.get("switches", []):
        node, bit = switch["node"], switch["bit"]
        nx.draw_networkx_nodes(graph, pos, ax=ax, nodelist=[node], node_color="#eee7f5", node_shape="s",
                               edgecolors="#8761a0", node_size=630, linewidths=1.7)
        nx.draw_networkx_labels(graph, pos, ax=ax, labels={node: f"K{bit}\n{node}"}, font_size=9,
                                font_color=DARK, font_weight="bold")
    nx.draw_networkx_nodes(graph, pos, ax=ax, nodelist=[current], node_color="none",
                           edgecolors=DARK, node_size=1030, linewidths=2.1)
    moves = study["model_output"]["path"][:d["focus_step"]]
    for step, move in enumerate(moves, 1):
        is_focus = step == d["focus_step"]
        color = RED if is_focus and v["illegal_move"] else ORANGE if is_focus else BLUE
        nx.draw_networkx_edges(graph, pos, ax=ax, edgelist=[(move["from"], move["to"])], arrows=True,
                               arrowstyle="-|>", arrowsize=18, node_size=500, width=3, edge_color=color,
                               style="dashed" if is_focus and v["illegal_move"] else "solid", connectionstyle="arc3,rad=0.1")
    ax.set_title(f"全图：第 {d['focus_step']} 步之前，当前位置 {current}", loc="left", fontsize=12, weight="bold", color=DARK)
    legend = [Line2D([0], [0], color="#a9b5c2", lw=2, label="此刻可通行"),
              Line2D([0], [0], color=BLUE, lw=3, label="此前已走过")]
    if case.get("bits"):
        legend.insert(1, Line2D([0], [0], color="#db9299", lw=2, ls="--", label="此刻关闭"))
    ax.legend(handles=legend, loc="lower center", bbox_to_anchor=(.5, -.075), ncol=3, frameon=False, fontsize=9)
    ax.margins(.17)
    ax.axis("off")


def draw_decision(ax, study):
    """Plain-language cause and an exact alternative at the SAME pre-action state."""
    from matplotlib.patches import FancyBboxPatch
    case, d, v = study["case"], study["details"], study["verdict"]
    ax.axis("off")
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.text(0, .98, "这一动作为什么失败", weight="bold", fontsize=16, color=DARK, va="top")
    state_label = f"开关 {bit_text(case, d['focus_state']['mask'])}" if case.get("bits") else "无开关"
    ax.text(0, .89, f"当前节点 {d['focus_state']['node']}  |  {state_label}",
            fontsize=12, color=DARK)
    for i, switch in enumerate(case.get("switches", [])):
        opened = bool(d["focus_state"]["mask"] & (1 << switch["bit"]))
        y = .805 - i * .095
        ax.add_patch(FancyBboxPatch((0, y - .031), .96, .07, boxstyle="round,pad=.012", linewidth=0,
                                   facecolor="#e3f1e9" if opened else "#fbe7e9"))
        ax.text(.025, y, f"开关 {switch['bit']} = {1 if opened else 0}：{'通行' if opened else '关闭'}"
                f"；进入节点 {switch['node']} 可翻转", va="center", fontsize=11, color=GREEN if opened else RED)
    move = d["focus_move"]
    y = .55 if case.get("bits") else .71
    ax.text(0, y, f"第 {d['focus_step']} 步：{move['from']} → {move['to']}", fontsize=18, weight="bold",
            color=RED if v["illegal_move"] else ORANGE)
    if v["illegal_move"]:
        bits = "/".join(str(g["bit"]) for g in d["blocked_gates"])
        reason = f"这条边要求开关 {bits} = 1，当前却为 0。\n因此动作被拒绝，仍停在 {move['from']}；\n没有到达 {move['to']}，也不会触发到达后的翻转。"
    else:
        i = d["focus_step"]
        reason = (f"动作合法，但走完后还需 {d['distance'][i]} 步。\n"
                  f"加上这一步共 {1 + d['distance'][i]} 步，"
                  f"而当前最少只需 {d['distance'][i - 1]} 步。\n"
                  f"这一次选择多花了 {d['slack'][i - 1]} 步。")
    ax.text(0, y - .07, reason, fontsize=11.5, linespacing=1.8, color=DARK, va="top")
    alternate = d["recovery_suffix"]["path"][0]
    ax.text(0, .205, "同一状态下有可行替代", fontsize=13, weight="bold", color=GREEN)
    ax.text(0, .137, f"下一步可走 {alternate['from']} → {alternate['to']}，再接可行后缀。", fontsize=11.5, color=DARK)
    spent, remaining = d["focus_step"] - 1, d["recovery_suffix"]["length"]
    ax.text(0, .081, f"此前 {spent} 步 + 剩余最少 {remaining} 步 = 共 {spent + remaining} 步。", fontsize=11, color=DARK)
    ax.text(0, .025, "下方绿色给出重新从起点规划的全局最短解。", fontsize=10, color="#566779")


def draw_timeline(ax, case, states, title, color, distances=None, focus_step=None, illegal_move=None, max_nodes=10):
    ax.axis("off")
    ax.set_xlim(-1.55, max_nodes - .4)
    ax.set_ylim(-1.1, 1.25)
    ax.text(-1.5, 1.0, title, fontsize=11, weight="bold", color=DARK, va="center")
    for i, state in enumerate(states):
        x, y = i, .16
        ax.scatter([x], [y], s=380, facecolors="white", edgecolors=color, linewidths=1.8, zorder=3)
        ax.text(x, y, str(state["node"]), ha="center", va="center", fontsize=10, color=DARK)
        aux = bit_text(case, state["mask"]) if case.get("bits") else ""
        if distances is not None:
            aux += ("\n" if aux else "") + f"d={distances[i]}"
        ax.text(x, -.4, aux, ha="center", va="top", fontsize=9, color=DARK)
        if i:
            edge_color = ORANGE if i == focus_step else color
            ax.annotate("", xy=(x - .22, y), xytext=(x - .78, y),
                        arrowprops={"arrowstyle": "-|>", "color": edge_color, "lw": 2})
            ax.text(x - .5, .57, str(i), ha="center", fontsize=8, color=edge_color)
    if illegal_move:
        x = len(states)
        ax.annotate("", xy=(x - .23, .16), xytext=(x - .78, .16),
                    arrowprops={"arrowstyle": "-|>", "color": RED, "lw": 2, "linestyle": "dashed"})
        ax.text(x - .5, .64, f"第{x}步被拒绝", color=RED, fontsize=8, ha="center")
        ax.scatter([x], [.16], s=380, facecolors="#fff0f0", edgecolors=RED, linewidths=1.8)
        ax.text(x, .16, str(illegal_move["move"]["to"]), ha="center", va="center", color=RED, fontsize=10)
        ax.text(x, -.43, "未到达", ha="center", va="top", fontsize=8, color=RED)


def render_path(study, target, graph, pos):
    plt = plot_modules(chinese=True)
    case, d, v = study["case"], study["details"], study["verdict"]
    illegal, reference = v["illegal_move"], d["reference"]
    fig = plt.figure(figsize=(14, 10.7))
    gs = fig.add_gridspec(3, 2, height_ratios=[4.4, 1, 1], width_ratios=[1.15, 1], hspace=.40, wspace=.13)
    draw_graph(fig.add_subplot(gs[0, 0]), graph, pos, case, study)
    draw_decision(fig.add_subplot(gs[0, 1]), study)
    width = max(len(d["states"]) + bool(illegal), len(reference["states"]))
    draw_timeline(fig.add_subplot(gs[1, :]), case, d["states"], "模型实际执行", BLUE, d["distance"],
                  d["first_optimality_loss_step"], illegal, width)
    draw_timeline(fig.add_subplot(gs[2, :]), case, reference["states"], f"从起点最短：{reference['length']} 步", GREEN,
                  list(range(reference["length"], -1, -1)), max_nodes=width)
    outcome = f"第 {illegal['step']} 步穿门失败" if illegal else f"合法到达，但 {v['attempted_moves']} 步 > 最短 {reference['length']} 步"
    fig.suptitle(f"{study['label']}  {outcome}", fontsize=18, weight="bold", y=.987)
    initial_label = f"初始开关 {bit_text(case, case.get('initial_mask', 0))}" if case.get("bits") else "原图，无开关"
    fig.text(.5, .947, f"S 起点 = {case['start']}     G 终点 = {case['goal']}     "
             f"{initial_label}     本题成功 {study['case_successes']}/8", ha="center", fontsize=12, color=DARK)
    fig.text(.5, .913, f"{study['case_id']}  ·  第 {study['replicate']} 次采样", ha="center", fontsize=10, color="#566779")
    map_legend = ("全图只显示关键动作前的状态：灰实线可走，红虚线关闭；K0/K1 方框是开关节点，黑圈是当前位置。"
                  if case.get("bits") else "全图只显示关键动作前的状态：S 是起点，G 是终点，黑圈是当前位置；所有道路均可通行。")
    fig.text(.5, .041, map_legend,
             ha="center", fontsize=10, color=DARK)
    fig.text(.5, .018, "下方路线：蓝色已合法执行，橙色首次多走步数，红色虚线为非法尝试。d = 此状态到目标的最少步数；每条边均算 1 步。",
             ha="center", fontsize=9, color=DARK)
    fig.subplots_adjust(left=.04, right=.97, top=.861, bottom=.091)
    fig.savefig(target, dpi=170)
    plt.close(fig)


def facts_for(study):
    case, d, v = study["case"], study["details"], study["verdict"]
    facts = [f"实例 {study['case_id']}，第 {study['replicate']} 次采样；该实例成功 {study['case_successes']}/{study['case_samples']}。"
             f"完全相同的动作／边序列出现在采样 {study['same_plan_replicates']}（只在这八次中比较）。"]
    if case["condition"].startswith("blocks"):
        facts.append(f"模型提交 {v['attempted_actions']} 个动作，实际合法执行 {v['executed_actions']} 个，"
                     f"最终余 {v['remaining_cells']} 格；报告 {v['reported_status']}，最终裁判类型 {v['failure_type']}。")
        if study["focus"] == "trap":
            facts.append(f"首次不可恢复：第 {d['focus_step']} 步 {action_text(d['focus_action'])}。"
                         f"移除后无法被任何合法形状覆盖的格为 {d['unsupported_cells']}；精确分解器确认无解，不涉及剩余步数限制。")
        if d["illegal_detail"]:
            bad = d["illegal_detail"]
            facts.append(f"非法动作：第 {bad['step']} 步 {action_text(bad['action'])}；"
                         f"重叠初始空格 {bad['initially_empty_cells']}，重叠已移除格 {bad['previously_removed_cells']}。")
            for removal in bad["removal_steps"]:
                facts.append(f"重用格 {removal['cells']} 早在第 {removal['step']} 步 {action_text(removal['action'])} 被移除。")
        facts.append(f"从图示替代方案的起始状态，最少 {d['minimum_before']} 步可完成；"
                     f"以下完整合法后缀共 {len(d['alternative_suffix'])} 步，连同保留的模型前缀已重放到空棋盘： "
                     + " → ".join(action_text(a) for a in d["alternative_suffix"]) + "。")
    else:
        facts.append(f"初态：节点 {case['start']}、开关 {bit_text(case, case.get('initial_mask', 0))}；目标节点 {case['goal']}。"
                     + ("开关位置：" + "；".join(f"进入 {s['node']} 翻转第 {s['bit']} 位" for s in case['switches']) + "。"
                        if case.get("bits") else "原图没有开关。"))
        for bit in range(case.get("bits", 0)):
            controlled = "，".join(f"{g['edge'][0]}↔{g['edge'][1]}" for g in case["gates"] if g["bit"] == bit)
            facts.append(f"开关 {bit} 控制的无向道路：{controlled}。该位为 0 时这些道路关闭，为 1 时允许通过。")
        facts.append("模型提交路线：" + route_text(case["start"], study["model_output"]["path"]) + "。"
                     + ("首次非法动作之后的后缀没有执行，图中也不画成已走过的路线。" if v["illegal_move"] else "全程合法到达。"))
        facts.append(f"一条精确最短路线（{d['reference']['length']} 步）：" + route_text(case["start"], d["reference"]["path"]) + "。")
        first = d["first_optimality_loss_step"]
        if first:
            move = study["model_output"]["path"][first - 1]
            facts.append(f"首次损失最短性在合法第 {first} 步 {move['from']}→{move['to']}："
                         f"精确剩余距离由 {d['distance'][first - 1]} 变为 {d['distance'][first]}，"
                         f"该步付出 1 + {d['distance'][first]} − {d['distance'][first - 1]} = {d['slack'][first - 1]} 个额外步数。"
                         "此判据允许多条等长最短路线。")
        if v["illegal_move"]:
            bad, state = v["illegal_move"], d["focus_state"]
            facts.append(f"首次非法：第 {bad['step']} 步 {bad['move']['from']}→{bad['move']['to']}，"
                         f"真实前态 ({state['node']}, {bit_text(case, state['mask'])})，"
                         f"关闭的门要求位 {[g['bit'] for g in d['blocked_gates']]} 为 1。")
            facts.append(f"从上述真实前态仍可恢复：{route_text(state['node'], d['recovery_suffix']['path'])}"
                         f"（最少再走 {d['recovery_suffix']['length']} 步；这不是从最初起点的最短总长）。")
        else:
            facts.append(f"总长 {v['attempted_moves']}，最短 {v['shortest_moves']}，多走 {v['excess_moves']} 步。")
        if case.get("bits"):
            facts.append(f"已合法执行前缀的开关报告正确 {v['switch_reports_correct']}/{v['switch_reports_evaluated']}。"
                         "非法动作后声称的到达状态不计为真实状态。")
        if case["condition"] == "path_gates1":
            cf = d["counterfactual_same_route"]
            facts.append(f"离线只把初始位改为 {bit_text(case, d['counterfactual_initial_mask'])}，重放同一份答案："
                         f"合法到达={cf['execution_pass']}，最短={cf['optimal']}，该初态最短长度={cf['shortest_moves']}。"
                         "这是裁判反事实，不是新增模型测试。")
    return facts


INTRO = [
    "最值得继续研究的是两类失败：积木中合法移除后再也无法清空，以及路径中正确报告开关值却仍走关闭道路。"
    "它们直接指向‘能否预测动作后果，并用当前状态约束行动’。另外保留重复移除、漏收尾和普通绕路，区分不同原因。",
    "本报告用于前期寻找反例、明确后续状态模型的改进目标。当前一次性输出完整计划的协议保持有效；"
    "可观察的错误已经核验，隐藏的内部原因和新方法收益仍待实验检验。",
    "共选 18 份失败答案，覆盖五种条件、15 个不同实例：8 块 5 例／4 棋盘，12 块 4 例／4 棋盘，原图 3 例／3 起终点对，"
    "单开关 2 例／同一实例，双开关 4 例／3 个实例。它们由完整结果事后挑选，用于解释错误，不以这 18 例重算总体成功率。"
    "所有路径共享同一张 32 节点图；开关组每种条件只有四个独立门布局，两个初始状态配对也不算独立布局。",
    "64 个实例各八次，共 512 个完整 Luna 答案，本次全部重放一致，未新增 API 调用。"
    "积木没有动作数上限，8／12 是生成块数；路径要求最短，并另报合法到达。这里的案例均不是截断答案。",
    "积木坐标均从零开始，记为 (row, column)，落点是形状包围盒左上角；形状固定，不额外旋转。"
    "图中橙色是选中的移除，红框是残余关键格，红叉是企图使用的空格。",
    "路径图的 S 是起点、G 是终点，K0／K1 方框是开关节点，黑圈是当前节点。全图显示关键动作之前的真实状态："
    "灰实线表示此刻能走，红虚线表示此刻关闭。蓝箭头是此前已执行动作；红箭头是被拒绝的尝试，橙箭头是合法但首次浪费步数的选择。"
    "右侧直接解释原因，下方绿色路线是从原始起点出发的最短解。每条边均算一步，图上距离不代表代价。",
]

SWITCH_RULES = (
    "每个开关控制一组道路：0＝该组关闭，1＝该组可走。进入 K0／K1 所在节点，会翻转对应位（0↔1）；再次进入也会翻转。"
    "两个开关独立，两组路可以同时开或同时关。先检查出发时的门，能合法走过去后才触发到达节点的开关；仅以开关节点为起点不触发。"
    "路线下的 [0,1] 表示开关0关、开关1开；d 表示这个节点和开关组合下，到目标最少还需几步。"
)


def resolution_for(study):
    d, v, case = study["details"], study["verdict"], study["case"]
    if case["condition"].startswith("path"):
        return f"从起点可走：{route_text(case['start'], d['reference']['path'])}，共 {d['reference']['length']} 步。已验证每一步合法且总长最短。"
    first, count = action_text(d["alternative_suffix"][0]), len(d["alternative_suffix"])
    if study["focus"] == "stop":
        return f"在当前棋盘继续，先执行 {first}，共补 {count} 个动作即可清空。"
    return f"保留前 {d['focus_step'] - 1} 个动作，把关键一步改为 {first}，再接 {count - 1} 步可清空。完整后缀已重放，见折叠的核验细节。"

RECOMMENDATIONS = [
    ("优先把双开关依赖与必要回访作为下一批强反例方向。",
     "P2-1／P2-2 的 0/8 发生在七步可解实例上，且已执行状态报告全对，特别适合检验状态信息是否真正参与动作前提和子目标协调。"
     "先扩大独立门布局、变化开关顺序和回访是否会关门，并匹配最短解长度；这样比直接把 32 节点扩得更大，更接近项目要解决的机制。"
     "P2-3 再提供‘同一节点、不同状态’的回访诊断。"),
    ("积木保留两个互补层次：可追溯的一步陷阱与更难的残余约束。",
     "B8-4、B12-1、B12-2 可直接展示：当下合法的移除会毁掉完整解，且有从同一状态出发的已验证替代后缀。"
     "全部 30 个观测到的首次死局已出现无合法覆盖的格，适合先测一步转移加残余检查。另从离线候选中选择每格仍有局部覆盖但整体无解的状态，"
     "作为单列诊断集，在调用前确定；不能把当前这些孤格案例描述为已经证明需要深层搜索。"),
    ("给状态接口明确的输入与输出目标，再决定最简学习组件。",
     "B8-1 对应观测与落点一致性，B8-3 对应动作后的占用更新，B12-2 对应候选动作后果和可完成性，P2-1 对应带状态的动作前提。"
     "按这些事实设计占用预测、转移预测、合法性和目标判断小测，再检验信息进入动作评价后是否提高完整任务成功率；"
     "无需预先承诺复杂网络，也无需把原有直接 API 动机实验改写为另一种任务。"),
    ("把容易修复的错误作为对照，集中展示最有价值的剩余失败。",
     "B12-4 仅差最后一个双格，终态自检可能足以解决；B8-5 的强制配对理由被一个合法三格反驳，完整八步分解证明有解。"
     "P0-2 与 P2-4 共享 4→5→6→7 的普通绕路，说明开关题的部分失分也来自静态路线比较。"
     "保留这些对照能帮助说明状态模块究竟补足了哪一环，而不是用总成功率把不同改进混为一谈。"),
    ("以匹配信息和预算的诊断逐步连接到方法实验。",
     "先比较等价坐标表示、显式状态自报、真实残余状态继续规划和候选后果预测；所有条件保持同一模型、样本、采样次数和可验证预算。"
     "真实状态只在约定的观察时刻提供；推理中查询未执行动作的精确后果属于工具辅助条件，应单独报告，不能混成无工具基线。"
     "本报告已证明替代方案存在，尚未实际测过这些干预或训练增益。"),
    ("结论按模型与设置落地，同时保留跨模型对照。",
     "Luna 的三个 0/8 是本批稳定失败事实，不代表任务无解或所有模型都失败。已完成的 Flash 对照在同套件每题至少成功一次、积木完整答案全对，"
     "并使用了明显更多推理 token。利用这一更强基线进一步寻找依赖结构、固定预算可靠性与状态增强收益，比扩大无法完成的泛化表述更能维护本项目的论证。"),
]


def write_reports(evidence, config, output):
    report_path = ROOT / config["report"]
    markdown_lines = ["# Luna 五种条件的错误案例与图解", "", *[p + "\n" for p in INTRO[:4]]]
    gallery_parts = ["<h1>Luna：18 个错误案例与精确替代方案</h1>"]
    gallery_parts += [f"<p>{html.escape(p)}</p>" for p in INTRO[:4]]
    markdown_lines += ["## 先读懂开关与图例", "", SWITCH_RULES, "", INTRO[-1], "", INTRO[-2], ""]
    gallery_parts += ["<h2>先读懂开关与图例</h2>", f"<p>{html.escape(SWITCH_RULES)}</p>",
                      f"<p>{html.escape(INTRO[-1])}</p>"]
    header = ["条件", "任务成功／答案", "合法到达／答案", "pass@8", "出现失败的实例", "最终失败类型"]
    rows = []
    for condition, name in CONDITIONS.items():
        p = evidence["population"][condition]
        rows.append([name, f"{p['successes']}/{p['answers']}",
                     f"{p['arrival']}/{p['answers']}" if condition.startswith("path") else "—",
                     f"{p['solved_instances']}/{p['instances']}", f"{p['failed_instances']}/{p['instances']}",
                     "；".join(f"{ {'illegal_action': '非法移除', 'dead_end': '死局', 'stopped_early': '提前停止', 'non_shortest': '非最短', 'illegal_move': '非法移动'}.get(k, k)}: {v}"
                              for k, v in p["failure_counts"].items() if k != "success")])
    markdown_lines += ["## 全量结果背景", "", "| " + " | ".join(header) + " |", "| " + " | ".join(["---"] * len(header)) + " |"]
    markdown_lines += ["| " + " | ".join(row) + " |" for row in rows]
    table = "<table><thead><tr>" + "".join(f"<th>{html.escape(h)}</th>" for h in header) + "</tr></thead><tbody>"
    table += "".join("<tr>" + "".join(f"<td>{html.escape(c)}</td>" for c in row) + "</tr>" for row in rows) + "</tbody></table>"
    gallery_parts.append(table)
    markdown_lines += ["", "积木最终标签与较早的死局事件会重叠：30 份答案、16 个棋盘在合法动作后已无解，其中 13 份后来又提交非法动作。"
                       "63 次非法动作中，56 次碰到初始空格，7 次碰到此前移除格。单开关全部四次失败来自同一个实例。", "",
                       "固定形状词表：`0=11/10`，`1=10/11`，`2=11/01`，`3=01/11`，`4=1/1`，`5=11`，`6=1/1/1`，`7=111`；`/` 换行。", ""]
    gallery_parts.append("<nav>" + " ".join(f'<a href="#{s["label"]}">{s["label"]}</a>' for s in evidence["studies"]) + "</nav>")
    previous = None
    for study in evidence["studies"]:
        condition = study["case"]["condition"]
        if condition != previous:
            markdown_lines += [f"## {CONDITIONS[condition]}", ""]
            gallery_parts.append(f"<h2>{CONDITIONS[condition]}</h2>")
            previous = condition
        title = study["label"] + "　" + study["title"]
        filename = study["label"] + ".png"
        facts = facts_for(study)
        resolution = resolution_for(study)
        markdown_lines += [f"### {title}", "", f"`{study['case_id']}` · 第 {study['replicate']} 次采样 · 本题成功 {study['case_successes']}/8", "",
                           f"![{title}](../runs/luna_case_studies/{filename})", "",
                           "**错在哪里：** " + study["observation"], "",
                           "**原因与判断：** " + study["interpretation"], "",
                           "**怎样能成功：** " + resolution, "",
                           "**下一步测什么：** " + study["next_probe"], "",
                           "<details>", "<summary>核验细节：完整后缀、状态与来源采样</summary>", ""]
        markdown_lines += ["- " + fact for fact in facts]
        markdown_lines += ["", "</details>", ""]
        gallery_parts += [f'<article id="{study["label"]}"><h3>{html.escape(title)}</h3>',
                          f"<p class=meta>{study['case_id']} · 第 {study['replicate']} 次采样 · 本题成功 {study['case_successes']}/8</p>",
                          f'<a href="{filename}" target="_blank"><img src="{filename}" alt="{html.escape(title)}" loading="lazy"></a>',
                          f"<p><strong>错在哪里：</strong>{html.escape(study['observation'])}</p>",
                          f"<p><strong>原因与判断：</strong>{html.escape(study['interpretation'])}</p>",
                          f"<p><strong>怎样能成功：</strong>{html.escape(resolution)}</p>",
                          f"<p><strong>下一步测什么：</strong>{html.escape(study['next_probe'])}</p>",
                          "<details><summary>核验细节：完整后缀、状态与来源采样</summary><ul>"
                          + "".join(f"<li>{html.escape(f)}</li>" for f in facts) + "</ul></details></article>"]
    markdown_lines += ["## 对任务与实验设置的建议", ""]
    gallery_parts.append("<h2>对任务与实验设置的建议</h2>")
    for title, body in RECOMMENDATIONS:
        markdown_lines += [f"**{title}** {body}", ""]
        gallery_parts.append(f"<p><strong>{html.escape(title)}</strong>{html.escape(body)}</p>")
    markdown_lines += ["## 数据来源与复现", "",
        "正式调用设置：[pilot.json](../configs/pilot.json)。本报告的事后选例：[luna_case_studies.json](../configs/luna_case_studies.json)。"
        "完整逐题统计：[summary.json](summary.json)。跨模型已完成结果：[DeepSeek Flash 汇总](deepseek_flash.json)。", "",
        "全部 512 个答案的裁判结果与原始保留记录一致；18 个案例的替代积木计划或最短路均由精确裁判重放通过。"
        "路径最短性使用真实 `(节点, 开关位)` 的剩余距离：合法一步的多余代价为 `1 + d(after) - d(before)`，"
        "正值才表示偏离所有最短解；非法步不更新节点与开关。案例 JSON 保存模型计划、合法前缀、残余状态、反事实完整后缀和逐步距离。", ""]
    for source in evidence["sources"]:
        markdown_lines.append(f"- `{source['path']}`；调用来源提交 `{source['source_commit']}`；{source['calls']} 个答案。")
    markdown_lines += ["", "从仓库根目录运行（仅离线分析，依赖 matplotlib、networkx）：", "", "```powershell",
        "python experiments/gcml_counterexamples/src/case_studies.py", "```", "",
        "生成图、`index.html` 图文册和 `evidence.json` 均在忽略目录 `experiments/gcml_counterexamples/runs/luna_case_studies/`；"
        "源代码、选例与本摘要保留在 Git 中。路径图用中文说明，渲染需要微软雅黑、黑体或 Noto Sans CJK SC 等中文字体；点击图可看原尺寸。"
        "报告中的下一步诊断均为建议，本次未执行。", ""]
    report_path.write_text("\n".join(markdown_lines), encoding="utf-8")
    style = """body{font-family:'Segoe UI','Microsoft YaHei',sans-serif;background:#f5f7fa;color:#25354a;margin:0;line-height:1.8}
main{max-width:1320px;margin:auto;padding:32px}h1{font-size:30px}h2{margin-top:46px}h3{font-size:22px}
article{background:white;padding:26px;margin:24px 0;border:1px solid #dce3e9;border-radius:12px;scroll-margin-top:20px}
details{background:#f4f7fa;padding:12px 18px;border-radius:8px;font-size:14px}summary{cursor:pointer;color:#2467a4}.meta{color:#617286;font-size:14px}
img{width:100%;height:auto}li{margin:8px 0}table{border-collapse:collapse;width:100%;font-size:14px;background:white}
th,td{border:1px solid #dce3e9;padding:10px;text-align:left}nav{padding:18px;background:#e8eff6;line-height:2.4}
a{color:#2467a4}nav a{margin-right:16px}@media print{article{break-before:page}main{padding:0}body{background:white}}
"""
    (output / "index.html").write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1"><title>Luna 错误案例图文册</title>'
        f"<style>{style}</style><main>" + "\n".join(gallery_parts) + "</main></html>", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="experiments/gcml_counterexamples/configs/luna_case_studies.json")
    args = parser.parse_args()
    config = read_json(ROOT / args.config)
    evidence = load_evidence(config)
    print(json.dumps(evidence["verification"], ensure_ascii=False), flush=True)
    output = ROOT / config["output"]
    output.mkdir(parents=True, exist_ok=True)
    (output / "evidence.json").write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    graph, pos = graph_layout(next(s["case"] for s in evidence["studies"] if s["case"]["condition"].startswith("path")))
    for study in evidence["studies"]:
        target = output / (study["label"] + ".png")
        if study["case"]["condition"].startswith("blocks"):
            render_blocks(study, target)
        else:
            render_path(study, target, graph, pos)
        print(f"Rendered {study['label']}", flush=True)
    write_reports(evidence, config, output)
    print(f"Report: {config['report']}\nGallery: {output / 'index.html'}", flush=True)


if __name__ == "__main__":
    main()
