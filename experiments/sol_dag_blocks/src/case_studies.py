"""Three compact post-hoc cases; render only retained outputs and verified witnesses."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import networkx as nx
from tasks import ROOT, STUDY, TASKS
from run import evaluate

DARK, RED, ORANGE, BLUE, GREEN = "#25354a", "#c83c4b", "#e47742", "#2467a4", "#168064"
plt.rcParams.update({"font.family": "Microsoft YaHei", "axes.titlesize": 12,
                     "savefig.facecolor": "white", "axes.unicode_minus": False})


def tile(task, action):
    return task.by_action[tuple(action[k] for k in ("shape_id", "row", "col"))]


def board(ax, task, mask, title, highlight=0, bad=0, pieces=None):
    colors, numbers = {}, {}
    for n, action in enumerate(pieces or [], 1):
        for i in range(100):
            if tile(task, action) >> i & 1:
                assert i not in colors
                colors[i], numbers[i] = plt.get_cmap("tab20")((n-1) % 20), n
    for i in range(100):
        r, c = divmod(i, 10)
        fill = colors.get(i, ORANGE if highlight >> i & 1 else DARK if mask >> i & 1 else "white")
        ax.add_patch(Rectangle((c, r), 1, 1, facecolor=fill, edgecolor="#d1d8df", lw=.7))
        if i in numbers:
            ax.text(c+.5, r+.5, str(numbers[i]), ha="center", va="center", fontsize=9)
        if bad >> i & 1:
            ax.add_patch(Rectangle((c+.03, r+.03), .94, .94, facecolor="#fff0f0", edgecolor=RED, lw=2))
            ax.plot([c+.2,c+.8], [r+.2,r+.8], color=RED, lw=2)
            ax.plot([c+.2,c+.8], [r+.8,r+.2], color=RED, lw=2)
    ax.set(xlim=(0,10), ylim=(10,0), aspect="equal", title=title)
    ax.set_xticks([x+.5 for x in range(10)], range(10), fontsize=8)
    ax.set_yticks([x+.5 for x in range(10)], range(10), fontsize=8)


def save(fig, name):
    target = STUDY / "results/figures" / name
    target.parent.mkdir(exist_ok=True)
    fig.savefig(target, dpi=155)
    plt.close(fig)


def main():
    run = json.loads((STUDY / "runs/sol_medium_recovery/run.json").read_text(encoding="utf-8"))
    suite = json.loads((ROOT / run["suite_path"]).read_text(encoding="utf-8"))
    cases = {c["id"]: c for c in suite["cases"]}
    records = {(r["case_id"], r["replicate"]): r for r in run["cases"]}

    def selected(cid, rep):
        case, record = cases[cid], records[cid, rep]
        assert evaluate(case, record["raw_output"]) == record["verdict"]
        return case, record, TASKS[case["condition"]]

    case, record, task = selected("blocks12_13", 1)
    assert record["verdict"]["attempted_actions"] == 0
    mask = task.from_grid(case["grid"])
    after = mask
    for action in case["construction_reference"]:
        after = task.apply(after, action, mask)
    assert after == 0
    fig, axes = plt.subplots(1,4, figsize=(14,4.5))
    board(axes[0],task,mask,"初始棋盘：40 格")
    board(axes[1],task,mask,"模型提交空计划\n棋盘保持不变")
    board(axes[2],task,mask,"离线构造解：12 块\n数字表示移除顺序",pieces=case["construction_reference"])
    board(axes[3],task,after,"12 步合法重放后\n棋盘清空")
    fig.suptitle("C1 · blocks12_13 · 第 1 次采样 · 本题 0/8", fontsize=16)
    fig.text(.5,.03,"空计划不等于证明无解；右侧是离线见证，未提供给模型。所有坐标从 0 开始。",ha="center",fontsize=11)
    fig.tight_layout(rect=(0,.08,1,.9))
    save(fig,"C1_empty_plan.png")

    case, record, task = selected("blocks8_00",6)
    v = record["verdict"]
    assert v["state_reports_correct"] == v["state_reports_evaluated"] == 2
    first, second = v["trace"]
    before = task.from_grid(first["remaining_grid"])
    current = task.from_grid(second["remaining_grid"])
    reported = task.from_grid(second["action"]["board_after"])
    illegal = v["illegal_action"]["action"]
    bad = tile(task,illegal) & ~current
    assert bad == 1 << (4*10+6) and task.apply(before,second["action"]) == current
    fig,axes=plt.subplots(1,4,figsize=(14,4.5))
    board(axes[0],task,before,"第 2 步移除 L 三格\n橙色包含 (4,6)",highlight=tile(task,second["action"]))
    board(axes[1],task,reported,"第 2 步后模型自报\n与真实棋盘完全一致")
    board(axes[2],task,current,"第 3 步尝试横向三格\n(4,6) 已被移除",highlight=tile(task,illegal),bad=bad)
    board(axes[3],task,current,"裁判拒绝第 3 步\n仍余 20 格",bad=bad)
    fig.suptitle("C2 · blocks8_00 · 第 6 次采样 · 状态报对后仍重复移除",fontsize=16)
    fig.text(.5,.03,"第 2 步：shape 0 @ (3,6)；第 3 步：shape 5 @ (4,4)。红叉标记不能再次移除的空格。",ha="center",fontsize=11)
    fig.tight_layout(rect=(0,.08,1,.9))
    save(fig,"C2_repeated_removal.png")

    case,record,task=selected("path_dag_07",5)
    graph=nx.DiGraph()
    graph.add_nodes_from(range(32))
    graph.add_edges_from((int(a),b) for a,vs in case["neighbors"].items() for b in vs)
    route=task.shortest(case)
    assert route==case["reference"]["path"] and task.judge(case,{"path":route,"final_node":case["goal"]})["pass"]
    assert not graph.has_edge(8,17) and graph.has_edge(8,7) and graph.has_edge(7,17)
    pos=nx.spring_layout(graph.to_undirected(),seed=17,k=.55,iterations=180)
    fig=plt.figure(figsize=(13,7))
    ax=fig.add_axes((.025,.19,.57,.7))
    nx.draw_networkx_edges(graph,pos,ax=ax,edge_color="#c8d1db",arrows=True,arrowsize=12,node_size=380,width=.9)
    nx.draw_networkx_edges(graph,pos,ax=ax,edgelist=[(a["from"],a["to"]) for a in route],edge_color=GREEN,width=2.8,arrowsize=20,node_size=380)
    nx.draw_networkx_edges(graph,pos,ax=ax,edgelist=[(29,9),(9,8)],edge_color=BLUE,width=3,arrowsize=21,node_size=380)
    nx.draw_networkx_edges(graph,pos,ax=ax,edgelist=[(8,17)],edge_color=RED,width=2.5,style="dashed",arrowsize=22,node_size=380,connectionstyle="arc3,rad=0.35")
    nx.draw_networkx_nodes(graph,pos,ax=ax,node_size=380,node_color=["#d8efe8" if n in (29,17) else "#e0eafa" if n==8 else "#f8fafc" for n in graph],edgecolors=[DARK if n==8 else "#a7b5c3" for n in graph],linewidths=[2.5 if n==8 else 1 for n in graph])
    nx.draw_networkx_labels(graph,pos,ax=ax,font_size=9)
    ax.axis("off")
    fig.suptitle("C3 · path_dag_07 · 第 5 次采样 · 漏掉中间节点的非法跳转",fontsize=17,y=.96)
    text=fig.add_axes((.64,.27,.34,.57));text.axis("off")
    text.text(0,.95,"起点 29 → 终点 17",fontsize=15,weight="bold")
    text.text(0,.75,"模型已合法执行：29 → 9 → 8\n第 3 步尝试：8 → 17（不存在）",fontsize=12,linespacing=1.8)
    text.text(0,.47,"图中存在的是 8 → 7 → 17。\n还需经过节点 7，共两步。\n裁判停在节点 8，不能到达 17。",fontsize=12,linespacing=1.8)
    text.text(0,.09,"蓝：合法前缀   绿：参考最短路\n红虚线：不存在的尝试边\n灰箭头：完整 DAG 的其他边",fontsize=11,linespacing=1.8)
    fig.text(.5,.1,"模型提交：29 → 9 → 8 → 17（非法）     |     验证最短路：29 → 9 → 8 → 7 → 17（4 步）",ha="center",fontsize=13)
    fig.text(.5,.045,"全图保留 32 个节点与所有真实有向边；红虚线不是图中的边，图上长度不代表代价。",ha="center",fontsize=10)
    save(fig,"C3_missing_edge.png")
    print("Three selected outputs replayed; empty-plan witness and shortest route verified; figures rendered.")


if __name__=="__main__":
    main()
