"""Locate the first irreversible greedy error; oracle labels never choose actions."""
import argparse
from collections import Counter
import json
from pathlib import Path
import random

import numpy as np
import torch

import grounded_llm.tasks.blocks.rules as blocks
from .blocks12_eval import construct
from .full_graph_data import Root, load_component
from grounded_llm.tasks.blocks.oracle import DistanceOracle, successors
from .tree_supervision_eval import QScorer


class RootOracle:
    def __init__(self, root):
        self.root = root

    def distances(self, sources, goal):
        values = np.zeros(len(sources), dtype=np.int16)
        valid = np.ones(len(sources), dtype=bool)
        for graph, cells in self.root.parts:
            s = np.asarray([sum(((int(mask) >> cell) & 1) << i for i, cell in enumerate(cells))
                            for mask in sources], dtype=np.uint64)
            g = sum(((int(goal) >> cell) & 1) << i for i, cell in enumerate(cells))
            d = graph.distance(s, g)
            valid &= d >= 0
            values += d
        return np.where(valid, values, -1)

    def distance(self, source, goal):
        return int(self.distances([source], goal)[0])


def trace_error(scorer, source, goal, oracle):
    state, trajectory = source, []
    initial = oracle.distance(source, goal)
    if initial < 0:
        raise ValueError("Diagnostic starts must be certified reachable")
    for step in range(source.bit_count() // 2 + 1):
        if state == goal:
            return {"success": True, "true_distance": initial, "trajectory": trajectory}
        candidates = successors(state)
        if not candidates:
            raise AssertionError("Certified reachable state has no legal action")
        masks = [mask for _, mask in candidates]
        scores = scorer.scores(masks, goal)
        labels = (oracle.distances(masks, goal) if hasattr(oracle, "distances") else
                  np.asarray([oracle.distance(mask, goal) for mask in masks]))
        chosen = int(scores.argmin())
        finite = labels >= 0
        true = oracle.distance(state, goal)
        best_good = float(scores[finite].min())
        cap = getattr(scorer, "label_cap", scorer.model.cap)
        star = {"source": str(state), "goal": str(goal), "step": step,
                "true_distance": true, "degree": len(candidates),
                "reachable_candidates": int(finite.sum()),
                "optimal_candidates": int((labels == true - 1).sum()),
                "chosen_action": int(candidates[chosen][0]),
                "chosen_truth": int(labels[chosen]),
                "chosen_score": float(scores[chosen]),
                "good_minus_chosen_score": best_good - float(scores[chosen]),
                "cap_conflict": bool((~finite).any() and labels[finite].min() >= cap),
                "candidate_finite_mae": float(np.abs(scores[finite] - labels[finite]).mean()),
                "candidates": [{"action": int(action), "shape_id": blocks.PLACEMENTS[action][1]["shape_id"],
                                "state": str(mask), "truth": int(label), "score": float(score)}
                               for (action, mask), label, score in zip(candidates, labels, scores)]}
        trajectory.append(star)
        if labels[chosen] < 0:
            child = masks[chosen]
            star["failure_type"] = ("removed_goal_cell" if child & goal != goal else
                                    "local_coverage_failure" if not blocks.locally_supported(child ^ goal)
                                    else "globally_unreachable")
            return {"success": False, "true_distance": initial,
                    "failure": star, "trajectory": trajectory}
        state = masks[chosen]
    raise AssertionError("Removal path exceeded its strict cell bound")


def summarize(rows):
    bins, errors = {}, []
    for row in rows:
        b = bins.setdefault(str(row["true_distance"]), {"tasks": 0, "success": 0})
        b["tasks"] += 1
        b["success"] += row["success"]
        if not row["success"]:
            errors.append(row["failure"])
    return {"tasks": len(rows), "success": sum(row["success"] for row in rows),
            "by_true_distance": bins,
            "failure_types": dict(Counter(e["failure_type"] for e in errors)),
            "failure_step": dict(Counter(e["step"] for e in errors)),
            "failure_remaining_distance": dict(Counter(e["true_distance"] for e in errors)),
            "failure_degree": dict(Counter(e["degree"] for e in errors)),
            "failure_action_shape": dict(Counter(blocks.PLACEMENTS[e["chosen_action"]][1]["shape_id"] for e in errors)),
            "cap_conflict_first_errors": sum(e["cap_conflict"] for e in errors),
            "all_first_errors_are_ranking_errors": all(e["chosen_truth"] < 0 and e["reachable_candidates"] > 0 for e in errors)}


def training_cases(records, seed, per_root=2, goal_mode="witness", roots=None, goal_fractions=None):
    """Generate training queries from declared source and goal coverage."""
    rng = random.Random(seed)
    for board, record in enumerate(records):
        if goal_mode=='goal_fractions':
            if not goal_fractions or any(not 0<f<1 for f in goal_fractions):
                raise ValueError('Provide goal fractions strictly between zero and one')
            exact=RootOracle(roots[board]);state=int(record['initial']);path=[state]
            while state:
                choices=successors(state);labels=exact.distances([s for _,s in choices],0)
                depth=exact.distance(state,0)
                state=next(s for (_,s),d in zip(choices,labels) if d==depth-1)
                path.append(state)
            for size,start in ((12,0),(8,max(0,len(path)-9))):
                source=path[start];length=len(path)-1-start
                yield dict(board=board,root_source=record['initial'],source=str(source),goal='0',query_group=f'clear{size}')
                for fraction in goal_fractions:
                    cut=max(1,min(length-1,round(length*fraction)))
                    yield dict(board=board,root_source=record['initial'],source=str(source),
                        goal=str(path[start+cut]),query_group=f'partial{size}',goal_fraction=fraction)
            continue
        tiles = [blocks.PLACEMENTS[blocks.BY_ACTION[(a["shape_id"], a["row"], a["col"]) ]][0]
                 for a in record["construction"]]
        for number in range(per_root):
            if goal_mode in ("random_path", "balanced6_13"):
                if goal_mode == "balanced6_13" and number % 2 == 0:
                    if roots is None:
                        raise ValueError("Balanced goals require existing exact root graphs")
                    exact = RootOracle(roots[board])
                    wanted = 6 + board % 3
                    for _ in range(100):
                        state, eligible = int(record["initial"]), []
                        while True:
                            choices = successors(state)
                            if not choices:
                                break
                            _, state = rng.choice(choices)
                            if state and exact.distance(int(record["initial"]), state) == wanted:
                                eligible.append(state)
                        if eligible:
                            goal = rng.choice(eligible)
                            break
                    else:
                        raise ValueError(f"Training root {board} lacks sampled exact {wanted}-move goal")
                    yield {"board": board, "source": record["initial"], "goal": str(goal)}
                    continue
                state, path = int(record["initial"]), []
                while True:
                    choices = successors(state)
                    if not choices:
                        break
                    _, state = rng.choice(choices)
                    if state:
                        path.append(state)
                if not path:
                    raise ValueError("No nonempty legal descendant")
                goal = path[-1]
                yield {"board": board, "source": record["initial"], "goal": str(goal)}
                continue
            keep = 1 + number % 3
            goal = 0
            for tile in rng.sample(tiles, keep):
                goal |= tile
            yield {"board": board, "source": record["initial"], "goal": str(goal)}


def run(checkpoint, graph_data, monitor_cases, out, seed, train_roots=1000, per_root=2, device="cpu", goal_mode="witness", goal_fractions=None):
    torch.set_num_threads(2)
    scorer = QScorer(checkpoint, device)
    records = json.loads((graph_data / "roots.json").read_text())
    graphs = {gid: load_component(graph_data / "graphs" / str(gid))
              for gid in sorted({p["graph"] for r in records for p in r["components"]})}
    roots = [Root(r, graphs) for r in records]
    monitor = json.loads(monitor_cases.read_text())["rows"]
    monitor = [r for r in monitor if r["goal"] != "0" and r["true_distance"] > 1]
    oracle = DistanceOracle(cache_limit=8000000, seconds=14400)
    result = {"checkpoint": str(checkpoint), "identity": scorer.identity,
              "seed": seed, "training_goal_mode": goal_mode,
              "scope": "Diagnostic only; monitoring failures never enter repair training"}
    for name, cases in (("monitor_long", monitor),
                        ("training_long", training_cases(records[:train_roots], seed, per_root, goal_mode, roots,goal_fractions))):
        rows = []
        for index, row in enumerate(cases):
            truth_oracle = oracle if name == "monitor_long" else RootOracle(roots[row["board"]])
            rows.append({**row,**trace_error(scorer, int(row["source"]), int(row["goal"]), truth_oracle)})
            if (index + 1) % 100 == 0:
                print(json.dumps({"section": name, "tasks": index + 1}), flush=True)
            if len(scorer.cache) > 200000:
                scorer.cache.clear()
        result[name] = {"summary": summarize(rows), "rows": rows}
        groups=sorted({r.get('query_group',r.get('group')) for r in rows if r.get('query_group',r.get('group'))})
        if groups:result[name]['groups']={g:summarize([r for r in rows if r.get('query_group',r.get('group'))==g]) for g in groups}
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result) + "\n")
    print(json.dumps({k: result[k]["summary"] for k in ("monitor_long", "training_long")}), flush=True)


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--graph-data", type=Path, required=True)
    p.add_argument("--monitor-cases", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--seed", type=int, default=2026101901)
    p.add_argument("--train-roots", type=int, default=1000)
    p.add_argument("--per-root", type=int, default=2)
    p.add_argument("--device", default="cpu")
    p.add_argument("--goal-mode", choices=["witness", "random_path", "balanced6_13", "goal_fractions"], default="witness")
    p.add_argument('--profile-config',type=Path)
    args = p.parse_args()
    profile=json.loads(args.profile_config.read_text())['diagnosis'] if args.profile_config else {}
    run(args.checkpoint, args.graph_data, args.monitor_cases, args.out, args.seed,
        args.train_roots, args.per_root, args.device, profile.get('goal_mode',args.goal_mode),profile.get('goal_fractions'))


if __name__ == "__main__":
    main()
