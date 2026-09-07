"""Fix all 48 inputs before model sampling, retaining private construction witnesses."""
from collections import Counter
import json
import random
from tasks import CONFIG, ROOT, STUDY, TASKS


def prepare(config=CONFIG):
    rng, cases, seen = random.Random(config["seed"]), [], set()
    for condition in ("blocks8", "blocks12"):
        task = TASKS[condition]
        for number in range(config["samples_per_condition"]):
            for _attempt in range(1600):
                mask, additions = 0, []
                for _ in range(task.construction_objects):
                    boundary = task.boundary(mask)
                    candidates = [(tile, action) for tile, action in task.placements if not tile & mask and (not mask or tile & boundary)]
                    if not candidates:
                        break
                    tile, action = rng.choice(candidates)
                    mask |= tile
                    additions.append(dict(action))
                if len(additions) == task.construction_objects and task.normalized_key(mask) not in seen:
                    break
            else:
                raise ValueError("Unable to generate enough distinct boards")
            seen.add(task.normalized_key(mask))
            current = mask
            for action in additions:
                current = task.apply(current, action, mask)
            assert current == 0
            cases.append({"id": f"{condition}_{number:02d}", "condition": condition,
                          "grid": "/".join(task.to_rows(mask)), "construction_objects": task.construction_objects,
                          "construction_reference": additions, "occupied_cells": mask.bit_count(),
                          "input_shape_counts": dict(Counter(str(a["shape_id"]) for a in additions)),
                          "replicates": config["replicates"]})
    graph = json.loads((ROOT / config["graph_config"]).read_text())["neighbors"]
    edges = [(u, v) for u in range(32) for v in graph[str(u)] if u < v]
    task, rng, seen_graphs = TASKS["path_dag"], random.Random(config["seed"] + 20000), set()
    for number in range(config["samples_per_condition"]):
        for _attempt in range(10000):
            order = rng.sample(range(32), 32)
            rank = {node: i for i, node in enumerate(order)}
            neighbors = {str(u): [] for u in range(32)}
            for u, v in edges:
                a, b = (u, v) if rank[u] < rank[v] else (v, u)
                neighbors[str(a)].append(b)
            identity = tuple((int(u), v) for u, vs in neighbors.items() for v in vs)
            if identity in seen_graphs:
                continue
            reachable = []
            for start in range(32):
                for goal in range(32):
                    if start != goal:
                        route = task.shortest({"start": start, "goal": goal, "neighbors": neighbors})
                        if route is not None and len(route) >= config["path"]["minimum_shortest_moves"]:
                            reachable.append((start, goal, route))
            if reachable:
                seen_graphs.add(identity)
                break
        else:
            raise ValueError("Unable to generate enough DAGs meeting the minimum shortest distance")
        start, goal, route = rng.choice(reachable)
        node_order = rng.sample(range(32), 32)
        for vs in neighbors.values():
            rng.shuffle(vs)
        cases.append({"id": f"path_dag_{number:02d}", "condition": "path_dag", "start": start, "goal": goal,
                      "neighbors": neighbors, "node_order": node_order, "hidden_topological_order": order,
                      "reference": {"length": len(route), "path": route}, "replicates": config["replicates"]})
    return {"config": config, "case_count": len(cases), "planned_calls": sum(c["replicates"] for c in cases), "cases": cases}


if __name__ == "__main__":
    suite = prepare()
    out = STUDY / "runs/suite.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        raise FileExistsError("Suite already exists; do not replace fixed inputs")
    out.write_text(json.dumps(suite, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"suite": str(out), "cases": suite["case_count"], "calls": suite["planned_calls"],
                      "path_lengths": [c["reference"]["length"] for c in suite["cases"] if c["condition"] == "path_dag" ]}))
