"""Generate and validate fixed sparse undirected path suites from a formal config."""
import argparse
from collections import deque
import json
from pathlib import Path
import random


def shortest(neighbors, start, goal):
    """Return one shortest path as move objects, or None when disconnected."""
    queue, parents = deque([start]), {start: None}
    while queue:
        node = queue.popleft()
        if node == goal:
            moves = []
            while parents[node] is not None:
                before = parents[node]
                moves.append({"from": before, "to": node})
                node = before
            return list(reversed(moves))
        for other in neighbors[node]:
            if other not in parents:
                parents[other] = node
                queue.append(other)
    return None


def add_matching(neighbors, rng):
    """Add one non-overlapping perfect matching to an already simple graph."""
    nodes = list(neighbors)
    for _ in range(10_000):
        shuffled = rng.sample(nodes, len(nodes))
        pairs = list(zip(shuffled[::2], shuffled[1::2]))
        if all(right not in neighbors[left] for left, right in pairs):
            for left, right in pairs:
                neighbors[left].add(right)
                neighbors[right].add(left)
            return
    raise ValueError("Unable to add a simple random matching")


def regular_connected_graph(node_count, degree, rng):
    """Create a connected simple regular graph using a shuffled cycle and matchings."""
    if node_count < 4 or node_count % 2 or degree < 2 or degree >= node_count:
        raise ValueError("Need an even node count and 2 <= degree < node_count")
    neighbors = {node: set() for node in range(node_count)}
    cycle = rng.sample(range(node_count), node_count)
    for left, right in zip(cycle, cycle[1:] + cycle[:1]):
        neighbors[left].add(right)
        neighbors[right].add(left)
    for _ in range(degree - 2):
        add_matching(neighbors, rng)
    if any(len(values) != degree for values in neighbors.values()):
        raise AssertionError("Regular graph construction failed")
    return neighbors


def validate_suite(suite):
    config, cases = suite["config"], suite["cases"]
    graph = config["graph"]
    node_count, degree = graph["node_count"], graph["regular_degree"]
    if len(cases) != graph["pair_count"]:
        raise ValueError("Wrong number of path cases")
    if sum(case["replicates"] for case in cases) != config["max_calls_per_model"]:
        raise ValueError("Suite call count does not match formal config")
    if len({case["id"] for case in cases}) != len(cases):
        raise ValueError("Duplicate case IDs")
    for case in cases:
        if case["condition"] != config["conditions"][0] or case["node_count"] != node_count:
            raise ValueError("Case contract does not match config")
        neighbors = {int(node): set(values) for node, values in case["neighbors"].items()}
        if set(neighbors) != set(range(node_count)):
            raise ValueError("Graph node IDs are not complete")
        if any(len(values) != degree or any(node not in neighbors[other] for other in values)
               for node, values in neighbors.items()):
            raise ValueError("Graph is not symmetric regular adjacency")
        route = shortest(neighbors, case["start"], case["goal"])
        if route is None or len(route) != case["reference"]["length"]:
            raise ValueError("Reference shortest path is invalid")
        if len(route) < graph["minimum_shortest_moves"]:
            raise ValueError("Reference path is too short")
        maximum = graph.get("maximum_shortest_moves")
        if maximum is not None and len(route) > maximum:
            raise ValueError("Reference path is too long")


def prepare(config):
    graph = config["graph"]
    if graph.get("orientation") != "undirected":
        raise ValueError("This generator only creates undirected suites")
    rng = random.Random(config["seed"])
    node_count, degree = graph["node_count"], graph["regular_degree"]
    neighbors = regular_connected_graph(node_count, degree, rng)
    minimum, maximum = graph["minimum_shortest_moves"], graph.get("maximum_shortest_moves")
    candidate_pairs = []
    for start in range(node_count):
        for goal in range(start + 1, node_count):
            route = shortest(neighbors, start, goal)
            if len(route) >= minimum and (maximum is None or len(route) <= maximum):
                candidate_pairs.append((start, goal))
    if len(candidate_pairs) < graph["pair_count"]:
        raise ValueError("Not enough distinct start-goal pairs at the requested path length")
    selected = rng.sample(candidate_pairs, graph["pair_count"])
    cases = []
    for number, (left, right) in enumerate(selected):
        start, goal = (left, right) if rng.randrange(2) else (right, left)
        display_neighbors = {str(node): list(neighbors[node]) for node in range(node_count)}
        for values in display_neighbors.values():
            rng.shuffle(values)
        route = shortest(neighbors, start, goal)
        cases.append({
            "id": f"path_undirected_{node_count}_{number:02d}",
            "condition": config["conditions"][0],
            "node_count": node_count,
            "start": start,
            "goal": goal,
            "neighbors": display_neighbors,
            "node_order": rng.sample(range(node_count), node_count),
            "reference": {"length": len(route), "path": route},
            "replicates": config["replicates"],
        })
    suite = {"config": config, "case_count": len(cases),
             "planned_calls": sum(case["replicates"] for case in cases), "cases": cases}
    validate_suite(suite)
    return suite


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    out = Path(args.out)
    if out.exists():
        raise FileExistsError("A fixed suite already exists at this path")
    suite = prepare(config)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(suite, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lengths = [case["reference"]["length"] for case in suite["cases"]]
    print(json.dumps({"out": str(out), "cases": len(suite["cases"]),
                      "calls": suite["planned_calls"], "path_lengths": lengths}))


if __name__ == "__main__":
    main()
