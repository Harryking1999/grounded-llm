"""Select disjoint graph pairs and rebuild deterministic SFT trajectories."""

from collections import deque
from collections.abc import Mapping
from pathlib import Path
import json

import numpy as np

from experiments.external_map_interface.src.q_map import GraphQMap
from experiments.external_map_interface.src.transitions import GraphEnvironment

from .sft import Demonstration, greedy_demonstration


def first_turn_from_record(environment, qmap, record):
    """Reproduce the first turn without requiring a successful later rollout."""
    from .graph import graph_step
    from .prompt import initial_prompt, turn_prompt
    from .sft import SupervisedTurn, decision_text

    rng = np.random.default_rng(int(record["sample_seed"]))
    count = len(environment.adjacency)
    node_order = rng.permutation(count).tolist()
    neighbor_order = {node: rng.permutation(np.flatnonzero(environment.adjacency[node])).tolist()
                      for node in range(count)}
    start, goal = int(record["start"]), int(record["goal"])
    step = graph_step(environment, qmap, start, goal, executed_path=[start], rng=rng)
    if step.done or not step.candidate_actions:
        raise ValueError("First-turn relation supervision needs legal nonterminal candidates")
    text = initial_prompt(environment.adjacency, start, goal, node_order=node_order,
                          neighbor_order=neighbor_order) + "\n\n" + turn_prompt(step, [start])
    chosen = int(rng.choice(step.map_minimal_candidates))
    return SupervisedTurn(text, decision_text(step, chosen), step, (start,), chosen)


def load_graph(source_root: Path, graph_id: str):
    graph_root = Path(source_root) / graph_id
    with (graph_root / "suite.json").open(encoding="utf-8") as handle:
        suite = json.load(handle)
    map_root = graph_root / "map"
    environment = GraphEnvironment.load(map_root / "inputs.npz")
    qmap = GraphQMap.load(map_root / "map.npz", len(environment.adjacency),
                          len(environment.actions))
    if not suite["cases"] or any(case["node_count"] != len(environment.adjacency)
                                 for case in suite["cases"]):
        raise ValueError(f"suite for {graph_id} does not match the graph")
    for case in suite["cases"]:
        if any(set(case["neighbors"][str(node)]) !=
               set(np.flatnonzero(environment.adjacency[node]))
               for node in range(len(environment.adjacency))):
            raise ValueError(f"suite for {graph_id} has different edges")
    return environment, qmap, suite


def shortest_move_counts(adjacency: np.ndarray) -> np.ndarray:
    """Only for pair stratification and evaluation; never passed to the reader."""
    count = len(adjacency)
    result = np.full((count, count), -1, dtype=np.int16)
    neighbors = [np.flatnonzero(row).tolist() for row in adjacency]
    for start in range(count):
        result[start, start] = 0
        queue = deque([start])
        while queue:
            current = queue.popleft()
            for destination in neighbors[current]:
                if result[start, destination] < 0:
                    result[start, destination] = result[start, current] + 1
                    queue.append(destination)
    return result


def demonstration_from_record(environment: GraphEnvironment, qmap: GraphQMap,
                              record: Mapping) -> Demonstration:
    rng = np.random.default_rng(int(record["sample_seed"]))
    count = len(environment.adjacency)
    node_order = rng.permutation(count).tolist()
    neighbor_order = {
        node: rng.permutation(np.flatnonzero(environment.adjacency[node])).tolist()
        for node in range(count)
    }
    return greedy_demonstration(
        environment, qmap, int(record["start"]), int(record["goal"]), rng=rng,
        node_order=node_order, neighbor_order=neighbor_order,
    )


def select_graph_records(environment: GraphEnvironment, qmap: GraphQMap,
                         suite: Mapping, graph_id: str, config: Mapping,
                         *, seed: int) -> tuple[list[dict], dict]:
    """Sample unique pairs; reserve every historical suite pair for testing."""
    rng = np.random.default_rng(seed)
    distances = shortest_move_counts(environment.adjacency)
    reserved = set()
    for case in suite["cases"]:
        start, goal = int(case["start"]), int(case["goal"])
        reserved.update(((start, goal), (goal, start)))
    records: list[dict] = []
    excluded = {"dead_end": 0, "over_action_limit": 0}
    for shortest_moves in config["ordinary_shortest_moves"]:
        candidates = [
            (int(start), int(goal))
            for start, goal in np.argwhere(distances == shortest_moves)
            if (int(start), int(goal)) not in reserved
        ]
        rng.shuffle(candidates)
        train_count = config["train_pairs_per_length_per_graph"]
        validation_count = config["validation_pairs_per_length_per_graph"]
        accepted = 0
        for start, goal in candidates:
            record = {
                "graph_id": graph_id,
                "start": start,
                "goal": goal,
                "shortest_moves": int(shortest_moves),
                "sample_seed": int(rng.integers(0, 2**63)),
            }
            demonstration = demonstration_from_record(environment, qmap, record)
            if not demonstration.success:
                excluded["dead_end"] += 1
                continue
            action_count = len(demonstration.executed_path) - 1
            if action_count > config["maximum_demonstration_actions"]:
                excluded["over_action_limit"] += 1
                continue
            record["action_count"] = action_count
            record["split"] = "train" if accepted < train_count else "validation"
            records.append(record)
            accepted += 1
            if accepted == train_count + validation_count:
                break
        if accepted != train_count + validation_count:
            raise ValueError(f"{graph_id} has insufficient usable length-{shortest_moves} pairs")
    nodes = rng.permutation(len(environment.adjacency)).tolist()
    terminal_train = config["train_start_equals_goal_per_graph"]
    terminal_validation = config["validation_start_equals_goal_per_graph"]
    if len(nodes) < terminal_train + terminal_validation:
        raise ValueError(f"{graph_id} has insufficient distinct terminal nodes")
    for index, node in enumerate(nodes[:terminal_train + terminal_validation]):
        records.append({
            "graph_id": graph_id,
            "start": int(node),
            "goal": int(node),
            "shortest_moves": 0,
            "sample_seed": int(rng.integers(0, 2**63)),
            "action_count": 0,
            "split": "train" if index < terminal_train else "validation",
        })
    return records, excluded


def build_manifest(source_root: Path, config: Mapping) -> dict:
    graph_ids = config["train_validation_graphs"]
    if config["unseen_test_graph"] in graph_ids:
        raise ValueError("unseen test graph may not be used for training")
    records = []
    exclusions = {}
    for graph_index, graph_id in enumerate(graph_ids):
        environment, qmap, suite = load_graph(source_root, graph_id)
        selected, counts = select_graph_records(
            environment, qmap, suite, graph_id, config,
            seed=int(config["seed"]) + graph_index,
        )
        records.extend(selected)
        exclusions[graph_id] = counts
    return {"study": config["study"], "source_root": str(Path(source_root).resolve()),
            "records": records, "excluded": exclusions}
