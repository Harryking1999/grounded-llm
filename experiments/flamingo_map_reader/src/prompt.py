"""Shared English text visible to map and no-map pathfinding conditions."""

from collections.abc import Mapping, Sequence

import numpy as np

from .graph import GraphStep


from .text import task_opening


INITIAL_TEMPLATE = """{opening}
Use the listed edges and visit each node at most once.

Neighbors:
{adjacency}"""


TURN_TEMPLATE = """[Environment update]
Current node: {current}
Actual executed path: [{path}]
Legal next moves:
{candidates}
Candidate numbers may change between turns; use only the numbers above.
[/Environment update]"""


def initial_prompt(
    adjacency: np.ndarray,
    start: int,
    goal: int,
    *,
    node_order: Sequence[int] | None = None,
    neighbor_order: Mapping[int, Sequence[int]] | None = None,
) -> str:
    count = len(adjacency)
    if adjacency.shape != (count, count) or not np.array_equal(adjacency, adjacency.T):
        raise ValueError("initial prompt requires a square undirected graph")
    if not (0 <= start < count and 0 <= goal < count):
        raise ValueError("start and goal must be graph nodes")
    order = list(range(count)) if node_order is None else list(node_order)
    if sorted(order) != list(range(count)):
        raise ValueError("node_order must contain every node once")
    rows = []
    for node in order:
        actual = np.flatnonzero(adjacency[node]).tolist()
        neighbors = (actual if neighbor_order is None else list(neighbor_order[node]))
        if sorted(neighbors) != actual:
            raise ValueError("presented neighbors disagree with the graph")
        rows.append(f"{node}: {', '.join(map(str, neighbors))}")
    return INITIAL_TEMPLATE.format(adjacency="\n".join(rows),
        opening=task_opening(f"Find a valid path from node {start} to node {goal} in this undirected graph, as short as you can."))


def turn_prompt(step: GraphStep, executed_path: Sequence[int]) -> str:
    if not executed_path or executed_path[-1] != step.current:
        raise ValueError("executed path must end at the actual current node")
    if set(step.candidate_destinations) & set(executed_path):
        raise ValueError("candidate moves may not revisit the executed path")
    candidates = "\n".join(
        f"{index}: {step.current} -> {destination}"
        for index, destination in enumerate(step.candidate_destinations, 1)
    ) or "(none)"
    return TURN_TEMPLATE.format(
        current=step.current,
        path=", ".join(map(str, executed_path)),
        candidates=candidates,
    )
