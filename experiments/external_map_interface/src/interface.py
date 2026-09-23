"""Fresh per-step prompt; only learned candidate distances differ by condition."""

from .q_map import GraphQMap
from .transitions import GraphEnvironment


def graph_prompt(env: GraphEnvironment, current: int, goal: int,
                 q_map: GraphQMap | None = None, *, case=None, path=None) -> str:
    if case is not None:
        return study_prompt(env, current, goal, q_map, case, path)
    neighbors = "\n".join(
        f"{node}: {', '.join(map(str, env.adjacency[node].nonzero()[0]))}"
        for node in range(len(env.adjacency))
    )
    candidates = []
    for action_id in env.legal_actions(current):
        target = int(env.actions[action_id, 1])
        line = f"action_id={action_id}, to_node={target}"
        if q_map is not None:
            line += f", learned_map_distance_to_goal={q_map.candidate_distance(current, action_id, goal):.6f}"
        candidates.append(line)
    return (
        "Find a route to the goal. Choose one legal action at this step. "
        "The graph and legal action endpoints come from the environment. "
        "If shown, candidate distances are predictions from a learned roadmap; "
        "they may be imperfect. Respond only with JSON: {\"action_id\": integer}.\n\n"
        f"Graph adjacency:\n{neighbors}\n\nCurrent node: {current}\nGoal node: {goal}\n"
        "Legal actions:\n" + "\n".join(candidates)
    )


def study_prompt(env, current, goal, q_map, case, path):
    """Preserve the fixed suite's presentation order and no-repeat rule."""
    neighbors = "\n".join(
        f"{node}: {', '.join(map(str, case['neighbors'][str(node)]))}"
        for node in case["node_order"])
    by_target = {int(env.actions[a, 1]): a for a in env.legal_actions(current)}
    candidates = []
    for target in case["neighbors"][str(current)]:
        if target in path:
            continue
        action_id = by_target[target]
        line = f"action_id={action_id}, to_node={target}"
        if q_map is not None:
            line += f", learned_map_distance_to_goal={q_map.candidate_distance(current, action_id, goal):.6f}"
        candidates.append(line)
    return (
        f"Task: Find a SHORTEST valid path from node {case['start']} to node {goal} "
        f"in this {len(env.adjacency)}-node undirected graph. Minimize the number of edges traversed.\n\n"
        "Rules: Each move costs 1. Adjacency lists contain all neighbors; edges are bidirectional. "
        "Each node, including the start, may be visited at most once. Node numbers and presentation "
        "order carry no spatial or numerical ordering. Any shortest path is accepted.\n\n"
        f"Neighbors:\n{neighbors}\n\nCurrent node: {current}\nGoal node: {goal}\n"
        f"Executed node sequence: {', '.join(map(str, path))}\n"
        "Legal actions (already visited nodes excluded):\n" + "\n".join(candidates) + "\n\n"
        "Choose ONE action now. The environment executes it and provides the actual new state "
        "for the next decision. If shown, distances are predictions from a learned roadmap, "
        "not exact shortest-path lengths; they may be imperfect. "
        'Respond with one JSON object: {"action_id": integer}.'
    )
