"""Fresh per-step prompt; only learned candidate distances differ by condition."""

from .q_map import GraphQMap
from .transitions import GraphEnvironment


def graph_prompt(env: GraphEnvironment, current: int, goal: int,
                 q_map: GraphQMap | None = None) -> str:
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
