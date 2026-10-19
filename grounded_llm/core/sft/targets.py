"""Render candidate rankings from a task-provided map step."""
import numpy as np

def _same_distance(left: float, right: float) -> bool:
    return bool(np.isclose(left, right, rtol=1e-10, atol=1e-12))


def reported_candidates(step, reported: int | None) -> tuple[int, ...]:
    """Nearest candidates the answer names, nearest first.

    The environment still shows every legal move; only the answer narrows, so
    supervision stops spending tokens on moves far from the goal. The cap is
    hard: equal distances are broken by candidate ID, which keeps the target
    bounded when many successors tie.
    """
    distances = (step.current_map_distance, *step.candidate_map_distances)
    order = sorted(range(1, len(distances)), key=lambda i: (distances[i], i))
    return tuple(order if reported is None else order[:reported])


def _ranking(step, reported: int | None = None) -> str:
    distances = (step.current_map_distance, *step.candidate_map_distances)
    ordered = sorted((0, *reported_candidates(step, reported)),
                     key=lambda local_id: (distances[local_id], local_id))
    groups: list[list[int]] = []
    for local_id in ordered:
        if groups and _same_distance(distances[local_id],
                                     distances[groups[-1][0]]):
            groups[-1].append(local_id)
        else:
            groups.append([local_id])
    names = lambda group: " = ".join("current" if i == 0 else str(i) for i in group)
    return " < ".join(names(group) for group in groups)


def decision_text(step, chosen_id: int, reported: int | None = None) -> str:
    """Supervise the ordering and the choice, never the distances themselves.

    Asking for numbers would require the frozen Q projection to survive the
    attention stack; asking only for the near/far ordering keeps the target
    short enough for a whole blocks trajectory to fit on one GPU.
    """
    if step.done or not step.candidate_actions:
        raise ValueError("action supervision requires a nonterminal step with candidates")
    if chosen_id not in step.map_minimal_candidates:
        raise ValueError("chosen action is not a map-minimal candidate")
    return "\n".join([
        "The current state has not reached the goal.",
        f"Map-distance ranking to the goal, closest to farthest: {_ranking(step, reported)}.",
        f"Choose candidate {chosen_id} because its successor has the smallest map distance among the candidates.",
        f"<action>{chosen_id}</action>",
    ])
