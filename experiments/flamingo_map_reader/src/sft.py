"""Build map-grounded, per-turn SFT text from actual graph trajectories."""

from dataclasses import dataclass
from collections.abc import Mapping, Sequence

import numpy as np

from experiments.external_map_interface.src.q_map import GraphQMap
from experiments.external_map_interface.src.transitions import GraphEnvironment

from .graph import GraphStep, graph_step
from .prompt import initial_prompt, turn_prompt


def _same_distance(left: float, right: float) -> bool:
    return bool(np.isclose(left, right, rtol=1e-10, atol=1e-12))


def _ranking(step: GraphStep) -> tuple[str, tuple[int, ...]]:
    ordered = sorted(enumerate(step.candidate_map_distances, 1),
                     key=lambda item: (item[1], item[0]))
    groups: list[list[int]] = []
    for local_id, distance in ordered:
        if groups and _same_distance(distance,
                                     step.candidate_map_distances[groups[-1][0] - 1]):
            groups[-1].append(local_id)
        else:
            groups.append([local_id])
    return ", ".join(" = ".join(map(str, group)) for group in groups), tuple(
        local_id for group in groups for local_id in group)


def decision_text(step: GraphStep, chosen_id: int) -> str:
    if step.done or not step.candidate_actions:
        raise ValueError("action supervision requires a nonterminal step with candidates")
    if chosen_id not in step.map_minimal_candidates:
        raise ValueError("chosen action is not a map-minimal candidate")
    ranking, ordered_ids = _ranking(step)
    relation = []
    for position, local_id in enumerate(ordered_ids):
        distance = step.candidate_map_distances[local_id - 1]
        if _same_distance(distance, step.current_map_distance):
            comparison = "at the same map distance as the current node" if position == 0 else "at the same map distance"
        elif distance < step.current_map_distance:
            comparison = "closer to the goal than the current node" if position == 0 else "closer"
        else:
            comparison = "farther from the goal than the current node" if position == 0 else "farther"
        subject = "Candidate" if position == 0 else "candidate"
        relation.append(f"{subject} {local_id} is {comparison}")
    displayed = "; ".join(
        f"{local_id}={distance:.4f}"
        for local_id, distance in enumerate(step.candidate_map_distances, 1)
    )
    return "\n".join([
        "The current node has not reached the goal.",
        f"Current-to-goal map distance: {step.current_map_distance:.4f}.",
        f"Candidate successor-to-goal map distances: {displayed}.",
        f"Ranking from closest to farthest: {ranking}.",
        "; ".join(relation) + ".",
        f"Choose candidate {chosen_id} because its predicted successor has the smallest map distance.",
        f"<action>{chosen_id}</action>",
    ])


def terminal_text(executed_path: Sequence[int]) -> str:
    if not executed_path:
        raise ValueError("terminal summary needs an actual path")
    moves = len(executed_path) - 1
    actions = " -> ".join(
        f"move({source},{destination})"
        for source, destination in zip(executed_path[:-1], executed_path[1:])
    ) or "(none)"
    count_text = "no executed moves" if moves == 0 else (
        "one executed move" if moves == 1 else f"{moves} executed moves"
    )
    return "\n".join([
        "The current node is the goal.",
        f"Executed actions: {actions}.",
        f"Summary: Reached node {executed_path[-1]} after {count_text}.",
        "<done/>",
    ])


@dataclass(frozen=True)
class SupervisedTurn:
    user_text: str
    answer_text: str
    step: GraphStep
    executed_path: tuple[int, ...]
    chosen_id: int | None


@dataclass(frozen=True)
class Demonstration:
    turns: tuple[SupervisedTurn, ...]
    executed_path: tuple[int, ...]
    success: bool


def greedy_demonstration(
    environment: GraphEnvironment,
    qmap: GraphQMap,
    start: int,
    goal: int,
    *,
    rng: np.random.Generator,
    node_order: Sequence[int] | None = None,
    neighbor_order: Mapping[int, Sequence[int]] | None = None,
) -> Demonstration:
    """Shuffle legal IDs each turn and stop only after an explicit final turn."""
    first_text = initial_prompt(environment.adjacency, start, goal,
                                node_order=node_order, neighbor_order=neighbor_order)
    path = [start]
    turns = []
    for _ in range(len(environment.adjacency) + 1):
        step = graph_step(environment, qmap, path[-1], goal, executed_path=path, rng=rng)
        update = turn_prompt(step, path)
        user_text = first_text + "\n\n" + update if not turns else update
        if step.done:
            turns.append(SupervisedTurn(user_text, terminal_text(path), step,
                                        tuple(path), None))
            return Demonstration(tuple(turns), tuple(path), True)
        if not step.candidate_actions:
            return Demonstration(tuple(turns), tuple(path), False)
        chosen_id = int(rng.choice(step.map_minimal_candidates))
        turns.append(SupervisedTurn(user_text, decision_text(step, chosen_id), step,
                                    tuple(path), chosen_id))
        _, destination = step.execute(environment, chosen_id)
        path.append(destination)
    raise AssertionError("a no-revisit graph trajectory cannot exceed its node count")
