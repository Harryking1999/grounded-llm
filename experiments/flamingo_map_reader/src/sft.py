"""Shared ranking-only targets and graph SFT examples."""

from dataclasses import dataclass
from collections.abc import Mapping, Sequence

import numpy as np

from experiments.external_map_interface.src.q_map import GraphQMap
from experiments.external_map_interface.src.transitions import GraphEnvironment

from .graph import GraphStep, graph_step
from .blocks import BlocksStep
from .prompt import initial_prompt, turn_prompt
from .text import terminal_text as shared_terminal_text


def _same_distance(left: float, right: float) -> bool:
    return bool(np.isclose(left, right, rtol=1e-10, atol=1e-12))


def _ranking(step) -> tuple[str, tuple[int, ...]]:
    distances = (step.current_map_distance, *step.candidate_map_distances)
    ordered = sorted(enumerate(distances),
                     key=lambda item: (item[1], item[0]))
    groups: list[list[int]] = []
    for local_id, distance in ordered:
        if groups and _same_distance(distance,
                                     distances[groups[-1][0]]):
            groups[-1].append(local_id)
        else:
            groups.append([local_id])
    names = lambda group: " = ".join("current" if i == 0 else str(i) for i in group)
    return " < ".join(names(group) for group in groups), tuple(
        local_id for group in groups for local_id in group if local_id != 0)


def decision_text(step, chosen_id: int) -> str:
    if step.done or not step.candidate_actions:
        raise ValueError("action supervision requires a nonterminal step with candidates")
    if chosen_id not in step.map_minimal_candidates:
        raise ValueError("chosen action is not a map-minimal candidate")
    ranking, ordered_ids = _ranking(step)
    relation = []
    for position, local_id in enumerate(ordered_ids):
        distance = step.candidate_map_distances[local_id - 1]
        if _same_distance(distance, step.current_map_distance):
            comparison = "at the same map distance as the current state"
        elif distance < step.current_map_distance:
            comparison = "closer to the goal than the current state"
        else:
            comparison = "farther from the goal than the current state"
        subject = "Candidate" if position == 0 else "candidate"
        relation.append(f"{subject} {local_id} is {comparison}")
    return "\n".join([
        "The current state has not reached the goal.",
        f"Map-distance ranking to the goal, closest to farthest: {ranking}.",
        "; ".join(relation) + ".",
        f"Choose candidate {chosen_id} because its successor has the smallest map distance among the candidates.",
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
    return shared_terminal_text("node", actions, moves, "moves")


@dataclass(frozen=True)
class SupervisedTurn:
    user_text: str
    answer_text: str
    step: GraphStep | BlocksStep
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
