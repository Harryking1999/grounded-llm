"""Shared ranking-only targets and graph SFT examples."""
from grounded_llm.tasks.path.environment import GraphEnvironment
from grounded_llm.tasks.blocks.qmap import BlocksStep
from grounded_llm.core.sft.text import terminal_text as shared_terminal_text
from grounded_llm.core.sft.types import SupervisedTurn, Demonstration
from grounded_llm.core.sft.targets import _same_distance, reported_candidates, _ranking, decision_text

from dataclasses import dataclass
from collections.abc import Mapping, Sequence

import numpy as np

from experiments.external_map_interface.src.q_map import GraphQMap

from .graph import GraphStep, graph_step
from .prompt import initial_prompt, turn_prompt


def terminal_text(executed_path: Sequence[int]) -> str:
    if not executed_path:
        raise ValueError("terminal summary needs an actual path")
    moves = len(executed_path) - 1
    actions = " -> ".join(
        f"move({source},{destination})"
        for source, destination in zip(executed_path[:-1], executed_path[1:])
    ) or "(none)"
    return shared_terminal_text("node", actions, moves, "moves")


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
