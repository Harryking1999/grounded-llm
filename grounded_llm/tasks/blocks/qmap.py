"""Frozen board encoder, legal removals, and goal-conditioned map snapshots."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from grounded_llm.core.qmap.reader import FrozenQMap

from grounded_llm.tasks.blocks.model import BoardEncoder
from grounded_llm.tasks.blocks.model import board_bits
from grounded_llm.core.qmap.model import distance
from grounded_llm.tasks.blocks.oracle import successors
import grounded_llm.tasks.blocks.rules as rules
from grounded_llm.core.reader.memory import MapBatch, q_map_batch


class FrozenBoardMap(FrozenQMap):
    """Use the shared Q encoder and its original metric without retraining it."""

    def __init__(self, encoder: BoardEncoder, device="cpu"):
        super().__init__(encoder, self.features, device, cpu_output=True)

    @classmethod
    def load(cls, checkpoint: Path, device="cpu"):
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        config = saved["config"]["model"]
        encoder = BoardEncoder(config["state_dim"], config["encoder_hidden_dim"],
                               saved["cap"], saved["metric"])
        encoder.load_state_dict(saved["model"])
        return cls(encoder, device)

    @staticmethod
    def features(masks):
        if any(type(mask) is not int or mask < 0 or mask.bit_length() > 100 for mask in masks):
            raise ValueError("expected 100-bit board states")
        return board_bits(masks)


@dataclass(frozen=True)
class BlocksStep:
    current: int
    goal: int
    map_batch: MapBatch
    candidate_actions: tuple[int, ...]
    candidate_destinations: tuple[int, ...]
    current_map_distance: float
    candidate_map_distances: tuple[float, ...]
    map_minimal_candidates: tuple[int, ...]
    done: bool

    def execute(self, local_id: int) -> tuple[int, int]:
        if type(local_id) is not int or not 1 <= local_id <= len(self.candidate_actions):
            raise ValueError("candidate ID is not present in this step")
        action_id = self.candidate_actions[local_id - 1]
        actual = rules.apply(self.current, rules.PLACEMENTS[action_id][1])
        return action_id, actual


def blocks_step(qmap: FrozenBoardMap, current: int, goal: int,
                *, rng: np.random.Generator | None = None) -> BlocksStep:
    candidates = successors(current)
    if rng is not None:
        candidates = [candidates[int(i)] for i in rng.permutation(len(candidates))]
    actions = tuple(action for action, _ in candidates)
    destinations = tuple(state for _, state in candidates)
    # V(o,a) = Q(T(o,a)) - Q(o); passing Q(T(o,a)) supplies Q(o) + V(o,a).
    vectors = qmap.encode([current, goal, *destinations])
    scores = distance(vectors, vectors[1], qmap.metric)
    distances = tuple(map(float, scores[2:]))
    minimum = min(distances, default=float("inf"))
    minimal = tuple(i for i, d in enumerate(distances, 1)
                    if np.isclose(d, minimum, rtol=1e-10, atol=1e-12))
    memory = q_map_batch(vectors)
    return BlocksStep(current, goal, memory, actions, destinations,
                      float(scores[0]), distances, minimal, current == goal)


def action_text(action_id: int) -> str:
    action = rules.PLACEMENTS[action_id][1]
    return f"remove({action['shape_id']},{action['row']},{action['col']})"
