"""Task-specific 81-bit input; shared state MLP and symmetric distance."""
import numpy as np
import torch
from grounded_llm.core.qmap.reader import EnvironmentQMap
from grounded_llm.core.qmap.model import StateEncoder
from grounded_llm.core.qmap.model import soft_rank_loss

MODEL_TYPE = "puzzle8_pure_q"


class QMap(StateEncoder):
    def __init__(self, dimension=128, hidden=256, metric="l1"):
        super().__init__(dimension, hidden, 31, metric, input_dim=81)


class MapReader(EnvironmentQMap):
    def __init__(self, env, checkpoint, device="cpu", *, coordinate_frame="absolute"):
        if coordinate_frame not in ("absolute", "goal_normalized"):
            raise ValueError(f"Unknown Puzzle8 coordinate frame: {coordinate_frame}")
        saved = torch.load(checkpoint, map_location=device, weights_only=False)
        if saved["model_type"] != MODEL_TYPE or saved["actions"] != list(env.actions):
            raise ValueError("Expected matching Puzzle8 Q checkpoint")
        model = QMap(**saved["config"]["model"]).to(device)
        model.load_state_dict(saved["model"])
        super().__init__(env, model, device)
        self.coordinate_frame = coordinate_frame

    def query_vectors(self, states, goal):
        """Transform current, goal and successors with one goal coordinate frame."""
        if self.coordinate_frame == 'goal_normalized':
            states, _ = self.env.goal_frame(states, goal)
        return self.encode(states)

    def scores(self, source, goal):
        """Optional task symmetry applies to pair scoring; encode stays unchanged."""
        if self.coordinate_frame == "goal_normalized":
            source, goal = self.env.goal_frame(source, goal)
        return super().scores(source, goal)

    def choose(self, state, goal=0):
        return super().choose(state, goal)
