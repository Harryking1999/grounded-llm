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
    def __init__(self, env, checkpoint, device="cpu"):
        saved = torch.load(checkpoint, map_location=device, weights_only=False)
        if saved["model_type"] != MODEL_TYPE or saved["actions"] != list(env.actions):
            raise ValueError("Expected matching Puzzle8 Q checkpoint")
        model = QMap(**saved["config"]["model"]).to(device)
        model.load_state_dict(saved["model"])
        super().__init__(env, model, device)


    def choose(self, state, goal=0):
        return super().choose(state, goal)
