"""Hanoi-only state Q and frozen reader; shared MLP/distance primitives."""
import torch
from grounded_llm.core.qmap.reader import EnvironmentQMap

from grounded_llm.core.qmap.model import StateEncoder
from grounded_llm.core.qmap.model import distance as shared_distance
from grounded_llm.core.qmap.model import soft_rank_loss


MODEL_TYPE = "pure_q_symmetric_blocks"


def distance(source, goal, metric="l1", group_size=8):
    return shared_distance(source, goal, metric, group_size)


class QMap(StateEncoder):
    def __init__(self, feature_dim, dimension, hidden, cap, metric="l1", group_size=8, symmetry=None):
        if metric not in ("l1", "linf", "l2", "max_l1"):
            raise ValueError(f"Unknown Q distance: {metric}")
        if metric == "max_l1" and (group_size < 1 or dimension % group_size):
            raise ValueError("Q dimension must be divisible by the positive group size")
        super().__init__(dimension, hidden, cap, metric, input_dim=feature_dim, group_size=group_size)
        if symmetry is not None:
            raise ValueError("Hanoi Q has no cube symmetry adapter")


class MapReader(EnvironmentQMap):
    """Reusable frozen inference API. Needs features/rules, never an oracle."""
    def __init__(self, env, checkpoint, device="cpu"):
        if not env.name.startswith("hanoi"):
            raise ValueError("Hanoi reader requires a Hanoi environment")
        saved = torch.load(checkpoint, map_location=device, weights_only=False)
        if saved.get("model_type") != MODEL_TYPE:
            raise ValueError("Expected a pure-Q checkpoint")
        if saved["task"] != env.name or saved["actions"] != list(env.actions):
            raise ValueError("Checkpoint belongs to a different task/action convention")
        model = QMap(env.feature_dim, **saved["model_config"], cap=saved["cap"]).to(device)
        model.load_state_dict(saved["model"])
        super().__init__(env, model, device)
