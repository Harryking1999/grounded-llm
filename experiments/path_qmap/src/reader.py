"""Frozen Q inference needs node IDs and topology, never shortest-path labels."""
from pathlib import Path

import numpy as np
import torch

from experiments.blocks_distance_map.src.model import StateEncoder


class FrozenPathMap:
    def __init__(self, encoder, adjacency, device="cpu"):
        self.encoder = encoder.to(device).eval().requires_grad_(False)
        self.adjacency = np.asarray(adjacency, dtype=bool).copy()
        self.device = torch.device(device)
        self.metric = encoder.metric
        self.features = torch.eye(len(adjacency), device=device)

    @classmethod
    def load(cls, path: Path, device="cpu"):
        saved = torch.load(path, map_location="cpu", weights_only=True)
        if saved.get("model_type") != "state_q_path":
            raise ValueError("Expected a state-Q path checkpoint")
        spec = saved["config"]["model"]
        adjacency = saved["adjacency"].numpy()
        model = StateEncoder(spec["state_dim"], spec["encoder_hidden_dim"], saved["cap"],
                             spec["metric"], input_dim=len(adjacency))
        model.load_state_dict(saved["model"])
        return cls(model, adjacency, device)

    @torch.no_grad()
    def encode(self, ids):
        values = np.asarray(ids)
        if values.dtype.kind not in "iu" or np.any(values < 0) or np.any(values >= len(self.adjacency)):
            raise ValueError("Node IDs must be integers inside the frozen graph")
        return self.encoder.encode(self.features[torch.as_tensor(values, device=self.device)]).cpu()

    def distance(self, source, goal):
        return self.encoder.distance(source, goal)
