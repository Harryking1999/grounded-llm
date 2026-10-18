"""Task-specific 81-bit input; shared state MLP and symmetric distance."""
import numpy as np
import torch
from experiments.blocks_distance_map.src.model import StateEncoder, soft_rank_loss

MODEL_TYPE = "puzzle8_pure_q"


class QMap(StateEncoder):
    def __init__(self, dimension=128, hidden=256, metric="l1"):
        super().__init__(dimension, hidden, 31, metric, input_dim=81)


class MapReader:
    def __init__(self, env, checkpoint, device="cpu"):
        saved = torch.load(checkpoint, map_location=device, weights_only=False)
        if saved["model_type"] != MODEL_TYPE or saved["actions"] != list(env.actions):
            raise ValueError("Expected matching Puzzle8 Q checkpoint")
        self.env, self.device = env, device
        self.model = QMap(**saved["config"]["model"]).to(device)
        self.model.load_state_dict(saved["model"])
        self.model.eval().requires_grad_(False)

    @torch.no_grad()
    def encode(self, ids):
        return self.model.encode(torch.as_tensor(self.env.features(ids), device=self.device))

    @torch.no_grad()
    def scores(self, source, goal):
        return self.model.distance(self.encode(source), self.encode(goal)).cpu().numpy()

    def choose(self, state, goal=0):
        children = self.env.successors([state])[0]
        legal = children >= 0
        action = np.flatnonzero(legal)[self.scores(children[legal], [goal]).argmin()]
        return self.env.actions[action], int(children[action])
