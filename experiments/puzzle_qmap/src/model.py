"""Goal-independent Q(s), fixed symmetric distance, separate transition head."""
import torch
from torch import nn


def distance(source, goal):
    # Same dense-coordinate principle as blocks' directed_sum, symmetrized for
    # reversible puzzles. No goal-conditioned scorer or learned distance head.
    return (source - goal).abs().mean(dim=-1)


class QMap(nn.Module):
    def __init__(self, feature_dim, actions, dimension, hidden, diameter):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(feature_dim, hidden), nn.ReLU(),
                                     nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, dimension))
        self.transition = nn.Sequential(nn.Linear(dimension + actions, hidden), nn.ReLU(),
                                        nn.Linear(hidden, dimension))
        self.actions = actions
        self.diameter = diameter

    def encode(self, features):
        # Range permits average |Q(s)-Q(g)| up to 2*diameter.
        return self.diameter * torch.tanh(self.encoder(features))

    def predict_next(self, q, actions):
        a = torch.nn.functional.one_hot(actions, self.actions).to(q.dtype)
        return q + self.transition(torch.cat((q, a), dim=-1))


class MapReader:
    """Reusable frozen inference API. Needs features/rules, never an oracle."""
    def __init__(self, env, checkpoint, device="cpu"):
        saved = torch.load(checkpoint, map_location=device, weights_only=False)
        if saved["task"] != env.name or saved["actions"] != list(env.actions):
            raise ValueError("Checkpoint belongs to a different task/action convention")
        self.env, self.device = env, device
        self.model = QMap(env.feature_dim, len(env.actions), **saved["model_config"], diameter=env.diameter).to(device)
        self.model.load_state_dict(saved["model"])
        self.model.eval()

    @torch.no_grad()
    def encode(self, ids):
        return self.model.encode(torch.as_tensor(self.env.features(ids), device=self.device))

    @torch.no_grad()
    def scores(self, source, goal):
        return distance(self.encode(source), self.encode(goal)).cpu().numpy()

    def choose(self, state, goal):
        children = self.env.successors([state])[0]
        legal = children >= 0
        scores = self.scores(children[legal], [goal])
        action = int(torch.as_tensor(legal).nonzero().flatten()[int(scores.argmin())])
        return self.env.actions[action], int(children[action])
