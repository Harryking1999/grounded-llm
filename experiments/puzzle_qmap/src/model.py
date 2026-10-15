"""Goal-independent Q encoder with a checkpoint-defined symmetric distance."""
import torch

from experiments.blocks_distance_map.src.model import BoardEncoder, distance as blocks_distance


MODEL_TYPE = "pure_q_symmetric_blocks"


def distance(source, goal, metric="l1"):
    if metric == "l1":
        return (blocks_distance(source, goal, "directed_sum") +
                blocks_distance(goal, source, "directed_sum")) / 2
    if metric == "linf":
        return (source - goal).abs().amax(dim=-1)
    if metric == "l2":
        return (source - goal).norm(p=2, dim=-1)
    raise ValueError(f"Unknown Q distance: {metric}")


class QMap(BoardEncoder):
    def __init__(self, feature_dim, dimension, hidden, cap, metric="l1"):
        if metric not in ("l1", "linf", "l2"):
            raise ValueError(f"Unknown Q distance: {metric}")
        super().__init__(dimension, hidden, cap, "symmetric_directed_sum", input_dim=feature_dim)
        self.metric = metric

    def distance(self, source, goal):
        return distance(source, goal, self.metric)

    def forward(self, features, pairs):
        return self.distance(self.encode(features[pairs[:, 0]]), self.encode(features[pairs[:, 1]]))


def soft_rank_loss(scores, truth, valid, temperature):
    """Blocks' full soft-distance target, with padding excluded from both sides."""
    target = torch.softmax((-truth.float() / temperature).masked_fill(~valid, -torch.inf), dim=-1)
    log_prob = torch.log_softmax((-scores / temperature).masked_fill(~valid, -torch.inf), dim=-1)
    return -(target * log_prob.masked_fill(~valid, 0)).sum(dim=-1).mean()


class MapReader:
    """Reusable frozen inference API. Needs features/rules, never an oracle."""
    def __init__(self, env, checkpoint, device="cpu"):
        saved = torch.load(checkpoint, map_location=device, weights_only=False)
        if saved.get("model_type") != MODEL_TYPE:
            raise ValueError("Expected a pure-Q checkpoint")
        if saved["task"] != env.name or saved["actions"] != list(env.actions):
            raise ValueError("Checkpoint belongs to a different task/action convention")
        self.env, self.device = env, device
        self.model = QMap(env.feature_dim, **saved["model_config"], cap=saved["cap"]).to(device)
        self.model.load_state_dict(saved["model"])
        self.model.eval()

    @torch.no_grad()
    def encode(self, ids):
        return self.model.encode(torch.as_tensor(self.env.features(ids), device=self.device))

    @torch.no_grad()
    def scores(self, source, goal):
        return self.model.distance(self.encode(source), self.encode(goal)).cpu().numpy()

    def choose(self, state, goal):
        children = self.env.successors([state])[0]
        legal = children >= 0
        scores = self.scores(children[legal], [goal])
        action = int(torch.as_tensor(legal).nonzero().flatten()[int(scores.argmin())])
        return self.env.actions[action], int(children[action])
