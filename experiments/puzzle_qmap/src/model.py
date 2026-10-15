"""Only Q: reuse the blocks encoder and symmetrize its directed_sum metric."""
import torch

from experiments.blocks_distance_map.src.model import BoardEncoder, distance as blocks_distance


MODEL_TYPE = "pure_q_symmetric_blocks"


def distance(source, goal):
    return (blocks_distance(source, goal, "directed_sum") +
            blocks_distance(goal, source, "directed_sum")) / 2


class QMap(BoardEncoder):
    def __init__(self, feature_dim, dimension, hidden, cap):
        super().__init__(dimension, hidden, cap, "symmetric_directed_sum", input_dim=feature_dim)

    def forward(self, features, pairs):
        return distance(self.encode(features[pairs[:, 0]]), self.encode(features[pairs[:, 1]]))


def soft_rank_loss(scores, truth, valid, temperature):
    """Blocks' full soft-distance target, with padding excluded from both sides."""
    target = torch.softmax((-truth.float() / temperature).masked_fill(~valid, -torch.inf), dim=-1)
    log_prob = torch.log_softmax((-scores / temperature).masked_fill(~valid, -torch.inf), dim=-1)
    return -(target * log_prob.masked_fill(~valid, 0)).sum(dim=-1).mean()


def edge_difference_loss(parent_scores, child_scores, parent_truth, child_truth, valid):
    """Supervise goal-distance change on every actual action, in move units."""
    prediction = child_scores - parent_scores[:, None]
    target = child_truth.float() - parent_truth.float()[:, None]
    return torch.nn.functional.smooth_l1_loss(prediction[valid], target[valid])


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
        return distance(self.encode(source), self.encode(goal)).cpu().numpy()

    def choose(self, state, goal):
        children = self.env.successors([state])[0]
        legal = children >= 0
        scores = self.scores(children[legal], [goal])
        action = int(torch.as_tensor(legal).nonzero().flatten()[int(scores.argmin())])
        return self.env.actions[action], int(children[action])
