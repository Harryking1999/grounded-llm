"""Cube Q encoder, axis symmetry adapter and frozen reader."""
import torch

from experiments.blocks_distance_map.src.model import StateEncoder, distance as shared_distance, soft_rank_loss


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
        if symmetry not in (None, "cube_axis_permutation_average"):
            raise ValueError(f"Unknown Q symmetry: {symmetry}")
        self.symmetry = symmetry
        if symmetry is not None:
            if feature_dim != 192 or dimension != 128 or metric != "linf":
                raise ValueError("Cube axis projection requires 192 inputs, 128 Q channels and Linf")
            from experiments.cube2_qmap.src.symmetry import axis_projection_indices
            feature, align, action = axis_projection_indices()
            for name, value in (("axis_feature_gather", feature), ("axis_output_align", align),
                                ("axis_output_action", action)):
                self.register_buffer(name, torch.as_tensor(value), persistent=False)

    def encode(self, bits):
        if self.symmetry is None:
            return super().encode(bits)
        presented = bits[..., self.axis_feature_gather]
        encoded = super().encode(presented)
        aligned = encoded.gather(-1, self.axis_output_align.expand_as(encoded))
        return aligned.mean(dim=-2)


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
        self.model.eval().requires_grad_(False)

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
