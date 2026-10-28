"""Cube Q encoder, axis symmetry adapter and frozen reader."""
import torch
from grounded_llm.core.qmap.reader import EnvironmentQMap

from grounded_llm.core.qmap.model import StateEncoder
from grounded_llm.core.qmap.model import distance as shared_distance
from grounded_llm.core.qmap.model import soft_rank_loss


MODEL_TYPE = "pure_q_symmetric_blocks"


def distance(source, goal, metric="l1", group_size=8):
    return shared_distance(source, goal, metric, group_size)


class QMap(StateEncoder):
    def __init__(self, feature_dim, dimension, hidden, cap, metric="l1", group_size=8, symmetry=None,
                 inverse_symmetry=False):
        if metric not in ("l1", "linf", "l2", "max_l1"):
            raise ValueError(f"Unknown Q distance: {metric}")
        if metric == "max_l1" and (group_size < 1 or dimension % group_size):
            raise ValueError("Q dimension must be divisible by the positive group size")
        super().__init__(dimension, hidden, cap, metric, input_dim=feature_dim, group_size=group_size)
        if symmetry not in (None, "cube_axis_permutation_average"):
            raise ValueError(f"Unknown Q symmetry: {symmetry}")
        self.symmetry = symmetry
        self.inverse_symmetry = inverse_symmetry
        if inverse_symmetry:
            if feature_dim != 192:raise ValueError('Cube inverse projection requires192corner features')
            from grounded_llm.tasks.cube2.symmetry import inverse_feature_indices
            self.register_buffer('inverse_feature_gather',torch.as_tensor(inverse_feature_indices()),persistent=False)
        if symmetry is not None:
            if feature_dim != 192 or dimension != 128 or metric != "linf":
                raise ValueError("Cube axis projection requires 192 inputs, 128 Q channels and Linf")
            from grounded_llm.tasks.cube2.symmetry import axis_projection_indices
            feature, align, action = axis_projection_indices()
            for name, value in (("axis_feature_gather", feature), ("axis_output_align", align),
                                ("axis_output_action", action)):
                self.register_buffer(name, torch.as_tensor(value), persistent=False)

    def encode(self, bits):
        direct=self._encode_axes(bits)
        if self.inverse_symmetry:
            direct=.5*(direct+self._encode_axes(bits[...,self.inverse_feature_gather]))
        return direct

    def _encode_axes(self,bits):
        if self.symmetry is None:
            return super().encode(bits)
        presented = bits[..., self.axis_feature_gather]
        encoded = super().encode(presented)
        aligned = encoded.gather(-1, self.axis_output_align.expand_as(encoded))
        return aligned.mean(dim=-2)


class MapReader(EnvironmentQMap):
    """Reusable frozen inference API. Needs features/rules, never an oracle."""
    def __init__(self, env, checkpoint, device="cpu", *, coordinate_frame="absolute", inverse_average=False):
        if coordinate_frame not in ("absolute", "goal_relative"):
            raise ValueError(f"Unknown cube coordinate frame: {coordinate_frame}")
        if inverse_average and coordinate_frame != 'goal_relative':
            raise ValueError('Inverse averaging requires goal-relative coordinates')
        saved = torch.load(checkpoint, map_location=device, weights_only=False)
        if saved.get("model_type") != MODEL_TYPE:
            raise ValueError("Expected a pure-Q checkpoint")
        if saved["task"] != env.name or saved["actions"] != list(env.actions):
            raise ValueError("Checkpoint belongs to a different task/action convention")
        if saved['model_config'].get('inverse_symmetry') and coordinate_frame!='goal_relative':
            raise ValueError('Inverse-projected Q is a goal-relative restoration map')
        model = QMap(env.feature_dim, **saved["model_config"], cap=saved["cap"]).to(device)
        model.load_state_dict(saved["model"])
        super().__init__(env, model, device)
        self.coordinate_frame = coordinate_frame
        self.inverse_average = inverse_average

    def scores(self, source, goal):
        """Goal-relative scoring uses the known group action before Q encoding.

        Left multiplication by goal^-1 preserves the original action labels.
        encode() continues to expose the checkpoint's ordinary state vectors;
        this pair-conditioned option applies to scores() and choose().
        inverse_average additionally averages the two relative directions to
        enforce the reversible task's distance symmetry, using two Q queries.
        """
        if self.coordinate_frame == "goal_relative":
            source = self.env.relative(goal, source)
            goal = [0]
        scores = super().scores(source, goal)
        if self.inverse_average:
            scores = .5*(scores+super().scores(self.env.relative(source,0),[0]))
        return scores
