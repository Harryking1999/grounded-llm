"""Board features and the checkpoint-compatible Q constructor."""
import numpy as np
import torch
from grounded_llm.core.qmap.model import StateEncoder

def board_bits(masks):
    """100 occupancy bits in the exact row-major order used by the rules."""
    values = [int(mask) for mask in masks]
    return torch.from_numpy(np.fromiter(((mask >> bit) & 1 for mask in values for bit in range(100)),
                                        dtype=np.float32, count=len(values) * 100).reshape(-1, 100))


class BoardEncoder(StateEncoder):
    """Historical constructor name; state_dict keys and outputs are preserved."""
