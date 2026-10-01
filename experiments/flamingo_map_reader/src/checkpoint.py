"""Atomic adapter and optimizer checkpoints for resumable SFT runs."""

import os
from pathlib import Path
import random

import torch

from .fusion import MapReader


def save_checkpoint(
    path: Path,
    reader: MapReader,
    optimizer: torch.optim.Optimizer,
    step: int,
    config: dict,
    *,
    scheduler=None,
    data_state=None,
) -> None:
    if step < 0:
        raise ValueError("step must be nonnegative")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "step": step,
        "config": config,
        "adapter": reader.adapter_state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": None if scheduler is None else scheduler.state_dict(),
        "data_state": data_state,
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "python_rng": random.getstate(),
    }
    temporary = path.with_name(path.name + ".tmp")
    torch.save(state, temporary)
    os.replace(temporary, path)


def load_checkpoint(
    path: Path,
    reader: MapReader,
    optimizer: torch.optim.Optimizer,
    config: dict,
    *,
    scheduler=None,
) -> tuple[int, object]:
    # Run artifacts are created locally by this trainer; torch checkpoints are
    # pickle based and must never be loaded from an untrusted source.
    state = torch.load(path, map_location="cpu", weights_only=False)
    if state["config"] != config:
        raise ValueError("checkpoint configuration differs from the current run")
    if (state["scheduler"] is None) != (scheduler is None):
        raise ValueError("checkpoint scheduler setup differs from the current run")
    reader.load_adapter_state_dict(state["adapter"])
    optimizer.load_state_dict(state["optimizer"])
    if scheduler is not None:
        scheduler.load_state_dict(state["scheduler"])
    torch.set_rng_state(state["torch_rng"])
    if state["cuda_rng"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda_rng"])
    random.setstate(state["python_rng"])
    return int(state["step"]), state["data_state"]
