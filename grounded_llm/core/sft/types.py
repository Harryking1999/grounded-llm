"""Task-authored text and map steps consumed by the shared SFT pipeline."""
from dataclasses import dataclass
from typing import Any

@dataclass(frozen=True)
class SupervisedTurn:
    user_text: str
    answer_text: str
    step: Any
    executed_path: tuple[int, ...]
    chosen_id: int | None
    supervise: bool = True


@dataclass(frozen=True)
class Demonstration:
    turns: tuple[SupervisedTurn, ...]
    executed_path: tuple[int, ...]
    success: bool
    no_solution: bool = False
