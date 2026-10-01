"""English blocks task and factual environment updates."""

from experiments.gcml_counterexamples.src import blocks as rules
from .blocks import action_text
from .text import task_opening, terminal_text as shared_terminal_text


def initial_prompt(start: int, goal: int) -> str:
    shapes = "; ".join(f"{i}={shape}" for i, shape in
                       enumerate(rules.CONFIG["blocks"]["shape_rows"]))
    opening = task_opening("Find a valid sequence of removals from the initial board to the goal board, as short as you can.")
    return f"""{opening}

Rules: The board is a 10x10 binary grid. 1 means occupied and 0 means empty.
An action remove(shape_id,row,col) removes one listed shape at its zero-based top-left anchor.
Every occupied shape cell must overlap a currently occupied board cell and lie inside the board.
Only those cells are removed. Empty shape cells impose no constraint.
Use only the listed orientations. Shapes may be reused; there is no gravity or inventory limit.
Reach the goal board exactly, including any occupied cells that must remain.
Shapes (/ separates rows): {shapes}

Initial board:
{rules.to_grid(start).replace('/', chr(10))}

Goal board:
{rules.to_grid(goal).replace('/', chr(10))}"""


def turn_prompt(step, executed_actions) -> str:
    candidates = "\n".join(f"{i}: {action_text(action)}"
                            for i, action in enumerate(step.candidate_actions, 1)) or "(none)"
    history = " -> ".join(action_text(action) for action in executed_actions) or "(none)"
    return f"""[Environment update]
Current board:
{rules.to_grid(step.current).replace('/', chr(10))}
Actual executed actions: {history}
Legal next moves:
{candidates}
Candidate numbers may change between turns; use only the numbers above.
[/Environment update]"""


def terminal_text(executed_actions) -> str:
    history = " -> ".join(action_text(action) for action in executed_actions) or "(none)"
    return shared_terminal_text("board", history, len(executed_actions), "removals")
