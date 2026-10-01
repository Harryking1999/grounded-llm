"""Task-independent English wording for both map-reader studies."""


def task_opening(task_description: str) -> str:
    return (f"{task_description}\n"
            "Reaching the goal is the first priority; among valid solutions, prefer fewer moves.")


def terminal_text(state_description: str, actions: str, move_count: int,
                  move_name: str) -> str:
    return "\n".join((
        f"The current {state_description} matches the goal.",
        f"Executed actions: {actions}.",
        f"Summary: Reached the goal after {move_count} executed {move_name}.",
        "<done/>",
    ))
