"""Task-independent English wording for both map-reader studies."""

from collections.abc import Mapping


def chat_ids(tokenizer, messages, *, add_generation_prompt, **kwargs) -> list[int]:
    """Return token IDs across Transformers chat-template return formats."""
    encoded = tokenizer.apply_chat_template(messages, tokenize=True,
        add_generation_prompt=add_generation_prompt, **kwargs)
    if isinstance(encoded, Mapping):
        encoded = encoded["input_ids"]
    if hasattr(encoded, "tolist"):
        encoded = encoded.tolist()
    if not isinstance(encoded, list) or any(not isinstance(token, int) for token in encoded):
        raise TypeError("chat template must return one sequence of token IDs")
    return encoded


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
