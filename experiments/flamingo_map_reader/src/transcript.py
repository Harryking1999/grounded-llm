"""Encode one complete ReAct-style trajectory with per-turn map assignments."""

from dataclasses import dataclass

from .sft import Demonstration


@dataclass(frozen=True)
class EncodedTrajectory:
    input_ids: list[int]
    labels: list[int]
    token_map_ids: list[int]
    answer_tokens: int


def encode_trajectory(demonstration: Demonstration, tokenizer,
                      max_tokens: int, *, chat_template_kwargs=None) -> EncodedTrajectory:
    """Mask environment text; every token reads the map of its own turn."""
    if not demonstration.success or not demonstration.turns:
        raise ValueError("SFT requires a complete successful trajectory")
    messages = []
    previous: list[int] = []
    labels: list[int] = []
    map_ids: list[int] = []
    answer_tokens = 0
    template_kwargs = chat_template_kwargs or {}
    for turn_index, turn in enumerate(demonstration.turns):
        messages.append({"role": "user", "content": turn.user_text})
        prefix = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            **template_kwargs,
        )
        if prefix[:len(previous)] != previous:
            raise ValueError("chat template changed an earlier conversation prefix")
        messages.append({"role": "assistant", "content": turn.answer_text})
        complete = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=False,
            **template_kwargs,
        )
        if complete[:len(prefix)] != prefix:
            raise ValueError("chat template does not preserve the generation prefix")
        new_answer = len(complete) - len(prefix)
        if new_answer == 0:
            raise ValueError("assistant answer produced no supervised tokens")
        labels.extend([-100] * (len(prefix) - len(previous)))
        labels.extend(complete[len(prefix):])
        map_ids.extend([turn_index] * (len(complete) - len(previous)))
        answer_tokens += new_answer
        previous = complete
        if len(previous) > max_tokens:
            raise ValueError(f"trajectory has {len(previous)} tokens, exceeding {max_tokens}")
    if len(previous) != len(labels) or len(previous) != len(map_ids):
        raise AssertionError("trajectory token annotations are misaligned")
    return EncodedTrajectory(previous, labels, map_ids, answer_tokens)
