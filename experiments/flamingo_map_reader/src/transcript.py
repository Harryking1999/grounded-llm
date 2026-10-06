"""Encode one complete ReAct-style trajectory with per-turn map assignments."""

from dataclasses import dataclass
import re

from .sft import Demonstration
from .text import chat_ids


@dataclass(frozen=True)
class EncodedTrajectory:
    input_ids: list[int]
    labels: list[int]
    token_map_ids: list[int]
    answer_tokens: int
    focus_mask: list[bool] | None = None


def decision_focus(answer: str, offsets: list[tuple[int, int]]) -> list[bool]:
    """Mark ranking contents and the chosen ID, leaving boilerplate at unit weight."""
    ranking = re.search(r"(?m)^Map-distance ranking to the goal, closest to farthest: (.+)\.$", answer)
    action = re.search(r"<action>([1-9][0-9]*)</action>", answer)
    if bool(ranking) != bool(action):
        raise ValueError("Decision answer needs both a ranking and an action")
    if not ranking:
        return [False] * len(offsets)
    spans = ((ranking.start(1), ranking.end(1)), (action.start(1), action.end(1)))
    focused = [any(start < right and end > left for left, right in spans)
               for start, end in offsets]
    if not any(focused):
        raise ValueError("Decision answer has no focused tokens")
    return focused


def encode_trajectory(demonstration: Demonstration, tokenizer,
                      max_tokens: int, *, chat_template_kwargs=None,
                      focus_decisions=False) -> EncodedTrajectory:
    """Mask environment text; every token reads the map of its own turn."""
    if not (demonstration.success or demonstration.no_solution) or not demonstration.turns:
        raise ValueError("SFT requires a complete successful trajectory")
    messages = []
    previous: list[int] = []
    labels: list[int] = []
    map_ids: list[int] = []
    answer_tokens = 0
    focus_mask: list[bool] | None = [] if focus_decisions else None
    template_kwargs = chat_template_kwargs or {}
    for turn_index, turn in enumerate(demonstration.turns):
        messages.append({"role": "user", "content": turn.user_text})
        prefix = chat_ids(tokenizer, messages, add_generation_prompt=True,
                          **template_kwargs)
        if prefix[:len(previous)] != previous:
            raise ValueError("chat template changed an earlier conversation prefix")
        messages.append({"role": "assistant", "content": turn.answer_text})
        complete = chat_ids(tokenizer, messages, add_generation_prompt=False,
                            **template_kwargs)
        if complete[:len(prefix)] != prefix:
            raise ValueError("chat template does not preserve the generation prefix")
        new_answer = len(complete) - len(prefix)
        if new_answer == 0:
            raise ValueError("assistant answer produced no supervised tokens")
        labels.extend([-100] * (len(prefix) - len(previous)))
        labels.extend(complete[len(prefix):] if turn.supervise else [-100] * new_answer)
        if focus_mask is not None:
            content = tokenizer(turn.answer_text, add_special_tokens=False,
                                return_offsets_mapping=True)
            content_ids = content["input_ids"]
            if complete[len(prefix):len(prefix) + len(content_ids)] != content_ids:
                raise ValueError("Answer content does not align with the chat template")
            if len(content["offset_mapping"]) != len(content_ids):
                raise ValueError("Answer offsets and tokens have different lengths")
            focused = decision_focus(turn.answer_text, content["offset_mapping"])
            focus_mask.extend([False] * (len(prefix) - len(previous)))
            focus_mask.extend(focused if turn.supervise else [False] * len(focused))
            focus_mask.extend([False] * (new_answer - len(content_ids)))
        map_ids.extend([turn_index] * (len(complete) - len(previous)))
        answer_tokens += new_answer if turn.supervise else 0
        previous = complete
        if len(previous) > max_tokens:
            raise ValueError(f"trajectory has {len(previous)} tokens, exceeding {max_tokens}")
    if (len(previous) != len(labels) or len(previous) != len(map_ids) or
            (focus_mask is not None and len(previous) != len(focus_mask))):
        raise AssertionError("trajectory token annotations are misaligned")
    return EncodedTrajectory(previous, labels, map_ids, answer_tokens, focus_mask)
