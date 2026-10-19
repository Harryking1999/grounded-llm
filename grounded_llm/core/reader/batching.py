"""Batch task-provided map steps without interpreting their rules."""
from collections.abc import Sequence
from dataclasses import replace
import torch
from torch.nn.utils.rnn import pad_sequence
from .memory import MapBatch, MapTimeline

def batch_maps(steps: list) -> MapBatch:
    if not steps:
        raise ValueError("cannot batch zero map steps")
    return MapBatch(
        vectors=pad_sequence([s.map_batch.vectors[0] for s in steps], batch_first=True),
        roles=pad_sequence([s.map_batch.roles[0] for s in steps], batch_first=True),
        candidate_ids=pad_sequence([s.map_batch.candidate_ids[0] for s in steps],
                                   batch_first=True),
        valid=pad_sequence([s.map_batch.valid[0] for s in steps], batch_first=True),
    )


def timeline_maps(steps: list, token_map_ids: Sequence[int]) -> MapTimeline:
    """Keep every turn's map; each text token reads only its assigned turn."""
    timeline = MapTimeline(
        snapshots=batch_maps(steps),
        token_map_ids=torch.tensor([list(token_map_ids)], dtype=torch.long),
    )
    timeline.validate(max(len(step.candidate_actions) for step in steps),
                      len(token_map_ids))
    return timeline

def to_device(timeline, device):
    maps = timeline.snapshots
    return replace(timeline, snapshots=MapBatch(
        maps.vectors.to(device), maps.roles.to(device),
        maps.candidate_ids.to(device), maps.valid.to(device)),
        token_map_ids=timeline.token_map_ids.to(device))


def collate_examples(examples, pad_token_id):
    """Pad complete SFT examples; keep each example's own map timeline."""
    width = max(len(encoded.input_ids) for encoded, _ in examples)
    input_ids, labels, attention_mask, map_ids, counts, steps = [], [], [], [], [], []
    focused = []
    for encoded, example_steps in examples:
        padding = width - len(encoded.input_ids)
        input_ids.append(encoded.input_ids + [pad_token_id] * padding)
        labels.append(encoded.labels + [-100] * padding)
        attention_mask.append([1] * len(encoded.input_ids) + [0] * padding)
        map_ids.append(encoded.token_map_ids + [len(example_steps) - 1] * padding)
        if encoded.focus_mask is not None:
            focused.append(encoded.focus_mask + [False] * padding)
        counts.append(len(example_steps))
        steps.extend(example_steps)
    timeline = MapTimeline(batch_maps(steps), torch.tensor(map_ids), tuple(counts))
    timeline.validate(max(len(step.candidate_actions) for step in steps), width)
    if focused and len(focused) != len(examples):
        raise ValueError("Cannot mix focused and ordinary trajectories")
    inputs = {"input_ids": torch.tensor(input_ids), "labels": torch.tensor(labels),
              "attention_mask": torch.tensor(attention_mask)}
    if focused:
        inputs["focus_mask"] = torch.tensor(focused, dtype=torch.bool)
    return timeline, inputs

def pseudo_timeline(timeline):
    """Nonzero deterministic slot vectors without state/goal information."""
    batch=timeline.snapshots
    slot=torch.arange(batch.vectors.shape[1],device=batch.vectors.device)[:,None]+1
    dim=torch.arange(batch.vectors.shape[2],device=batch.vectors.device)[None,:]+1
    vectors=torch.sin(slot*dim*.017).unsqueeze(0).expand(batch.vectors.shape[0],-1,-1).clone()
    return replace(timeline,snapshots=replace(batch,vectors=vectors))
