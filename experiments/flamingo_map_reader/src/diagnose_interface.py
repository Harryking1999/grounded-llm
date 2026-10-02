"""Per-layer statistics of what the map interface delivers to the decision token.

A swap pair has identical text, identical roles and identical candidate IDs; only the
two queried value vectors are exchanged. Two facts therefore decide whether the pair is
answerable at all, and this instrument measures both at the decision position:

  * how much of the map read survives into the residual stream, and
  * whether attention addresses the two queried slots unequally.

The adapter's own output already contains the first quantity: its forward returns
`hidden + tanh(gate) * read`, so the hook subtracts its input to obtain that term
exactly. Only the attention distribution is recomputed, and it is checked against that
term before being trusted.
"""

from argparse import ArgumentParser
from collections import defaultdict
import json
import math
from pathlib import Path

import torch
from torch.nn import functional as F

from .convergence import with_decision_mask
from .diagnose_readout import candidate_pair, load_contract
from .readout_aux import CounterfactualFirstTurnDataset
from .train import MapCollator, build_reader, to_device


class DecisionPositionRecorder:
    """Record adapter contributions at one fixed text position, layer by layer."""

    def __init__(self, reader, position):
        self.reader, self.position = reader, position
        self.adapters, self.residuals, self.handles = [], [], []

    def __enter__(self):
        for index, layer in enumerate(self.reader.conditioned_layers):
            self.handles.append(layer.map_attention.register_forward_hook(self.adapter(index)))
            self.handles.append(layer.register_forward_hook(self.residual(index)))
        return self

    def __exit__(self, *exception):
        for handle in self.handles:
            handle.remove()
        self.handles = []
        return False

    def adapter(self, index):
        def hook(module, args, output):
            hidden, memory, valid = args[0], args[1], args[2]
            token_map_ids = args[3] if len(args) > 3 else None
            if hidden.shape[1] <= self.position:
                return
            position = self.position
            delta = (output - hidden)[0, position]
            weights, reproduced = self.attention(module, hidden, memory, valid,
                                                 token_map_ids, position)
            self.adapters.append({
                "layer": index,
                "gate": float(torch.tanh(module.gate)),
                "delta_over_hidden": float(delta.float().norm() /
                                           (hidden[0, position].float().norm() + 1e-12)),
                "delta_rounded_away": bool(torch.equal(output[0, position], hidden[0, position])),
                "attention": [float(value) for value in weights.mean(dim=0)],
                "per_head": [[float(value) for value in head] for head in weights],
                "reproduction_error": float((reproduced - delta.float()).norm() /
                                            (delta.float().norm() + 1e-12)),
            })
        return hook

    def residual(self, index):
        def hook(module, args, output):
            hidden = output[0] if isinstance(output, tuple) else output
            if hidden.shape[1] > self.position:
                self.residuals.append([float(value) for value in
                                       hidden[0, self.position].float()])
        return hook

    @staticmethod
    def attention(module, hidden, memory, valid, token_map_ids, position):
        """Recompute the addressed read; mirrors GatedMapCrossAttention.forward."""
        keys, values = memory.keys, memory.values
        snapshots, slots = keys.shape[1], keys.shape[2]
        snapshot_ids = torch.arange(snapshots, device=keys.device).repeat_interleave(slots)
        mask = (valid.reshape(1, -1)[:, None, :] &
                (token_map_ids[:, :, None] == snapshot_ids[None, None, :]))[:, None, :, :]
        keys = keys.reshape(1, snapshots * slots, -1)
        values = values.reshape(1, snapshots * slots, -1)
        query = module.to_query(module.query_norm(hidden))
        key = module.to_key(module.key_norm(keys))
        value = module.to_value(values) * module.value_scale
        batch, length = query.shape[0], query.shape[1]
        query = query.reshape(batch, length, module.heads, module.head_dim)
        key = key.reshape(batch, -1, module.heads, module.head_dim)
        value = value.reshape(batch, -1, module.heads, module.head_dim)
        # Einsum keeps the head axis explicit; a positional slice plus "@" would let
        # torch read the heads as a batch of matrices and broadcast them into positions.
        scores = torch.einsum("bhd,bmhd->bhm", query[:, position].float(),
                              key.float()) / math.sqrt(module.head_dim)
        scores = scores.masked_fill(~mask[:, :, position], float("-inf"))
        weights = F.softmax(scores, dim=-1)
        read = torch.einsum("bhm,bmhd->bhd", weights.to(value.dtype), value)
        read = read.reshape(module.heads * module.head_dim)
        # The adapter adds tanh(gate) * read, so the check must include that factor.
        reproduced = module.to_output(read.to(hidden.dtype)) * torch.tanh(module.gate)
        return weights[0], reproduced


def decision_position(example):
    """Index whose residual produces the first supervised decision token.

    Labels are read shifted by one, so the token at absolute index t is predicted from
    hidden state t-1; the shift is applied here once so both agree.
    """
    encoded, _ = example
    positions = [index for index, (marked, label) in
                 enumerate(zip(encoded.focus_mask[1:], encoded.labels[1:]))
                 if marked and label != -100]
    if not positions:
        raise ValueError("Example has no decision position")
    return positions[0]


def slots_for(left, right):
    """Slot layout is current, goal, then candidates in ID order."""
    return {"current": 0, "goal": 1, "left": left + 1, "right": right + 1}


def summarize(records):
    layers = sorted({row["layer"] for row in records})
    summary = []
    for layer in layers:
        rows = [row for row in records if row["layer"] == layer]
        summary.append({
            "layer": layer,
            "gate": sum(row["gate"] for row in rows) / len(rows),
            "delta_over_hidden": sum(row["delta_over_hidden"] for row in rows) / len(rows),
            "rounding_loss_rate": sum(row["delta_rounded_away"] for row in rows) / len(rows),
            "reproduction_error_max": max(row["reproduction_error"] for row in rows),
            "attention_correct": sum(row["attention_correct"] for row in rows) / len(rows),
            "attention_wrong": sum(row["attention_wrong"] for row in rows) / len(rows),
            "attention_current": sum(row["attention_current"] for row in rows) / len(rows),
            "attention_goal": sum(row["attention_goal"] for row in rows) / len(rows),
            "asymmetry_signed": sum(row["asymmetry_signed"] for row in rows) / len(rows),
            "asymmetry_absolute": sum(row["asymmetry_absolute"] for row in rows) / len(rows),
            "swap_relative_change": sum(row["swap_relative_change"] for row in rows) / len(rows),
            "gap_slot_left_correct": mean_of(rows, "gap_slot", "left"),
            "gap_slot_right_correct": mean_of(rows, "gap_slot", "right"),
            "n": len(rows),
        })
    return summary


def mean_of(rows, key, correct=None):
    chosen = [row[key] for row in rows if correct is None or row["correct"] == correct]
    return sum(chosen) / len(chosen) if chosen else None


def main():
    parser = ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--q-checkpoint", type=Path)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--pairs", type=int, default=32)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if manifest["config"] != config:
        raise ValueError("Manifest and diagnostic configurations differ")
    contract = load_contract(config, args.manifest, args.source_root, args.q_checkpoint,
                             args.model_path, args.batch_size)
    saved = torch.load(args.checkpoint / "adapter.pt", map_location="cpu", weights_only=True)
    if saved["contract"] != contract:
        raise ValueError("Checkpoint was trained under a different contract")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model_source = str(args.model_path) if args.model_path else config["model"]
    tokenizer = AutoTokenizer.from_pretrained(model_source)
    base = AutoModelForCausalLM.from_pretrained(model_source,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32)
    base.config.use_cache = False
    reader = build_reader(base, config)
    reader.load_adapter_state_dict(saved["adapter"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    reader.to(device).eval()
    records = [record for record in manifest["records"] if record["split"] == "train"]
    dataset = CounterfactualFirstTurnDataset(records, None, tokenizer, config, args.source_root)
    collator = MapCollator(tokenizer.pad_token_id if tokenizer.pad_token_id is not None
                           else tokenizer.eos_token_id)
    collected = defaultdict(list)
    pair_count = min(args.pairs, len(dataset) // 2)
    for pair_index in range(pair_count):
        members = []
        for offset in (0, 1):
            index = pair_index * 2 + offset
            example = with_decision_mask(dataset[index], tokenizer)
            left, right, margin = candidate_pair(dataset.turns, index)
            position = decision_position(example)
            inputs = collator([example])
            map_batch = to_device(inputs.pop("map_batch"), device)
            inputs = {name: value.to(device) for name, value in inputs.items()}
            inputs.pop("focus_mask")
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16,
                                                        enabled=torch.cuda.is_available()), \
                    DecisionPositionRecorder(reader, position) as recorder:
                reader(map_batch, **inputs)
            correct = "left" if dataset.turns[index].chosen_id == left else "right"
            wrong = "right" if correct == "left" else "left"
            slots = slots_for(left, right)
            rows = {row["layer"]: row for row in recorder.adapters}
            for layer, row in rows.items():
                heads = row.pop("per_head")
                # Heads can be asymmetric in opposite directions and cancel in the mean,
                # so report the per-head gap as well as its signed average.
                gaps = [head[slots[correct]] - head[slots[wrong]] for head in heads]
                mean_attention = [sum(head[slot] for head in heads) / len(heads)
                                  for slot in range(len(heads[0]))]
                # Slot-ordered gap: a fixed positional preference gives the same value
                # whichever side is correct, an answer-tracking one flips its sign.
                row["gap_slot"] = sum(head[slots["left"]] - head[slots["right"]]
                                      for head in heads) / len(heads)
                row["correct"] = correct
                row["attention"] = mean_attention
                row.update(
                    attention_correct=sum(head[slots[correct]] for head in heads) / len(heads),
                    attention_wrong=sum(head[slots[wrong]] for head in heads) / len(heads),
                    attention_current=mean_attention[slots["current"]],
                    attention_goal=mean_attention[slots["goal"]],
                    asymmetry_signed=sum(gaps) / len(gaps),
                    asymmetry_absolute=sum(abs(gap) for gap in gaps) / len(gaps))
            members.append((rows, recorder.residuals, margin, correct))
        for layer in members[0][0]:
            first, second = members[0][0][layer], members[1][0][layer]
            hidden = members[0][1][layer]
            swapped = members[1][1][layer]
            change = math.dist(hidden, swapped) / (math.dist(hidden, [0.0] * len(hidden)) + 1e-12)
            for row in (first, second):
                row["swap_relative_change"] = change
        for rows, _, margin, _ in members:
            for layer, row in rows.items():
                row["margin"] = margin
                collected[layer].append(row)
        if pair_index % 8 == 0:
            print(f"measured {pair_index}/{pair_count} pairs", flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "layers.jsonl").open("w", encoding="utf-8") as handle:
        for layer in sorted(collected):
            for row in collected[layer]:
                handle.write(json.dumps(row) + "\n")
    report = {"checkpoint": str(args.checkpoint), "pairs": pair_count,
              "layers": summarize([row for rows in collected.values() for row in rows])}
    (args.out / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["layers"], indent=2))


if __name__ == "__main__":
    main()
