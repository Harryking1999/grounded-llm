"""Read-only statistics of a trained readout checkpoint on its own fixed training set.

This never trains and never touches the environment. It re-measures the same
examples the convergence check uses, then joins each per-example decision loss
with the candidate margin that produced its label, so a loss that sits at a
constant can be attributed to a margin regime or to a uniform failure.
"""

from argparse import ArgumentParser
from collections import Counter
import json
import statistics
from pathlib import Path

import torch
from torch.nn import functional as F

from .convergence import with_decision_mask
from .readout_aux import CounterfactualFirstTurnDataset
from .train import MapCollator, build_reader, to_device


def load_contract(config, manifest_path, source_root, q_checkpoint, model_path, batch_size):
    """Mirror train.py's identity contract so a mismatched checkpoint is refused."""
    task = config["task"]
    return {"config": config, "manifest": str(Path(manifest_path).resolve()),
            "map_source": str(Path(q_checkpoint if task == "blocks"
                                   else source_root).resolve()),
            "model_source": str(Path(model_path).resolve()) if model_path else config["model"],
            "batch_size": batch_size}


def candidate_pair(turns, index):
    """Recover the queried IDs from a matched swap pair; text alone cannot show them."""
    members = turns[index - index % 2:index - index % 2 + 2]
    chosen = {turn.chosen_id for turn in members}
    if len(chosen) != 2 or members[0].user_text != members[1].user_text:
        raise ValueError("A swap pair must be two answers to one identical question")
    left, right = sorted(chosen)
    distances = members[0].step.candidate_map_distances
    return left, right, abs(distances[left - 1] - distances[right - 1])


def measure(reader, collator, example, tokenizer, left, right, device):
    encoded, _ = example
    inputs = collator([example])
    map_batch = to_device(inputs.pop("map_batch"), device)
    inputs = {name: value.to(device) for name, value in inputs.items()}
    focus = inputs.pop("focus_mask")[:, 1:]
    labels = inputs["labels"][:, 1:]
    selected = focus & (labels != -100)
    if int(selected.sum()) < 1:
        raise ValueError("Decision positions are missing")
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16,
                                                enabled=torch.cuda.is_available()):
        logits = reader(map_batch, **inputs).logits.float()
    # selected indexes the shifted array, so logits[0, position] predicts labels[0, position].
    position = int(selected[0].nonzero()[0])
    row = {"decision_ce": float(F.cross_entropy(logits[:, :-1][selected], labels[selected])),
           "answer": tokenizer.decode(labels[0][position:position + 1]).strip(),
           "span_tokens": int(selected.sum())}
    # Which option does the model actually put its mass on at the first decision digit?
    probabilities = F.softmax(logits[0, position], dim=-1)
    top = int(probabilities.argmax())
    row["top_token"] = tokenizer.decode([top]).strip()
    ids = {label: tokenizer(str(label), add_special_tokens=False)["input_ids"]
           for label in (left, right)}
    if all(len(value) == 1 for value in ids.values()):
        row["p_left"], row["p_right"] = (float(probabilities[ids[left][0]]),
                                         float(probabilities[ids[right][0]]))
        row["top_in_pair"] = top in (ids[left][0], ids[right][0])
        row["mass_in_pair"] = row["p_left"] + row["p_right"]
    else:
        row["p_left"] = row["p_right"] = row["mass_in_pair"] = None
        row["top_in_pair"] = False
    return row


def summarize(rows):
    correct = [row["top_token"] == row["answer"] for row in rows]
    paired = [row for row in rows if row["mass_in_pair"] is not None]
    report = {"examples": len(rows),
              "single_token_answers": sum(row["span_tokens"] == 1 for row in rows),
              "decision_ce": statistics.mean(row["decision_ce"] for row in rows),
              "accuracy": sum(correct) / len(correct),
              "top_token_counts": dict(Counter(row["top_token"] for row in rows).most_common()),
              "top_in_pair_rate": statistics.mean(bool(row["top_in_pair"]) for row in rows),
              "mean_mass_in_pair": (statistics.mean(row["mass_in_pair"] for row in paired)
                                    if paired else None),
              "mean_p_correct_minus_wrong": (statistics.mean(row["p_correct"] - row["p_wrong"]
                                                            for row in paired) if paired else None)}
    return report


def by_margin(rows, buckets=5):
    ordered = sorted(rows, key=lambda row: row["margin"])
    size = max(1, len(ordered) // buckets)
    strata = []
    for start in range(0, len(ordered), size):
        chunk = [row for row in ordered[start:start + size]
                 if row["mass_in_pair"] is not None]
        if not chunk:
            continue
        strata.append({"n": len(chunk),
                       "margin_min": chunk[0]["margin"], "margin_max": chunk[-1]["margin"],
                       "accuracy": sum(row["top_token"] == row["answer"] for row in chunk) / len(chunk),
                       "decision_ce": statistics.mean(row["decision_ce"] for row in chunk),
                       "mean_p_correct_minus_wrong": statistics.mean(
                           row["p_correct"] - row["p_wrong"] for row in chunk)})
    return strata


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
    parser.add_argument("--limit", type=int, default=0)
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
    total = args.limit or len(dataset)
    rows = []
    for index in range(total):
        example = with_decision_mask(dataset[index], tokenizer)
        left, right, margin = candidate_pair(dataset.turns, index)
        row = measure(reader, collator, example, tokenizer, left, right, device)
        row.update(index=index, pair=[left, right], margin=margin,
                   correct_id=dataset.turns[index].chosen_id)
        row["p_correct"] = row["p_left"] if row["correct_id"] == left else row["p_right"]
        row["p_wrong"] = row["p_right"] if row["correct_id"] == left else row["p_left"]
        rows.append(row)
        if index % 128 == 0:
            print(f"measured {index}/{total}", flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "examples.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    report = {"checkpoint": str(args.checkpoint), "manifest": str(args.manifest),
              "overall": summarize(rows), "by_margin_quintile": by_margin(rows)}
    (args.out / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
