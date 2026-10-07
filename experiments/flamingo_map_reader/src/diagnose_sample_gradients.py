"""Measure sample gradients on checkpoint copies without optimizer updates."""

from argparse import ArgumentParser
from collections import defaultdict
import json
from pathlib import Path
import random
import time

import numpy as np
import torch


def sample_group(record):
    if record.get("no_solution"):
        return "failure_empty_goal" if int(record["goal"]) == 0 else "failure_nonempty_goal"
    return "initial_goal" if record["action_count"] == 0 else "success_trajectory"


def select_samples(records, spec):
    rng = random.Random(spec["seed"])
    selected = []
    for group, count in spec["samples_by_group"].items():
        pool = [r for r in records if r["split"] == "train" and sample_group(r) == group]
        for record in rng.sample(pool, count):
            selected.append((record, rng.randrange(record["variants"])))
    return selected


def parameter_group(name):
    if name.startswith("memory_encoder."):
        return "feature_ffn" if ".feature_ffn." in name else "memory_other"
    if ".map_attention." not in name:
        raise ValueError(f"Unexpected trainable parameter: {name}")
    component = name.split(".map_attention.")[1].split(".")[0]
    return "attention_" + (component if component.startswith("to_") or component == "gate" else "norm")


def gradient_vector(named_parameters):
    pieces, squared = [], defaultdict(list)
    for name, parameter in named_parameters:
        if parameter.grad is None:
            raise ValueError(f"Missing gradient: {name}")
        gradient = parameter.grad.detach().float().reshape(-1)
        pieces.append(gradient)
        squared[parameter_group(name)].append(torch.sum(gradient.square()))
    vector = torch.cat(pieces)
    if not bool(torch.isfinite(vector).all()):
        raise ValueError("Nonfinite raw gradient")
    groups = {group: torch.stack(values).sum().sqrt().item() for group, values in squared.items()}
    return vector, groups


def batch_measurement(gram, indices, clip_norm):
    """Use the Gram matrix to measure the mean of independent sample gradients."""
    sub = gram[np.ix_(indices, indices)]
    squared_sum = max(float(sub.sum()), 0.0)
    norm = squared_sum ** 0.5 / len(indices)
    return {
        "sample_indices": indices,
        "raw_mean_norm": norm,
        "clip_scale": min(1.0, clip_norm / (norm + 1e-6)),
        # These are signed projections, not nonnegative percentages.
        "projection_shares": (sub.sum(axis=1) / squared_sum).tolist() if squared_sum else None,
    }


def summarize(rows, clip_norm):
    output = {}
    for group in sorted({r["group"] for r in rows}):
        values = [r for r in rows if r["group"] == group]
        norms = np.array([r["raw_norm"] for r in values])
        module_names = values[0]["module_norms"]
        output[group] = {
            "samples": len(values),
            "mean_loss": float(np.mean([r["loss"] for r in values])),
            "mean_supervised_tokens": float(np.mean([r["supervised_tokens"] for r in values])),
            "norm_median": float(np.median(norms)),
            "norm_p95": float(np.percentile(norms, 95)),
            "norm_max": float(norms.max()),
            "fraction_above_clip_if_alone": float(np.mean(norms > clip_norm)),
            "median_module_norms": {name: float(np.median([r["module_norms"][name] for r in values]))
                                    for name in module_names},
        }
    return output


def synthetic_batches(gram, rows, clip_norm, mixed_count):
    by_group = defaultdict(list)
    for index, row in enumerate(rows):
        by_group[row["group"]].append(index)
    success, initial = by_group["success_trajectory"], by_group["initial_goal"]
    failure = by_group["failure_nonempty_goal"] + by_group["failure_empty_goal"]
    result = {}
    for name, indices in (("four_success", success), ("four_initial_goal", initial),
                          ("four_failure", failure)):
        result[name] = [batch_measurement(gram, indices[start:start + 4], clip_norm)
                        for start in range(0, len(indices) - 3, 4)]
    for name, others in (("three_success_one_failure", failure),
                         ("three_success_one_initial_goal", initial)):
        result[name] = []
        for index in range(min(mixed_count, len(others))):
            chosen = [success[(index * 3 + offset) % len(success)] for offset in range(3)]
            result[name].append(batch_measurement(gram, chosen + [others[index]], clip_norm))
    return result


def main():
    parser = ArgumentParser()
    for name in ("spec", "run", "model-path", "out"):
        parser.add_argument("--" + name, required=True, type=Path)
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError("Use a new diagnostic output directory")
    spec = json.loads(args.spec.read_text())
    if args.run.name != spec["training_run"]:
        raise ValueError("Diagnostic spec belongs to another training run")
    manifest_path = args.run / "data/manifest.json"
    manifest = json.loads(manifest_path.read_text())
    selected = select_samples(manifest["records"], spec)
    torch.set_num_threads(1)
    if not torch.cuda.is_available():
        raise ValueError("Real-model sample diagnostic requires a CUDA device")
    device = torch.device("cuda:0")

    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
    from .train import build_reader, collate_examples, to_device
    from .trajectory_dataset import PreparedTrajectoryDataset

    first_checkpoint = args.run / f"training/models/checkpoint-{spec['checkpoint_steps'][0]}"
    first_saved = torch.load(first_checkpoint / "adapter.pt", map_location="cpu", weights_only=True)
    config = first_saved["contract"]["config"]
    if config["training"].get("decision_focus_weight", 1) != 1:
        raise ValueError("Diagnostic currently supports ordinary assistant-token CE")
    if config["training"]["world_size"] != 4:
        raise ValueError("Synthetic batch comparison assumes the current four-rank run")
    clip_norm = config["training"]["gradient_clip_norm"]
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    set_seed(spec["seed"])
    base = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=torch.bfloat16)
    base.config.use_cache = False
    reader = build_reader(base, config).to(device).train()
    parameters = [(name, p) for name, p in reader.named_parameters() if p.requires_grad]
    args.out.mkdir(parents=True)
    result = {
        "spec": spec,
        "trainable_parameters": sum(p.numel() for _, p in parameters),
        "clip_norm": clip_norm,
        "method": "Full cached training contexts, per-sample supervised-token mean CE, bf16 autocast, training mode, no clipping or optimizer step. Synthetic four-rank gradient is the arithmetic mean of four sample vectors, not token-weighted pooled CE. Stratified exploratory samples are not representative of training frequencies.",
        "checkpoints": {},
    }
    for step in spec["checkpoint_steps"]:
        checkpoint = args.run / f"training/models/checkpoint-{step}"
        ready = json.loads((checkpoint / "evaluation_ready.json").read_text())
        saved = torch.load(checkpoint / "adapter.pt", map_location="cpu", weights_only=True)
        if saved["contract"] != first_saved["contract"]:
            raise ValueError("Checkpoint training contracts differ")
        reader.load_adapter_state_dict(saved["adapter"])
        rows, vectors = [], []
        started = time.monotonic()
        for index, (record, variant) in enumerate(selected):
            reader.zero_grad(set_to_none=True)
            set_seed(spec["seed"] + index)
            example = PreparedTrajectoryDataset(manifest_path, [record], config)[variant]
            timeline, inputs = collate_examples([example], tokenizer.eos_token_id)
            timeline = to_device(timeline, device)
            inputs = {key: value.to(device) for key, value in inputs.items()}
            supervised_tokens = int((inputs["labels"][:, 1:] != -100).sum())
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output = reader(map_batch=timeline, **inputs)
            output.loss.backward()
            vector, modules = gradient_vector(parameters)
            if any(p.grad is not None for p in reader.parameters() if not p.requires_grad):
                raise ValueError("Frozen parameter received a gradient")
            row = dict(index=index, trajectory_id=record["trajectory_id"], variant=variant,
                       group=sample_group(record), goal_empty=int(record["goal"]) == 0,
                       sequence_tokens=inputs["input_ids"].shape[1], supervised_tokens=supervised_tokens,
                       action_count=record["action_count"], loss=output.loss.item(),
                       raw_norm=vector.norm().item(), module_norms=modules)
            rows.append(row)
            vectors.append(vector)
            with (args.out / "samples.jsonl").open("a") as handle:
                handle.write(json.dumps(dict(step=step, **row)) + "\n")
            print(json.dumps(dict(step=step, sample=index, group=row["group"],
                                  loss=row["loss"], raw_norm=row["raw_norm"])), flush=True)
            del output, inputs, timeline, example, vector
        matrix = torch.stack(vectors)
        # Disable TF32 for the dot products underlying norm and directional estimates.
        previous_tf32 = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        try:
            gram = (matrix @ matrix.T).cpu().numpy().astype(np.float64)
        finally:
            torch.backends.cuda.matmul.allow_tf32 = previous_tf32
        diagonal = np.sqrt(np.maximum(np.diag(gram), 0))
        if not np.allclose(diagonal, [r["raw_norm"] for r in rows], rtol=1e-4, atol=1e-5):
            raise ValueError("Gradient Gram matrix disagrees with independently measured norms")
        result["checkpoints"][str(step)] = dict(
            epoch=ready["epoch"], rows=rows, groups=summarize(rows, clip_norm),
            gradient_gram=gram.tolist(),
            synthetic_batches=synthetic_batches(gram, rows, clip_norm, spec["mixed_batches"]),
            seconds=time.monotonic() - started,
            peak_gpu_gib=torch.cuda.max_memory_allocated() / 2**30)
        (args.out / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
        del matrix, vectors, rows, gram
        reader.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
    print(json.dumps(dict(completed=True, out=str(args.out))), flush=True)


if __name__ == "__main__":
    main()
