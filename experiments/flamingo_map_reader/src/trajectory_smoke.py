"""One real optimizer update on the longest prepared trajectory, then discard weights."""

from argparse import ArgumentParser
import json
from pathlib import Path

import torch

from .train import build_reader, collate_examples, to_device
from .trajectory_dataset import PreparedTrajectoryDataset


def main():
    parser = ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    manifest = json.loads(args.manifest.read_text())
    config = manifest["config"]
    record = max((r for r in manifest["records"] if r["split"] == "train"), key=lambda r: r["max_tokens"])
    dataset = PreparedTrajectoryDataset(args.manifest, [record], config)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    base = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=torch.bfloat16)
    base.config.use_cache = False
    reader = build_reader(base, config).cuda().train()
    timeline, inputs = collate_examples([dataset[0]], tokenizer.eos_token_id)
    timeline = to_device(timeline, torch.device("cuda"))
    inputs = {k: v.cuda() for k, v in inputs.items()}
    optimizer = torch.optim.AdamW((p for p in reader.parameters() if p.requires_grad), lr=config["training"]["learning_rate"])
    with torch.autocast("cuda", dtype=torch.bfloat16):
        output = reader(map_batch=timeline, **inputs)
    output.loss.backward()
    gradients = [p.grad for p in reader.parameters() if p.requires_grad and p.grad is not None]
    if not gradients or not torch.isfinite(output.loss) or any(not torch.isfinite(g).all() for g in gradients):
        raise ValueError("Nonfinite loss or gradients in the longest full-trajectory update")
    if not any(torch.count_nonzero(g) for g in gradients):
        raise ValueError("All adapter gradients are zero")
    if any(p.grad is not None for p in reader.parameters() if not p.requires_grad):
        raise ValueError("Frozen parameter received a gradient")
    optimizer.step()
    result = dict(trajectory_id=record["trajectory_id"], tokens=inputs["input_ids"].shape[1],
                  loss=output.loss.item(), peak_gpu_gib=torch.cuda.max_memory_allocated() / 2**30,
                  finite_gradients=True, frozen_parameters_without_gradients=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
