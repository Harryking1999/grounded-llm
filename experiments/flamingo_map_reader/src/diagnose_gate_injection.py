"""Observe gated map residuals on matched inputs without changing model behavior."""

from argparse import ArgumentParser
from functools import partial
import json
import math
from pathlib import Path
import random
import time

import numpy as np
import torch


class ResidualObserver:
    """Measure intended and realized additions; preserve bf16 rounding effects."""

    def __init__(self, attentions):
        self.attentions = list(attentions)
        self.handles, self.pending = [], {}
        self.totals = {}
        self.dtypes = set()
        for index, attention in enumerate(self.attentions):
            self.handles.append(attention.register_forward_pre_hook(partial(self.before, index)))
            self.handles.append(attention.to_output.register_forward_hook(partial(self.read, index)))
            self.handles.append(attention.register_forward_hook(partial(self.after, index)))

    def start_sample(self, prediction_mask):
        if self.pending:
            raise ValueError("Previous sample left unfinished attention hooks")
        self.masks = {
            "all_tokens": torch.ones_like(prediction_mask, dtype=torch.bool),
            "supervised_prediction_tokens": prediction_mask,
        }

    def before(self, index, module, inputs):
        self.pending[index] = {"hidden": inputs[0]}

    def read(self, index, module, inputs, output):
        self.pending[index]["read_squared"] = output.float().square().sum(dim=-1)

    def after(self, index, module, inputs, output):
        state = self.pending.pop(index)
        hidden = state["hidden"]
        hidden_squared = hidden.float().square().sum(dim=-1)
        realized_squared = (output.float() - hidden.float()).square().sum(dim=-1)
        changed_coordinates = (output != hidden).sum(dim=-1)
        gate = float(torch.tanh(module.gate.detach()))
        self.dtypes.add((str(hidden.dtype), str(output.dtype)))
        for scope, mask in self.masks.items():
            values = np.array([
                float(hidden_squared[mask].sum()),
                float(state["read_squared"][mask].sum()),
                float(state["read_squared"][mask].sum()) * gate**2,
                float(realized_squared[mask].sum()),
                float(changed_coordinates[mask].sum()),
                int(mask.sum()) * hidden.shape[-1],
                1,
            ])
            self.totals.setdefault(scope, np.zeros((len(self.attentions), 7)))[index] += values

    @staticmethod
    def ratios(values):
        h, read, intended, realized, changed, coordinates, calls = values
        return dict(
            ungated_read_to_hidden_rms=(read / h) ** 0.5 if h else None,
            intended_residual_to_hidden_rms=(intended / h) ** 0.5 if h else None,
            realized_residual_to_hidden_rms=(realized / h) ** 0.5 if h else None,
            changed_coordinate_fraction=changed / coordinates if coordinates else None,
            coordinates=int(coordinates), calls=int(calls),
        )

    def result(self):
        return {
            "dtypes": sorted(self.dtypes),
            "scopes": {
                scope: {"overall": self.ratios(values.sum(axis=0)),
                        "layers": [self.ratios(row) for row in values]}
                for scope, values in self.totals.items()
            },
        }

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def main():
    parser = ArgumentParser()
    for name in ("spec", "old-run", "new-run", "model-path", "out"):
        parser.add_argument("--" + name, required=True, type=Path)
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError("Use a new diagnostic output directory")
    spec = json.loads(args.spec.read_text())
    if args.new_run.name != spec["new_run"] or args.old_run.as_posix().split("/runs/")[-1] != spec["old_run"]:
        raise ValueError("Run paths disagree with the diagnostic contract")
    torch.set_num_threads(1)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from .train import build_reader, collate_examples, to_device
    from .trajectory_dataset import PreparedTrajectoryDataset

    runs = {"old": args.old_run, "new": args.new_run}
    contracts = {name: json.loads((run / "training/config.json").read_text()) for name, run in runs.items()}
    if contracts["old"]["map_source"] != contracts["new"]["map_source"]:
        raise ValueError("Comparing different frozen maps is outside this diagnostic")
    manifests = {name: json.loads((run / "data/manifest.json").read_text()) for name, run in runs.items()}
    old_records = {r["trajectory_id"]: r for r in manifests["old"]["records"] if r["split"] == "train"}
    rng = random.Random(spec["seed"])
    pool = [r for r in manifests["new"]["records"] if r["split"] == "train"
            and not r.get("no_solution") and r["action_count"] > 0]
    selected = [(r, rng.randrange(r["variants"])) for r in rng.sample(pool, spec["success_trajectories"])]
    examples = []
    for record, variant in selected:
        old_record = old_records[record["trajectory_id"]]
        if any(record[field] != old_record[field] for field in ("start", "goal")):
            raise ValueError("Physical tasks differ")
        left = PreparedTrajectoryDataset(args.old_run / "data/manifest.json", [old_record], contracts["old"]["config"])[variant]
        right = PreparedTrajectoryDataset(args.new_run / "data/manifest.json", [record], contracts["new"]["config"])[variant]
        if left[0] != right[0] or len(left[1]) != len(right[1]):
            raise ValueError("Cached texts, numbering or trajectory lengths differ")
        for a, b in zip(left[1], right[1]):
            for field in ("vectors", "roles", "candidate_ids", "valid"):
                if not torch.equal(getattr(a.map_batch, field), getattr(b.map_batch, field)):
                    raise ValueError("Cached map inputs differ")
        examples.append(right)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    args.out.mkdir(parents=True)
    result = {
        "spec": spec,
        "samples": [dict(trajectory_id=r["trajectory_id"], variant=v,
                         sequence_tokens=len(example[0].input_ids), supervised_tokens=example[0].answer_tokens)
                    for (r, v), example in zip(selected, examples)],
        "matched_inputs_verified": True,
        "method": "Same full cached successful training trajectories and numbered variants in both architectures; identical tokens and map tensors verified. Eval mode, bf16 autocast, no gradients or optimizer. RMS ratios pool squared energies over selected positions and layers; intended measures tanh(alpha)*read before residual addition, realized measures the actual output-input difference including dtype rounding. These ratios measure local injection amplitude, not causal task benefit.",
        "architectures": {},
    }
    for name, run in runs.items():
        training = contracts[name]["config"]["training"]
        numbered = sum(r["variants"] for r in manifests[name]["records"] if r["split"] == "train")
        steps_per_epoch = math.ceil(numbered / (contracts[name]["batch_size"] * training.get("world_size", 1)))
        base = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=torch.bfloat16)
        base.config.use_cache = False
        reader = build_reader(base, contracts[name]["config"]).cuda().eval()
        result["architectures"][name] = {}
        for step in spec[name + "_checkpoint_steps"]:
            started = time.monotonic()
            saved = torch.load(run / f"training/models/checkpoint-{step}/adapter.pt", map_location="cpu", weights_only=True)
            reader.load_adapter_state_dict(saved["adapter"])
            observer = ResidualObserver(layer.map_attention for layer in reader.conditioned_layers)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                for example in examples:
                    timeline, inputs = collate_examples([example], tokenizer.eos_token_id)
                    timeline = to_device(timeline, torch.device("cuda"))
                    inputs = {key: value.cuda() for key, value in inputs.items()}
                    prediction_mask = torch.zeros_like(inputs["labels"], dtype=torch.bool)
                    prediction_mask[:, :-1] = inputs["labels"][:, 1:] != -100
                    observer.start_sample(prediction_mask)
                    with reader.conditioned(timeline):
                        output = reader.base_model.model(input_ids=inputs["input_ids"],
                            attention_mask=inputs["attention_mask"], use_cache=False)
                    del output, timeline, inputs, prediction_mask
            observed = observer.result()
            observer.close()
            gates = np.array([float(layer.map_attention.gate.detach()) for layer in reader.conditioned_layers])
            for scope in observed["scopes"].values():
                if any(layer["calls"] != len(examples) for layer in scope["layers"]):
                    raise ValueError("Missing or repeated layer observations")
            observed.update(step=step, epoch=step / steps_per_epoch,
                            gate_absolute_median=float(np.median(np.abs(gates))),
                            gate_absolute_max=float(np.max(np.abs(gates))), seconds=time.monotonic() - started)
            result["architectures"][name][str(step)] = observed
            (args.out / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
            print(json.dumps(dict(architecture=name, step=step, **observed["scopes"]["supervised_prediction_tokens"]["overall"])), flush=True)
        del observer, reader, base, saved
        torch.cuda.empty_cache()
    print(json.dumps(dict(completed=True, out=str(args.out))), flush=True)


if __name__ == "__main__":
    main()
