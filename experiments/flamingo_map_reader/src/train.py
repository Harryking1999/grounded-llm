"""Standard HF Trainer SFT with a map-aware collator and adapter-only saves."""

from argparse import ArgumentParser
from dataclasses import replace
import json
import math
import os
import random
from pathlib import Path

import torch
import numpy as np
from torch.nn import functional as F
from torch.utils.data import Dataset
from transformers import Trainer, TrainerCallback, TrainingArguments, set_seed

from .blocks import FrozenBoardMap
from .blocks_data import demonstration_from_record
from .data import demonstration_from_record as graph_demonstration_from_record, load_graph
from .fusion import MapReader
from .graph import batch_maps
from .memory import (AddressedMapMemoryEncoder, JointFeatureMapMemoryEncoder,
                     MapBatch, MapMemoryEncoder, MapTimeline)
from .readout_aux import CounterfactualFirstTurnDataset
from .transcript import encode_trajectory


def build_reader(base, config, *, sequence_mean_loss=False, cache_generation=False):
    spec = config["map"]
    mode = spec.get("memory_mode", "joint")
    if mode == "joint":
        # Deprecated fallback, reached only by configs that predate addressed memory
        # or by a config that omits memory_mode. Current configs set it explicitly.
        encoder = MapMemoryEncoder
    elif mode == "address_key_state_value":
        encoder = AddressedMapMemoryEncoder
    elif mode == "joint_feature_kv":
        encoder = JointFeatureMapMemoryEncoder
    else:
        raise ValueError(f"Unknown map memory mode: {mode}")
    memory = encoder(spec["state_dim"], base.config.hidden_size,
                     spec["maximum_candidates"], spec["projection_dim"],
                     spec["role_and_id_dim"], **({"feature_ffn_hidden_dim":
                         spec.get("feature_ffn_hidden_dim", 0)} if mode != "joint" else {}))
    return MapReader(base, memory, spec["attention_heads"], spec["attention_head_dim"],
                      spec["cross_attention_every_n_layers"],
                      # Both deprecated; the current configs set neither. Read through
                      # only so the archived runs that did stay reproducible.
                      fixed_gate_tanh=spec.get("fixed_gate_tanh"),
                      value_scale=spec.get("value_scale", 1.0),
                      checkpoint_layers=config["training"].get("checkpoint_layers", False),
                      loss_chunk_tokens=config["training"].get("loss_chunk_tokens", 0),
                      sequence_mean_loss=sequence_mean_loss, cache_generation=cache_generation)


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


def decision_weighted_loss(ordinary_loss, logits, labels, focus_mask, weight):
    """Preserve normal CE while increasing the selected token contributions."""
    if weight <= 1:
        raise ValueError("Decision weight must exceed one")
    selected = focus_mask[:, 1:] & (labels[:, 1:] != -100)
    if not bool(selected.any()):
        return ordinary_loss
    focused_logits = logits[:, :-1][selected].float()
    focused_labels = labels[:, 1:][selected]
    focused_loss = F.cross_entropy(focused_logits, focused_labels, reduction="sum")
    ordinary_count = (labels[:, 1:] != -100).sum()
    total_weight = ordinary_count + (weight - 1) * selected.sum()
    return (ordinary_loss * ordinary_count + (weight - 1) * focused_loss) / total_weight


class SFTDataset(Dataset):
    def __init__(self, records, qmap, tokenizer, config, source_root=None):
        self.records, self.qmap, self.tokenizer, self.config = records, qmap, tokenizer, config
        self.source_root = source_root
        self.graphs = {}

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        if self.config["task"] == "blocks":
            demo = demonstration_from_record(self.qmap, record,
                self.config["maximum_demonstration_actions"],
                self.config["data"].get("reported_candidates"))
        else:
            graph_id = record["graph_id"]
            if graph_id not in self.graphs:
                self.graphs[graph_id] = load_graph(self.source_root, graph_id)[:2]
            demo = graph_demonstration_from_record(*self.graphs[graph_id], record)
        encoded = encode_trajectory(demo, self.tokenizer, self.config["maximum_sequence_tokens"],
                                     chat_template_kwargs=self.config.get("chat_template_kwargs", {}),
                                     focus_decisions=self.config["training"].get("decision_focus_weight", 1) > 1)
        return encoded, [turn.step for turn in demo.turns]


class MapCollator:
    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id

    def __call__(self, examples):
        timeline, inputs = collate_examples(examples, self.pad_token_id)
        return dict(inputs, map_batch=timeline)


class EpochCheckpoint(TrainerCallback):
    """Request saves after each tenth of an epoch, at optimizer boundaries."""
    def __init__(self, fraction, early_steps=()):
        self.fraction = fraction
        self.bucket = 0
        self.early_steps = set(early_steps)

    def on_train_begin(self, args, state, control, **kwargs):
        self.bucket = int(((state.epoch or 0) + 1e-8) / self.fraction)

    def on_step_end(self, args, state, control, **kwargs):
        bucket = int(((state.epoch or 0) + 1e-8) / self.fraction)
        if bucket > self.bucket or state.global_step in self.early_steps:
            control.should_save = True
            self.bucket = bucket
        return control

    def on_epoch_end(self, args, state, control, **kwargs):
        control.should_save = True
        return control

    def on_save(self, args, state, control, **kwargs):
        if torch.distributed.is_initialized():
            # Publish only after every rank has finished writing its RNG state.
            torch.distributed.barrier()
        if state.is_world_process_zero:
            checkpoint_ready(Path(args.output_dir) / f"checkpoint-{state.global_step}",
                             state.global_step, state.epoch)


def checkpoint_ready(path, step, epoch):
    """Publish only after Trainer finished saving; concurrent evaluators wait here."""
    temporary = Path(path) / "evaluation_ready.tmp"
    temporary.write_text(json.dumps({"step": step, "epoch": epoch}) + "\n")
    temporary.replace(Path(path) / "evaluation_ready.json")


class DatasetEpoch(TrainerCallback):
    def __init__(self, dataset):
        self.dataset = dataset

    def on_epoch_begin(self, args, state, control, **kwargs):
        self.dataset.set_epoch(int(state.epoch or 0))


class JsonLog(TrainerCallback):
    def __init__(self, path):
        self.path = Path(path)

    def on_log(self, args, state, control, logs=None, **kwargs):
        if state.is_world_process_zero:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(dict(logs or {}, step=state.global_step)) + "\n")


class MapSFTTrainer(Trainer):
    # The base LM computes normal next-token CE. An optional decision focus
    # adds weight to selected labels; Trainer still owns optimization and resume.
    def __init__(self, *args, contract, **kwargs):
        self.contract = contract
        self.focus_weight = contract.get("config", {}).get("training", {}).get("decision_focus_weight", 1)
        super().__init__(*args, **kwargs)
        self.model_accepts_loss_kwargs = False

    def _prepare_input(self, data):
        if isinstance(data, MapTimeline):
            return to_device(data, self.args.device)
        return super()._prepare_input(data)

    def _inner_training_loop(self, batch_size=None, args=None, resume_from_checkpoint=None,
                             trial=None, ignore_keys_for_eval=None):
        if self.contract.get('resume_topology'):
            # Transformers 4 restores the source rank's batch before entering here.
            # The explicit migration keeps global groups but changes each rank to 2.
            batch_size = self.contract['batch_size']
            self._train_batch_size = batch_size
        return super()._inner_training_loop(batch_size=batch_size, args=args,
            resume_from_checkpoint=resume_from_checkpoint, trial=trial,
            ignore_keys_for_eval=ignore_keys_for_eval)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        focus_mask = inputs.pop("focus_mask", None)
        if self.focus_weight <= 1:
            return super().compute_loss(model, inputs, return_outputs=return_outputs,
                                        num_items_in_batch=num_items_in_batch)
        if focus_mask is None:
            raise ValueError("Focused training needs token-aligned focus_mask")
        labels = inputs["labels"]
        outputs = model(**inputs)
        loss = decision_weighted_loss(outputs.loss, outputs.logits, labels,
                                      focus_mask, self.focus_weight)
        return (loss, outputs) if return_outputs else loss

    def _save(self, output_dir=None, state_dict=None):
        output = Path(output_dir or self.args.output_dir)
        output.mkdir(parents=True, exist_ok=True)
        torch.save({"adapter": self.model.adapter_state_dict(), "contract": self.contract},
                   output / "adapter.pt")
        torch.save(self.args, output / "training_args.bin")

    def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
        saved = torch.load(Path(resume_from_checkpoint) / "adapter.pt",
                           map_location="cpu", weights_only=True)
        if pinned_supervision(saved["contract"]) != pinned_supervision(self.contract):
            raise ValueError("Checkpoint and current SFT contracts differ")
        (model or self.model).load_adapter_state_dict(saved["adapter"])

    def _load_rng_state(self, checkpoint):
        if not self.contract.get("resume_topology") or checkpoint is None:
            return super()._load_rng_state(checkpoint)
        # A four-device RNG list cannot be installed wholesale on a two-device node.
        # Each process restores its own saved rank and only its active CUDA device.
        np_core = getattr(np, '_core', None)
        if np_core is None:
            from numpy import core as np_core
        path = Path(checkpoint) / f"rng_state_{self.args.process_index}.pth"
        with torch.serialization.safe_globals([np_core.multiarray._reconstruct,
                np.ndarray, np.dtype, type(np.dtype(np.uint32))]):
            saved = torch.load(path, map_location="cpu", weights_only=True)
        random.setstate(saved["python"])
        np.random.set_state(saved["numpy"])
        torch.random.set_rng_state(saved["cpu"])
        if torch.cuda.is_available():
            torch.cuda.random.set_rng_state(saved["cuda"][self.args.local_process_index], self.args.device)


def validate_resume_topology(config, topology, world_size, batch_size):
    """Explicitly allow the observed 4x1 -> 2x2 move without changing sample weights."""
    expected = dict(source_world_size=4, source_per_device_batch_size=1, world_size=2,
                    per_device_batch_size=2, loss_reduction="sequence_token_mean")
    training = config["training"]
    if (topology != expected or world_size != topology["world_size"] or
            batch_size != topology["per_device_batch_size"] or training.get("world_size") != 4 or
            training.get("batch_size_candidates_per_device") != [1] or
            training.get("batch_size_candidates_global") != [4] or
            training.get("decision_focus_weight", 1) != 1 or not training.get("loss_chunk_tokens")):
        raise ValueError("Topology migration only supports this run's equivalent 4x1 -> 2x2 objective")
    return topology


def pinned_supervision(contract):
    """Contract with the training budget left out.

    A checkpoint has to agree on what its adapter was trained on -- map source,
    model, data, batch and supervision settings -- but not on how long training
    was meant to run. The explicit 4x1 -> 2x2 migration normalizes only its
    equivalent per-sequence objective; other contracts retain their batch check.
    Extending a budget is an expected move: the convergence
    path takes max_total_steps from outside the contract for that same reason,
    and an extension has to resume the checkpoint the shorter budget produced.
    """
    config = json.loads(json.dumps(contract["config"]))
    for key in ("epochs", "max_steps"):
        config["training"].pop(key, None)
    normalized = {**contract, "config": config}
    topology = normalized.pop("resume_topology", None)
    if topology:
        validate_resume_topology(config, topology, topology["world_size"], contract["batch_size"])
        normalized["batch_size"] = topology["source_per_device_batch_size"]
    return normalized


def full_checkpoint(path, world_size=1):
    """A continuation needs all ranks' RNG state as well as optimizer progress."""
    path = Path(path)
    files = ("adapter.pt", "optimizer.pt", "scheduler.pt", "trainer_state.json")
    rng = ["rng_state.pth"] if world_size == 1 else [f"rng_state_{i}.pth" for i in range(world_size)]
    return all((path / name).is_file() for name in (*files, *rng))


def last_measured_step(output):
    """Last step already recorded for this output; None when nothing was measured."""
    history = Path(output) / "convergence.jsonl"
    if not history.is_file():
        return None
    rows = [json.loads(line) for line in history.read_text(encoding="utf-8").splitlines()]
    return rows[-1]["step"] if rows else None


def check_resume_target(output, resume, source_step, convergence):
    """Reject anything but continuing this output's own training from its last step.

    A run that ran out of budget still records where it stopped. A run killed from
    outside leaves its status at "running", so its last recorded measurement is the
    only evidence of where training actually reached.
    """
    if (resume.parent.resolve() != (Path(output) / "models").resolve() or
            convergence["max_total_steps"] <= source_step):
        raise ValueError("Existing convergence output must continue its own models directory")
    status = json.loads((Path(output) / "convergence_status.json").read_text(encoding="utf-8"))
    if status["status"] == "budget_reached_not_converged":
        if status["completed_step"] != source_step:
            raise ValueError("Extension does not resume from the exhausted budget's last step")
    elif status["status"] == "running":
        if last_measured_step(output) != source_step:
            raise ValueError("Interrupted run's last measurement differs from the resume step")
    else:
        raise ValueError("Existing convergence output is not resumable")


def fixed_warmup_arguments(training, source_steps):
    # Transformers 5 converts warmup_ratio into warmup_steps during __post_init__,
    # even when an explicit step count is also supplied. Omit the ratio entirely;
    # Transformers 4 requires its default numeric ratio rather than explicit None.
    return {"warmup_steps": training.get(
        "warmup_steps", math.ceil(source_steps * training["warmup_fraction"]))}


def extend_epoch_budget(config, extension, source_state, examples, global_batch):
    """Extend the source budget, or resume an interrupted checkpoint inside it."""
    training = config['training']
    steps_per_epoch = math.ceil(examples / global_batch)
    if (training.get('supervision_mode') != 'prepared_trajectory' or
            training.get('max_steps', -1) > 0 or
            extension['source_epochs'] != training['epochs'] or
            extension['total_epochs'] <= extension['source_epochs'] or
            extension['source_step'] != steps_per_epoch * extension['source_epochs'] or
            extension['max_total_steps'] != steps_per_epoch * extension['total_epochs']):
        raise ValueError('Epoch extension must continue the declared completed budget and sample groups')
    step = source_state['global_step']
    original_endpoint = step == extension['source_step']
    interrupted_extension = (
        extension['source_step'] < step < extension['max_total_steps'] and
        source_state.get('max_steps') == extension['max_total_steps'] and
        source_state.get('num_train_epochs') == extension['total_epochs'])
    if (not (original_endpoint or interrupted_extension) or
            not math.isclose(source_state['epoch'], step / steps_per_epoch)):
        raise ValueError('Checkpoint must match the original endpoint or interrupted extended budget')
    extended = json.loads(json.dumps(config))
    extended['training']['epochs'] = extension['total_epochs']
    return extended


def main():
    parser = ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--q-checkpoint", type=Path)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--model-path", type=Path,
                        help="Local copy of the model named in the contract")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--resume-topology", type=Path,
                        help="Explicit 4x1 to 2x2 continuation with equal per-sequence losses")
    parser.add_argument("--epoch-extension", type=Path,
                        help="Continue a completed epoch budget without rewriting the prepared manifest")
    parser.add_argument("--convergence-contract", type=Path,
                        help="Use fixed-set plateau checks; source_step=0 permits a fresh run")
    parser.add_argument("--batch-size", type=int, required=True,
                        help="Measured actual per-device batch size; no gradient accumulation")
    args = parser.parse_args()
    if args.epoch_extension and (args.resume is None or args.convergence_contract):
        parser.error('--epoch-extension needs --resume and cannot use convergence stopping')
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    if world_size > 1:
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        torch.distributed.init_process_group("nccl")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    topology = None
    if args.resume_topology:
        if args.resume is None:
            raise ValueError("Topology migration requires a full continuation checkpoint")
        topology = validate_resume_topology(config, json.loads(args.resume_topology.read_text()),
                                            world_size, args.batch_size)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if manifest["config"] != config:
        raise ValueError("Manifest and training configurations differ")
    task = config["task"]
    if task == "blocks":
        if args.q_checkpoint is None or Path(manifest["q_checkpoint"]).resolve() != args.q_checkpoint.resolve():
            raise ValueError("SFT labels require the same frozen Q checkpoint")
    elif task == "graph":
        if args.source_root is None or Path(manifest["source_root"]).resolve() != args.source_root.resolve():
            raise ValueError("Graph SFT requires its frozen source graphs")
    else:
        raise ValueError(f"Unsupported map-reader task: {task}")
    if args.resume is not None and not full_checkpoint(args.resume, world_size):
        raise ValueError("Resume requires adapter, optimizer, scheduler, trainer state and every rank's RNG state")
    if args.out.exists() and args.resume is None:
        raise FileExistsError(args.out)
    if world_size > 1:
        torch.distributed.barrier()
    records = [r for r in manifest["records"] if r["split"] == "train"]
    if not records or args.batch_size <= 0:
        raise ValueError("Training needs examples and a positive batch size")
    extension = None
    if args.epoch_extension:
        extension = json.loads(args.epoch_extension.read_text(encoding='utf-8'))
        source_state = json.loads((args.resume / 'trainer_state.json').read_text())
        config = extend_epoch_budget(config, extension, source_state,
                                     sum(r['variants'] for r in records), args.batch_size * world_size)
    contract = {"config": config, "manifest": str(args.manifest.resolve()),
                "map_source": str((args.q_checkpoint if task == "blocks" else args.source_root).resolve()),
                "model_source": str(args.model_path.resolve()) if args.model_path else config["model"],
                "batch_size": args.batch_size}
    if topology:
        contract["resume_topology"] = topology
    training = config["training"]
    convergence = None
    if args.convergence_contract:
        convergence = json.loads(args.convergence_contract.read_text(encoding="utf-8"))
        convergence = convergence.get("convergence", convergence)
        if args.resume is None:
            if convergence["source_step"] != 0:
                raise ValueError("Nonzero source_step needs a full checkpoint")
        else:
            for filename in ("optimizer.pt", "scheduler.pt", "trainer_state.json", "rng_state.pth"):
                if not (args.resume / filename).is_file():
                    raise ValueError(f"Full continuation is missing {filename}")
            source_state = json.loads((args.resume / "trainer_state.json").read_text())
            if source_state["global_step"] != convergence["source_step"]:
                raise ValueError("Continuation checkpoint does not match the declared source step")
        if args.out.exists():
            check_resume_target(args.out, args.resume, source_state["global_step"], convergence)
    set_seed(config["seed"])
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model_source = str(args.model_path) if args.model_path else config["model"]
    tokenizer = AutoTokenizer.from_pretrained(model_source)
    base = AutoModelForCausalLM.from_pretrained(model_source,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32)
    base.config.use_cache = False
    reader = build_reader(base, config, sequence_mean_loss=topology is not None)
    qmap = FrozenBoardMap.load(args.q_checkpoint) if task == "blocks" else None
    supervision_mode = training.get("supervision_mode", "trajectory")
    if supervision_mode == "trajectory":
        dataset = SFTDataset(records, qmap, tokenizer, config, args.source_root)
    elif supervision_mode == "prepared_trajectory":
        from .trajectory_dataset import PreparedTrajectoryDataset
        dataset = PreparedTrajectoryDataset(args.manifest, records, config)
    elif supervision_mode == "counterfactual_first_turn":
        dataset = CounterfactualFirstTurnDataset(records, qmap, tokenizer,
                                                 config, args.source_root)
    else:
        raise ValueError(f"Unknown supervision mode: {supervision_mode}")
    callbacks = [JsonLog(args.out / "logs/train.jsonl")]
    if getattr(dataset, "order_augmentation", "none") != "none":
        if convergence:
            raise ValueError("Per-epoch augmentation needs fixed-epoch training, not cached plateau data")
        callbacks.append(DatasetEpoch(dataset))
    maximum_steps = training.get("max_steps", -1)
    source_steps = maximum_steps if maximum_steps > 0 else math.ceil(
        len(dataset) / (args.batch_size * world_size)) * training["epochs"]
    extra_arguments = fixed_warmup_arguments(training, extension['source_step'] if extension else source_steps)
    if convergence:
        from .convergence import ConvergenceCheck, with_decision_mask
        # Caching preserves exactly the deterministic examples used by the source run.
        dataset = [with_decision_mask(dataset[index], tokenizer) for index in range(len(dataset))]
        check = ConvergenceCheck(convergence, args.out, dataset)
        callbacks.append(check)
        source_steps = maximum_steps if maximum_steps > 0 else math.ceil(
            len(dataset) / (args.batch_size * world_size)) * training["epochs"]
        maximum_steps = convergence["max_total_steps"]
        # Preserve the original warmup; extending the budget must not restart it.
        extra_arguments = {**fixed_warmup_arguments(training, source_steps),
                           "save_steps": convergence["evaluation_every_steps"],
                           "save_total_limit": 2, "disable_tqdm": True}
    else:
        callbacks.append(EpochCheckpoint(config["checkpoint"]["every_epoch_fraction"],
                                        config["checkpoint"].get("early_steps", ())))
    arguments = TrainingArguments(
        output_dir=str(args.out / "models"), logging_dir=str(args.out / "logs"),
        per_device_train_batch_size=args.batch_size, gradient_accumulation_steps=1,
        num_train_epochs=training["epochs"], max_steps=maximum_steps,
        learning_rate=training["learning_rate"],
        weight_decay=training["weight_decay"],
        lr_scheduler_type="constant_with_warmup", optim="adamw_torch",
        max_grad_norm=training["gradient_clip_norm"], bf16=torch.cuda.is_available(),
        remove_unused_columns=False, label_names=["labels"],
        save_strategy="steps" if convergence else "no", logging_steps=1, report_to=[],
        dataloader_num_workers=0, seed=config["seed"], data_seed=config["seed"],
        ddp_find_unused_parameters=False,
        **extra_arguments,
    )
    trainer = MapSFTTrainer(model=reader, args=arguments, train_dataset=dataset,
        data_collator=MapCollator(tokenizer.pad_token_id if tokenizer.pad_token_id is not None
                                  else tokenizer.eos_token_id), contract=contract,
        callbacks=callbacks)
    args.out.mkdir(parents=True, exist_ok=True)
    if convergence:
        check.trainer = trainer
        continuation = {
            "source_checkpoint": str(args.resume.resolve()) if args.resume else None,
            "source_contract": contract, "convergence": convergence}
        if (args.out / "continuation.json").exists():
            with (args.out / "continuation_extensions.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(continuation) + "\n")
        else:
            (args.out / "continuation.json").write_text(json.dumps(continuation, indent=2) + "\n")
    if rank == 0:
        (args.out / "config.json").write_text(json.dumps(contract, indent=2) + "\n")
        if extension:
            (args.out / 'continuation.json').write_text(json.dumps(dict(
                source_checkpoint=str(args.resume.resolve()), epoch_extension=extension,
                resume_topology=topology, warmup=extra_arguments), indent=2) + '\n')
    if config.get("checkpoint", {}).get("save_initial", False) and args.resume is None:
        initial = args.out / "models/checkpoint-0"
        trainer.save_model(str(initial))
        if rank == 0:
            checkpoint_ready(initial, 0, 0)
    result = trainer.train(resume_from_checkpoint=str(args.resume) if args.resume else None)
    trainer.save_model(str(args.out / "models/final"))
    if rank == 0:
        checkpoint_ready(args.out / "models/final", trainer.state.global_step, trainer.state.epoch)
        (args.out / "results").mkdir(exist_ok=True)
        (args.out / "results/summary.json").write_text(json.dumps(result.metrics, indent=2) + "\n")
    if world_size > 1:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
