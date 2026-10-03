"""Run the authorized two-loss completion study and 8-to-10-epoch continuation."""

from argparse import ArgumentParser
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def command(python, module, **options):
    args = [python, "-u", "-m", f"experiments.flamingo_map_reader.src.{module}"]
    for key, value in options.items():
        if value is not None:
            args.extend(["--" + key.replace("_", "-"), str(value)])
    return args


def execute(spec, name, gpu, args):
    root = Path(spec["out"])
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS="1")
    with (root / "logs" / f"{name}.log").open("w") as handle:
        child = subprocess.Popen(args, env=env, stdout=handle, stderr=subprocess.STDOUT)
        write_json(root / f"{name}_process.json", {"pid": child.pid, "gpu": gpu, "command": args})
        code = child.wait()
    write_json(root / f"{name}_exit.json", {"exit_code": code})
    if code:
        raise RuntimeError(f"{name} failed with {code}; see logs/{name}.log")


def prepare(spec):
    root = Path(spec["out"])
    root.mkdir()
    (root / "logs").mkdir()
    (root / "data").mkdir()
    prior = json.loads(Path(spec["prior_manifest"]).read_text())
    counts = Counter(r["split"] for r in prior["records"])
    if counts != {"train": 1024, "validation": 256, "test": 256}:
        raise ValueError(f"Unexpected source split sizes: {counts}")
    for name, path in (("ordinary", spec["config"]), ("ranking5", spec["weighted_config"])):
        config = json.loads(Path(path).read_text())
        manifest = {**prior, "config": config, "source_manifest": spec["prior_manifest"],
            "supervision": "full original answer completion, original/Q-swap pairs, per-epoch consistent candidate permutation"}
        write_json(root / "data" / f"{name}.json", manifest)
    normal = json.loads(Path(spec["config"]).read_text())
    weighted = json.loads(Path(spec["weighted_config"]).read_text())
    normalized = json.loads(json.dumps(weighted))
    normalized["study"] = normal["study"]
    if normalized["training"].pop("decision_focus_weight") != 5:
        raise ValueError("Weighted comparison must use ranking weight five")
    normalized["training"].pop("decision_focus_scope")
    if normalized != normal:
        raise ValueError("Loss comparison differs beyond its ranking weight")
    write_json(root / "run.json", spec)


def training(spec, name, gpu):
    root = Path(spec["out"])
    path = spec["config"] if name == "ordinary" else spec["weighted_config"]
    execute(spec, name, gpu, command(spec["python"], "train", config=path,
        manifest=root / "data" / f"{name}.json", source_root=spec["source_root"],
        model_path=spec["model_path"], batch_size=1, out=root / name))


def continuation(spec):
    original = Path(spec["prior_training"])
    prior = json.loads((original / "config.json").read_text())
    config = Path(spec["out"]) / "data/short_original_config.json"
    write_json(config, prior["config"])
    execute(spec, "short_continuation", 2, command(spec["python"], "train",
        config=config, manifest=prior["manifest"], source_root=prior["map_source"],
        model_path=prior["model_source"], batch_size=prior["batch_size"],
        convergence_contract=spec["extension_config"],
        resume=original / "models/checkpoint-16384", out=Path(spec["out"]) / "short_continuation"))


def worker_command(spec, index, final_only=False):
    args = command(spec["python"], "epoch_readout_queue", mode="worker", out=spec["out"],
                   worker_index=index)
    if final_only:
        args.append("--final-only")
    return args


def await_checkpoint(root, name, step):
    checkpoint = root / name / "models" / f"checkpoint-{step}"
    while not (checkpoint / "evaluation_ready.json").exists():
        ended = root / f"{name}_exit.json"
        if ended.exists():
            raise RuntimeError(f"{name} ended without required checkpoint {step}")
        time.sleep(20)
    return checkpoint


def worker(root, index, final_only):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from .completion_diagnostics import evaluate_completion
    from .train import build_reader

    spec = json.loads((root / "run.json").read_text())
    normal = json.loads(Path(spec["config"]).read_text())
    tokenizer = AutoTokenizer.from_pretrained(spec["model_path"])
    base = AutoModelForCausalLM.from_pretrained(spec["model_path"], torch_dtype=torch.bfloat16)
    reader = build_reader(base, normal).to("cuda:0").eval()
    epochs = [10] if final_only else range(1, 11)
    for epoch in epochs:
        for name in ("ordinary", "ranking5"):
            checkpoint = await_checkpoint(root, name, epoch * 2048)
            saved = torch.load(checkpoint / "adapter.pt", map_location="cpu", weights_only=True)
            manifest_path = root / "data" / f"{name}.json"
            manifest = json.loads(manifest_path.read_text())
            if (saved["contract"]["config"] != manifest["config"] or
                    saved["contract"]["manifest"] != str(manifest_path.resolve())):
                raise ValueError("Checkpoint/evaluation contract mismatch")
            reader.load_adapter_state_dict(saved["adapter"])
            del saved
            summary = evaluate_completion(reader, tokenizer, manifest["config"],
                manifest["records"], Path(spec["source_root"]),
                root / "evaluation" / name / f"epoch-{epoch:02d}" / f"shard-{index}",
                shard_index=index, shard_count=4 if epoch == 10 else 2, final=epoch == 10)
            print(json.dumps({"condition": name, "epoch": epoch, "records": summary["records"]}), flush=True)


def summarize(root):
    rows = []
    for name in ("ordinary", "ranking5"):
        for epoch in range(1, 11):
            needed = 4 if epoch == 10 else 2
            files = [root / "evaluation" / name / f"epoch-{epoch:02d}" /
                     f"shard-{i}/summary.json" for i in range(needed)]
            if not all(path.exists() for path in files):
                continue
            scores, losses = defaultdict(Counter), defaultdict(Counter)
            for path in files:
                part = json.loads(path.read_text())
                for group, values in part["scores"].items():
                    scores[group].update(values)
                for group, values in part["loss_sums"].items():
                    losses[group].update(values)
            rates = {}
            for group, count in scores.items():
                rates[group] = {"original_exact": count["original_exact"] / count["records"],
                    "original_first_accuracy": count["original_first_correct"] / count["records"],
                    "paired_order_accuracy": count["paired_order_correct"] / count["pairs"],
                    "paired_exact_accuracy": count["paired_exact"] / count["pairs"],
                    "permutation_consistency": count["renumber_same_physical_action"] / count["records"],
                    "illegal_action_rate": 1 - count["original_legal"] / count["records"]}
            ce = {group: {"overall_ce": count["total_nll"] / count["total_tokens"],
                          "ranking_ce": count["ranking_nll"] / count["ranking_tokens"],
                          "first_ranking_token_accuracy": count["first_ranking_token_correct"] / count["examples"]}
                  for group, count in losses.items()}
            rows.append({"condition": name, "epoch": epoch, "scores": dict(scores),
                         "rates": rates, "loss": ce})
    write_json(root / "epoch_metrics.json", rows)
    return rows


def run(spec):
    prepare(spec)
    root = Path(spec["out"])

    def baseline_then_worker():
        continuation(spec)
        execute(spec, "diagnostic_worker0", 2, worker_command(spec, 0))

    with ThreadPoolExecutor(max_workers=6) as pool:
        ordinary = pool.submit(training, spec, "ordinary", 0)
        weighted = pool.submit(training, spec, "ranking5", 1)
        worker0 = pool.submit(baseline_then_worker)
        worker1 = pool.submit(execute, spec, "diagnostic_worker1", 3, worker_command(spec, 1))
        ordinary.result()
        weighted.result()
        final2 = pool.submit(execute, spec, "diagnostic_worker2", 0, worker_command(spec, 2, True))
        final3 = pool.submit(execute, spec, "diagnostic_worker3", 1, worker_command(spec, 3, True))
        for future in (worker0, worker1, final2, final3):
            future.result()
    rows = summarize(root)
    if len(rows) != 20:
        raise ValueError("Incomplete two-condition epoch evaluation")
    execute(spec, "short_final_readout", 0, command(spec["python"], "evaluate_path_readout",
        manifest=spec["prior_manifest"], source_root=spec["source_root"], model_path=spec["model_path"],
        adapter_checkpoint=root / "short_continuation/models/final/adapter.pt",
        out=root / "evaluation/short_final"))
    write_json(root / "queue_status.json", {"status": "complete", "evaluated_epochs": len(rows)})
    print("TEN_EPOCH_QUEUE_COMPLETE", flush=True)


def main():
    parser = ArgumentParser()
    parser.add_argument("--mode", choices=("run", "worker", "summarize"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--weighted-config", type=Path)
    parser.add_argument("--extension-config", type=Path)
    parser.add_argument("--prior-manifest", type=Path)
    parser.add_argument("--prior-training", type=Path)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--source-commit")
    parser.add_argument("--worker-index", type=int, default=0)
    parser.add_argument("--final-only", action="store_true")
    args = parser.parse_args()
    if args.mode == "worker":
        worker(args.out, args.worker_index, args.final_only)
    elif args.mode == "summarize":
        summarize(args.out)
    else:
        spec = {key: str(value.resolve()) if isinstance(value, Path) else value
                for key, value in vars(args).items()}
        spec["python"] = sys.executable
        try:
            run(spec)
        except Exception as error:
            if args.out.exists():
                write_json(args.out / "queue_status.json", {"status": "failed", "error": str(error)})
            raise


if __name__ == "__main__":
    main()
