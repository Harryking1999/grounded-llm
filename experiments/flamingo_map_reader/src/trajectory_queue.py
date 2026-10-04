"""Detached four-GPU queue: two independent trainers, shared checkpoint evaluators."""

from argparse import ArgumentParser
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


MODULE = "experiments.flamingo_map_reader.src."
CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def atomic_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def checkpoint_jobs(task, directory, manifest, root):
    """Finite schedule. Hold test evaluation until the final, unselected checkpoint."""
    spec = manifest["config"]["evaluation"]
    checkpoint = directory.name
    if checkpoint == "final":
        schedules = [("train", "reference", 1, False, spec["train_diagnostic_tasks"]),
                     ("validation", "reference", 1, False, None),
                     ("test", "reference", spec["numbering_variants"], False, None),
                     ("test", "rollout", spec["numbering_variants"], False, None),
                     ("test", "reference", 1, True, None),
                     ("test", "rollout", 1, True, None)]
    else:
        ready = json.loads((directory / "evaluation_ready.json").read_text())
        if ready["step"] == 0:
            schedules = [("train", "reference", 1, False, spec["initial_diagnostic_tasks"])]
        elif ready["step"] in manifest["config"]["checkpoint"]["early_steps"]:
            schedules = [("train", "reference", 1, False, spec["train_diagnostic_tasks"])]
        elif abs(ready["epoch"] - round(ready["epoch"])) < 1e-6 and ready["epoch"] < manifest["config"]["training"]["epochs"]:
            schedules = [("train", "reference", 1, False, spec["train_diagnostic_tasks"]),
                         ("validation", "reference", 1, False, None)]
        else:
            return []
    jobs = []
    for split, mode, variants, no_map, limit in schedules:
        count = sum(r["split"] == split for r in manifest["records"])
        count = min(count, limit) if limit else count
        name = f"{task}/{checkpoint}/{split}_{mode}_{'no_map' if no_map else 'map'}"
        for start in range(0, count, spec["shard_tasks"]):
            stop = min(start + spec["shard_tasks"], count)
            key = name + f"/{start:05d}_{stop:05d}"
            args = ["--manifest", str(root / task / "data/manifest.json"), "--adapter-checkpoint", str(directory / "adapter.pt"),
                "--split", split, "--mode", mode, "--variants", str(variants), "--start", str(start), "--stop", str(stop),
                "--out", str(root / "evaluation" / key)]
            if no_map:
                args.append("--no-map")
            if limit:
                args.extend(["--limit", str(limit)])
            jobs.append(dict(key=key, args=args))
    return jobs


def resume_point(models):
    """Newest checkpoint Trainer finished writing, or None for a fresh run.

    Trainer publishes evaluation_ready.json only after adapter, optimizer,
    scheduler, RNG and state are all on disk, so its presence is what separates
    a checkpoint worth continuing from one a crash left half-written.
    """
    published = [ready.parent for ready in models.glob("checkpoint-*/evaluation_ready.json")]
    return max(published, key=lambda path: int(path.name.split("-")[-1])) if published else None


def completed_shards(evaluation):
    """Shard keys that already wrote their own summary.json.

    trajectory_eval writes summary.json only after the last case of its shard,
    so it marks a finished shard; a shard killed mid-run leaves a partial
    cases.jsonl and no summary, and must be redone rather than trusted.
    """
    return {summary.parent.relative_to(evaluation).as_posix()
            for summary in evaluation.glob("*/*/*/*/summary.json")}


def clear_partial_shard(directory):
    """Drop an interrupted shard's output; trajectory_eval refuses a directory that exists."""
    if directory.exists() and not (directory / "summary.json").is_file():
        shutil.rmtree(directory)


def main():
    parser = ArgumentParser()
    for arg in ("run-root", "model-path", "graph-source", "blocks-source-manifest", "blocks-official", "blocks-q", "blocks-q-data"):
        parser.add_argument("--" + arg, type=Path, required=True)
    parser.add_argument("--gpus", type=int, nargs=4, default=[0, 1, 2, 3])
    parser.add_argument("--prepare-workers", type=int, default=1,
                        help="Forked processes each preparation task may use; the two tasks run at once")
    args = parser.parse_args()
    root = args.run_root.resolve()
    logs = root / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", TOKENIZERS_PARALLELISM="false")
    tasks = {"path": dict(config=CONFIGS / "path_single_long.json", gpu=args.gpus[0],
                 prepare=["--source-root", str(args.graph_source)], train=["--source-root", str(args.graph_source)]),
             "blocks": dict(config=CONFIGS / "blocks1000_long.json", gpu=args.gpus[1],
                 prepare=["--source-manifest", str(args.blocks_source_manifest), "--official-data", str(args.blocks_official),
                          "--q-checkpoint", str(args.blocks_q), "--q-training-data", str(args.blocks_q_data)],
                 train=["--q-checkpoint", str(args.blocks_q)])}
    processes, pending, known, completed, failed = {}, [], set(), [], []

    def launch(key, module, arguments, gpu=None):
        log = logs / (key.replace("/", "_") + ".log")
        command = [sys.executable, "-u", "-m", MODULE + module, *arguments]
        child_env = dict(env, CUDA_VISIBLE_DEVICES=str(gpu) if gpu is not None else "")
        with log.open("wb") as handle:
            process = subprocess.Popen(command, env=child_env, stdout=handle, stderr=subprocess.STDOUT)
        processes[key] = dict(process=process, gpu=gpu, command=command, log=str(log))
        print(json.dumps(dict(event="launched", key=key, pid=process.pid, gpu=gpu, log=str(log))), flush=True)

    def launch_train(task, spec, resume=None):
        arguments = ["--config", str(spec["config"]), "--manifest", str(root / task / "data/manifest.json"),
                     "--model-path", str(args.model_path), "--batch-size", "1",
                     "--out", str(root / task / "training"), *spec["train"]]
        if resume is not None:
            arguments += ["--resume", str(resume)]
        launch(task + "/train", "train", arguments, spec["gpu"])

    # A restarted queue adopts what the previous process left behind instead of
    # redoing it: prepared data, a passing smoke test, and the newest published
    # checkpoint, which train.py continues rather than restarting from scratch.
    for task, spec in tasks.items():
        manifest = root / task / "data/manifest.json"
        if not manifest.is_file():
            launch(task + "/prepare", "prepare_trajectories", ["--config", str(spec["config"]), "--model-path", str(args.model_path),
                   "--out", str(root / task / "data"), "--workers", str(args.prepare_workers), *spec["prepare"]])
            spec["phase"] = "prepare"
            continue
        spec["manifest"] = json.loads(manifest.read_text())
        if not (root / task / "smoke.json").is_file():
            launch(task + "/smoke", "trajectory_smoke", ["--manifest", str(manifest),
                "--model-path", str(args.model_path), "--out", str(root / task / "smoke.json")], spec["gpu"])
            spec["phase"] = "smoke"
            continue
        launch_train(task, spec, resume_point(root / task / "training/models"))
        spec["phase"] = "train"
    known |= completed_shards(root / "evaluation")
    while True:
        for key, running in list(processes.items()):
            code = running["process"].poll()
            if code is None:
                continue
            del processes[key]
            (failed if code else completed).append(dict(key=key, exit_code=code, log=running["log"]))
            print(json.dumps(dict(event="finished", key=key, exit_code=code)), flush=True)
            for task, spec in tasks.items():
                if key == task + "/prepare":
                    if code:
                        spec["phase"] = "failed"
                    else:
                        spec["manifest"] = json.loads((root / task / "data/manifest.json").read_text())
                        launch(task + "/smoke", "trajectory_smoke", ["--manifest", str(root / task / "data/manifest.json"),
                            "--model-path", str(args.model_path), "--out", str(root / task / "smoke.json")], spec["gpu"])
                        spec["phase"] = "smoke"
                elif key == task + "/smoke":
                    if code:
                        spec["phase"] = "failed"
                    else:
                        launch_train(task, spec)
                        spec["phase"] = "train"
                elif key == task + "/train":
                    spec["phase"] = "failed" if code else "trained"
        for task, spec in tasks.items():
            if "manifest" not in spec:
                continue
            for ready in sorted((root / task / "training/models").glob("*/evaluation_ready.json")):
                for job in checkpoint_jobs(task, ready.parent, spec["manifest"], root):
                    if job["key"] not in known:
                        known.add(job["key"])
                        pending.append(job)
        occupied = {p["gpu"] for p in processes.values() if p["gpu"] is not None}
        # Training GPUs are reserved through preparation, then join evaluation
        # when their own trainer finishes. No GPU has two simultaneous jobs.
        available = [g for g in args.gpus if g not in occupied and not any(
            s["gpu"] == g and s["phase"] in ("prepare", "smoke", "train") for s in tasks.values())]
        for gpu in available:
            if pending:
                job = pending.pop(0)
                clear_partial_shard(root / "evaluation" / job["key"])
                launch(job["key"], "trajectory_eval", [*job["args"], "--model-path", str(args.model_path)], gpu)
        status = dict(pid=os.getpid(), tasks={k: dict(phase=v["phase"], gpu=v["gpu"]) for k, v in tasks.items()},
            running={k: dict(pid=v["process"].pid, gpu=v["gpu"], log=v["log"]) for k, v in processes.items()},
            pending_evaluations=len(pending), completed=completed, failed=failed)
        atomic_json(root / "queue_status.json", status)
        if not processes and not pending:
            break
        time.sleep(15)
    # Aggregate across shards so paired target changes can cross shard boundaries.
    from .trajectory_eval import aggregate
    for task, spec in tasks.items():
        if "manifest" not in spec:
            continue
        groups = defaultdict_list(root / "evaluation" / task)
        for group, shards in groups.items():
            results = [json.loads(line) for shard in shards for line in (shard / "cases.jsonl").read_text().splitlines()]
            atomic_json(group / "summary.json", aggregate(results, spec["manifest"]))
    atomic_json(root / "completion.json", dict(success=not failed, failed=failed, completed_jobs=len(completed)))
    if failed:
        raise SystemExit(1)


def defaultdict_list(task_root):
    groups = {}
    for summary in task_root.glob("*/*/*/summary.json"):
        groups.setdefault(summary.parent.parent, []).append(summary.parent)
    return groups


if __name__ == "__main__":
    main()
