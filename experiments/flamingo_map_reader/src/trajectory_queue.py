"""Detached training/evaluation queue, with disjoint evaluation workers."""

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
        # Sorts above every numbered checkpoint: "final" is wherever training
        # stopped, which is later than any epoch boundary already published.
        epoch = float("inf")
        schedules = [("train", "reference", 1, False, spec["train_diagnostic_tasks"]),
                     ("validation", "reference", 1, False, None),
                     ("validation", "rollout", 1, False, None),
                     ("test", "reference", spec["numbering_variants"], False, None),
                     ("test", "rollout", spec["numbering_variants"], False, None),
                     ("test", "reference", 1, True, None),
                     ("test", "rollout", 1, True, None)]
    else:
        ready = json.loads((directory / "evaluation_ready.json").read_text())
        epoch = ready["epoch"]
        if ready["step"] == 0:
            schedules = [("train", "reference", 1, False, spec["initial_diagnostic_tasks"])]
        elif ready["step"] in manifest["config"]["checkpoint"]["early_steps"]:
            schedules = [("train", "reference", 1, False, spec["train_diagnostic_tasks"])]
        elif abs(ready["epoch"] - round(ready["epoch"])) < 1e-6 and ready["epoch"] < manifest["config"]["training"]["epochs"]:
            schedules = [("train", "reference", 1, False, spec["train_diagnostic_tasks"]),
                         ("validation", "reference", 1, False, None),
                         ("validation", "rollout", 1, False, None)]
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
            jobs.append(dict(key=key, args=args, split=split, mode=mode, no_map=no_map, epoch=epoch))
    return jobs


def job_priority(job):
    """Queue order: later training first, and within a checkpoint the quickest reading.

    Two axes, and the checkpoint's stage is the second one rather than an
    artifact of scan order. The scan walks one whole task before the next, so a
    tie broken by insertion order would run every path reading ahead of every
    blocks reading no matter how early path's checkpoint was. Epoch is what
    compares across the two tasks -- raw steps do not share a scale -- and
    "final" sorts above every numbered checkpoint.

    Within one checkpoint: a validation rollout is a sixth of a test battery's
    case count and is the only closed-loop number that exists at more than one
    checkpoint, so it is what epochs can actually be compared on, and it goes
    first. Test follows -- it is the headline number but it is six times the
    size and exists only at the final checkpoint. The no_map control goes last:
    nothing waits on it, and it only matters once a map result exists to
    attribute a gap to.
    """
    if job["split"] == "validation" and job["mode"] == "rollout":
        rank = 0
    elif job["no_map"]:
        rank = 3
    elif job["split"] == "test":
        rank = 2
    else:
        rank = 1
    return (rank, -job["epoch"])


def checkpoint_progress(directory):
    """How far training had gone when this checkpoint was published.

    The scan enqueues the newest checkpoints first. An early diagnostic
    checkpoint carries almost no information yet costs as much GPU time as any
    other shard, so it must not push the reading from a trained checkpoint
    hours down the queue.
    """
    name = directory.name
    return float("inf") if name == "final" else int(name.split("-")[-1])


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


def still_running(pid):
    """True while that process exists; a finished one leaves no /proc entry.

    /proc/<pid> is a directory, so this is a presence check on the path, not a
    regular-file check on it.
    """
    return Path(f"/proc/{pid}").exists()


def available_gpus(gpus, tasks, busy):
    """GPUs no job holds and no training phase reserves.

    A training GPU stays reserved through preparation and smoke, then joins
    evaluation once its own trainer finishes. Adopted children count as holding
    their GPU: they are not in the queue's own process table, so leaving them
    out would put a second evaluation on a card that already has one.
    """
    occupied = {job["gpu"] for job in busy if job["gpu"] is not None}
    return [gpu for gpu in gpus if gpu not in occupied
            and not any(spec["gpu"] == gpu and spec["phase"] in ("prepare", "smoke", "train")
                        for spec in tasks.values())]


def live_children(status, root):
    """Jobs the previous queue started that this one must not start again.

    A restarted queue must not put a second trainer on a GPU that already has
    one. The predecessor's status file names each pid; /proc then confirms that
    pid still runs the module its key implies and, for an evaluation, still
    writes this run's own output directory, which a recycled pid would not.
    """
    modules = dict(prepare="prepare_trajectories", smoke="trajectory_smoke", train="train")
    alive = {}
    for key, running in (status or {}).get("running", {}).items():
        command = Path(f"/proc/{running['pid']}/cmdline")
        if not command.is_file():
            continue
        try:
            actual = command.read_bytes().decode(errors="replace")
        except OSError:
            continue
        module = modules.get(key.split("/")[-1], "trajectory_eval")
        if MODULE + module not in actual:
            continue
        if module == "trajectory_eval" and str(root / "evaluation" / key) not in actual:
            continue
        alive[key] = running
    return alive


def main():
    parser = ArgumentParser()
    for arg in ("run-root", "model-path"):
        parser.add_argument("--" + arg, type=Path, required=True)
    training_inputs = ("graph-source", "blocks-source-manifest", "blocks-official", "blocks-q", "blocks-q-data")
    for arg in training_inputs:
        parser.add_argument("--" + arg, type=Path)
    parser.add_argument("--gpus", type=int, nargs="+", default=[0, 1, 2, 3])
    parser.add_argument("--evaluation-only", action="store_true",
                        help="Read existing manifests and checkpoints; never prepare data or train")
    parser.add_argument("--include-prefix", action="append", default=[],
                        help="Evaluation-only worker owns these keys; the main queue must delegate them")
    parser.add_argument("--status-file", type=Path,
                        help="Separate status path for a worker on another host sharing this run")
    parser.add_argument("--prepare-workers", type=int, default=1,
                        help="Forked processes each preparation task may use; the two tasks run at once")
    parser.add_argument("--delegate-prefix", action="append", default=[],
                        help="Key prefix another machine owns, so this queue neither claims it nor "
                             "aggregates its shards; the owner aggregates them itself. A prefix may "
                             "also name a task's trainer, e.g. blocks/train, which that machine runs "
                             "instead: its checkpoints still land in the shared run root, so this "
                             "queue keeps evaluating them without holding a card for a trainer it "
                             "will never start")
    args = parser.parse_args()
    if args.evaluation_only:
        if not args.include_prefix or args.status_file is None:
            parser.error("--evaluation-only requires --include-prefix and --status-file")
    elif args.include_prefix or len(args.gpus) < 2 or any(getattr(args, name.replace("-", "_")) is None for name in training_inputs):
        parser.error("training queue requires its data/map inputs and at least two GPUs; --include-prefix is evaluation-only")
    root = args.run_root.resolve()
    previous = args.status_file.resolve() if args.status_file else root / "queue_status.json"
    if args.evaluation_only and previous == root / "queue_status.json":
        parser.error("evaluation worker must not overwrite the main queue's status")
    previous.parent.mkdir(parents=True, exist_ok=True)
    logs = root / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", TOKENIZERS_PARALLELISM="false")
    tasks = {"path": dict(config=CONFIGS / "path_single_long.json", gpu=args.gpus[0],
                 prepare=["--source-root", str(args.graph_source)], train=["--source-root", str(args.graph_source)]),
             "blocks": dict(config=CONFIGS / "blocks1000_long.json", gpu=args.gpus[min(1, len(args.gpus) - 1)],
                 prepare=["--source-manifest", str(args.blocks_source_manifest), "--official-data", str(args.blocks_official),
                          "--q-checkpoint", str(args.blocks_q), "--q-training-data", str(args.blocks_q_data)],
                 train=["--q-checkpoint", str(args.blocks_q)])}
    processes, pending, known, completed, failed = {}, [], set(), [], []
    if args.evaluation_only:
        tasks = {task: spec for task, spec in tasks.items()
                 if any(prefix.startswith(task + "/") for prefix in args.include_prefix)}
        if not tasks:
            parser.error("include prefixes must start with path/ or blocks/")
    watched = live_children(json.loads(previous.read_text()) if previous.is_file() else None, root)

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
    # redoing it: prepared data, a passing smoke test, the newest published
    # checkpoint, which train.py continues rather than restarting from scratch,
    # and any child still running, which it watches instead of starting again.
    for task, spec in tasks.items():
        manifest = root / task / "data/manifest.json"
        if args.evaluation_only:
            spec["manifest"] = json.loads(manifest.read_text())
            spec["phase"] = "evaluation"
            continue
        if task + "/prepare" in watched:
            spec["phase"] = "prepare"
            continue
        if not manifest.is_file():
            launch(task + "/prepare", "prepare_trajectories", ["--config", str(spec["config"]), "--model-path", str(args.model_path),
                   "--out", str(root / task / "data"), "--workers", str(args.prepare_workers), *spec["prepare"]])
            spec["phase"] = "prepare"
            continue
        spec["manifest"] = json.loads(manifest.read_text())
        if task + "/smoke" in watched:
            spec["phase"] = "smoke"
            continue
        if not (root / task / "smoke.json").is_file():
            launch(task + "/smoke", "trajectory_smoke", ["--manifest", str(manifest),
                "--model-path", str(args.model_path), "--out", str(root / task / "smoke.json")], spec["gpu"])
            spec["phase"] = "smoke"
            continue
        if (task + "/train").startswith(tuple(args.delegate_prefix)):
            # Another machine owns this task's trainer. Its checkpoints arrive in the
            # shared run root, which the scan below already picks up, so this queue
            # keeps reading them without holding a card for a trainer it will never
            # start; "delegated" is outside available_gpus' reserve set for exactly
            # that reason, while the manifest is loaded so evaluation still runs.
            spec["phase"] = "delegated"
            continue
        if task + "/train" in watched:
            spec["phase"] = "train"
            continue
        if (root / task / "training/models/final/evaluation_ready.json").is_file():
            spec["phase"] = "trained"
            continue
        launch_train(task, spec, resume_point(root / task / "training/models"))
        spec["phase"] = "train"
    known |= completed_shards(root / "evaluation")
    # Work already in flight is accounted for; queueing it again would run a
    # second copy of a shard whose live process is still writing that directory.
    known |= set(watched)
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
        for key, running in list(watched.items()):
            if still_running(running["pid"]):
                continue
            del watched[key]
            print(json.dumps(dict(event="adopted_finished", key=key, pid=running["pid"])), flush=True)
            task = key.split("/")[0]
            if key == task + "/train":
                # Its exit code died with the previous queue, so the published
                # final checkpoint is what says the trainer finished cleanly.
                final = root / task / "training/models/final/evaluation_ready.json"
                tasks[task]["phase"] = "trained" if final.is_file() else "failed"
                if tasks[task]["phase"] == "failed":
                    failed.append(dict(key=key, exit_code=None, log=running["log"]))
            elif key.count("/") > 1 and not (root / "evaluation" / key / "summary.json").is_file():
                # A shard that stopped without its summary never finished; forget
                # it so the scan below puts it back in the queue.
                known.discard(key)
        for task, spec in tasks.items():
            if "manifest" not in spec:
                continue
            published = list((root / task / "training/models").glob("*/evaluation_ready.json"))
            for ready in sorted(published, key=lambda path: checkpoint_progress(path.parent), reverse=True):
                for job in checkpoint_jobs(task, ready.parent, spec["manifest"], root):
                    if args.include_prefix and not job["key"].startswith(tuple(args.include_prefix)):
                        continue
                    if job["key"] in known or job["key"].startswith(tuple(args.delegate_prefix)):
                        continue
                    known.add(job["key"])
                    pending.append(job)
        pending.sort(key=job_priority)
        for gpu in available_gpus(args.gpus, tasks, [*processes.values(), *watched.values()]):
            if pending:
                job = pending.pop(0)
                clear_partial_shard(root / "evaluation" / job["key"])
                launch(job["key"], "trajectory_eval", [*job["args"], "--model-path", str(args.model_path)], gpu)
        status = dict(pid=os.getpid(), tasks={k: dict(phase=v["phase"], gpu=v["gpu"]) for k, v in tasks.items()},
            running={**{k: dict(pid=v["process"].pid, gpu=v["gpu"], log=v["log"]) for k, v in processes.items()},
                     **watched},
            pending_evaluations=len(pending), completed=completed, failed=failed)
        atomic_json(previous, status)
        waiting_for_final = args.evaluation_only and any(
            not (root / task / "training/models/final/evaluation_ready.json").is_file() for task in tasks)
        if not processes and not pending and not watched and not waiting_for_final:
            break
        time.sleep(15)
    # Aggregate across shards so paired target changes can cross shard boundaries.
    from .trajectory_eval import aggregate
    for task, spec in tasks.items():
        if "manifest" not in spec:
            continue
        groups = defaultdict_list(root / "evaluation" / task)
        for group, shards in groups.items():
            key = group.relative_to(root / "evaluation").as_posix() + "/"
            if (args.include_prefix and not key.startswith(tuple(args.include_prefix))) or key.startswith(tuple(args.delegate_prefix)):
                # Delegated shards are still arriving; aggregating now would freeze a
                # partial battery under the group's final summary.json. Its owner does it.
                continue
            results = [json.loads(line) for shard in shards for line in (shard / "cases.jsonl").read_text().splitlines()]
            atomic_json(group / "summary.json", aggregate(results, spec["manifest"]))
    completion = previous.with_name(previous.stem + "_completion.json") if args.status_file else root / "completion.json"
    atomic_json(completion, dict(success=not failed, failed=failed, completed_jobs=len(completed)))
    if failed:
        raise SystemExit(1)


def defaultdict_list(task_root):
    groups = {}
    for summary in task_root.glob("*/*/*/summary.json"):
        groups.setdefault(summary.parent.parent, []).append(summary.parent)
    return groups


if __name__ == "__main__":
    main()
