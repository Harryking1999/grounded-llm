"""Evaluate published half-epoch adapters; rescan before each GPU assignment."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from filelock import FileLock

from .trajectory_queue import atomic_json


def checkpoints(models):
    found = {}
    # Prefer final when it duplicates the last numbered checkpoint.
    for directory in [models / 'final', *models.glob('checkpoint-*')]:
        ready = directory / 'evaluation_ready.json'
        if not ready.is_file() or not (directory / 'adapter.pt').is_file():
            continue
        value = json.loads(ready.read_text())
        step, epoch = value['step'], value['epoch']
        half = round(epoch * 2)
        if step <= 0 or half <= 0:
            continue
        # Trainer saves at optimizer boundaries, potentially one step late.
        if epoch < half / 2 - 1e-7 or epoch - half / 2 > epoch / step + 1e-6:
            continue
        found.setdefault(step, dict(path=directory, step=step, epoch=epoch))
    return sorted(found.values(), key=lambda item: -item['step'])


def gpu_modes(gpus):
    if len(gpus) not in (3, 4) or len(set(gpus)) != len(gpus) or not all(gpus):
        raise ValueError('Three or four distinct GPUs are required: two rollout, remaining reference')
    return dict(zip(gpus, ['rollout', 'rollout', *(['reference'] * (len(gpus) - 2))]))


def evaluation_modes(checkpoint, manifest):
    modes = [('test', 'rollout'), ('test', 'reference')]
    if checkpoint['path'].name == 'final':
        modes.append(('test_no_solution', 'reference'))
    return [(split, mode, sum(r['split'] == split for r in manifest['records']))
            for split, mode in modes]


def validate_resume_contract(previous, current):
    # Moving to a smaller node changes scheduling, not the evaluated experiment.
    def experiment(contract):
        return {key: value for key, value in contract.items() if key not in ('gpus', 'gpu_modes')}
    if experiment(previous) != experiment(current):
        raise ValueError('Existing queue contract differs')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--model-path', type=Path, required=True)
    parser.add_argument('--gpus', default='0,1,2,3')
    args = parser.parse_args()
    manifest_path = args.run / 'data/manifest.json'
    manifest = json.loads(manifest_path.read_text())
    gpus = args.gpus.split(',')
    assignments = gpu_modes(gpus)
    count = sum(record['split'] == 'test' for record in manifest['records'])
    if manifest['config']['task'] != 'blocks' or count != 510:
        raise ValueError(f'Expected blocks main test with 510 tasks, found {count}')
    root = args.run / 'evaluation_half_epoch'
    root.mkdir(exist_ok=True)
    logs = root / 'logs'
    logs.mkdir(exist_ok=True)
    contract = dict(run=str(args.run), manifest=str(manifest_path),
                    model_path=str(args.model_path), split='test', tasks=count,
                    modes=['rollout', 'reference'], variants=1, epoch_interval=0.5,
                    priority='newest checkpoint first, then rollout, then reference',
                    gpus=gpus, gpu_modes=assignments,
                    shard_tasks=64)
    with FileLock(str(root / '.queue.lock'), timeout=0):
        # A killed queue can leave evaluation workers alive. Do not move their
        # output or launch a second worker against it during a manual restart.
        for command_file in Path('/proc').glob('[0-9]*/cmdline'):
            if command_file.parent.name == str(os.getpid()):
                continue
            try:
                command_line = command_file.read_bytes().replace(b'\0', b' ').decode(errors='replace')
            except OSError:
                continue
            if 'experiments.flamingo_map_reader.src.trajectory_eval' in command_line and str(root) in command_line:
                raise RuntimeError(f'Existing evaluation worker {command_file.parent.name} still owns this output')
        previous = root / 'contract.json'
        if previous.is_file():
            validate_resume_contract(json.loads(previous.read_text()), contract)
        atomic_json(previous, contract)
        busy, failures = {}, []
        while True:
            for gpu, running in list(busy.items()):
                code = running['process'].poll()
                if code is None:
                    continue
                running['log'].close()
                if code or not (running['output'] / 'summary.json').is_file():
                    failures.append(dict(key=running['key'], exit_code=code))
                del busy[gpu]
            if failures:
                # Keep successful and in-flight shards; fail visibly, never silently retry.
                for running in busy.values():
                    running['process'].wait()
                    running['log'].close()
                atomic_json(root / 'status.json', dict(status='failed', failures=failures))
                raise RuntimeError(f'Evaluation failed: {failures}')
            published = checkpoints(args.run / 'training/models')
            jobs = []
            completed = 0
            for checkpoint in published:
                for split, mode, tasks in evaluation_modes(checkpoint, manifest):
                    for start in range(0, tasks, contract['shard_tasks']):
                        stop = min(start + contract['shard_tasks'], tasks)
                        key = f"step-{checkpoint['step']}/{split}_{mode}/{start:05d}_{stop:05d}"
                        output = root / key
                        if (output / 'summary.json').is_file():
                            completed += 1
                        elif not any(value['key'] == key for value in busy.values()):
                            jobs.append((checkpoint, split, mode, start, stop, key, output))
            for gpu in contract['gpus']:
                if gpu in busy or not jobs:
                    continue
                assignment = next((i for i, job in enumerate(jobs)
                                   if job[2] == contract['gpu_modes'][gpu]), None)
                if assignment is None:
                    continue
                checkpoint, split, mode, start, stop, key, output = jobs.pop(assignment)
                if output.exists():
                    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
                    output.rename(output.with_name(output.name + '.interrupted.' + stamp))
                command = [sys.executable, '-m', 'experiments.flamingo_map_reader.src.trajectory_eval',
                           '--manifest', str(manifest_path), '--adapter-checkpoint',
                           str(checkpoint['path'] / 'adapter.pt'), '--model-path', str(args.model_path),
                           '--out', str(output), '--split', split, '--mode', mode,
                           '--variants', '1', '--start', str(start), '--stop', str(stop)]
                log_path = logs / (key.replace('/', '_') + '.log')
                handle = log_path.open('a')
                process = subprocess.Popen(command, env=dict(os.environ, CUDA_VISIBLE_DEVICES=gpu),
                                           stdout=handle, stderr=subprocess.STDOUT,
                                           stdin=subprocess.DEVNULL)
                busy[gpu] = dict(process=process, log=handle, output=output, key=key)
            # Use the same aggregate as the official evaluator for full checkpoint results.
            for checkpoint in published:
                for split, mode, tasks in evaluation_modes(checkpoint, manifest):
                    directory = root / f"step-{checkpoint['step']}" / f'{split}_{mode}'
                    summaries = [directory / f'{start:05d}_{min(start + 64, tasks):05d}' / 'summary.json'
                                 for start in range(0, tasks, 64)]
                    if (directory / 'summary.json').is_file() or not all(p.is_file() for p in summaries):
                        continue
                    from experiments.flamingo_map_reader.src.trajectory_eval import aggregate
                    cases = [json.loads(line) for p in summaries
                             for line in (p.parent / 'cases.jsonl').read_text().splitlines()]
                    if len(cases) != tasks:
                        raise ValueError(f'Incomplete checkpoint aggregate: {len(cases)}')
                    summary = aggregate(cases, manifest)
                    summary['contract'] = dict(contract, split=split, tasks=tasks, checkpoint=str(checkpoint['path']),
                                               step=checkpoint['step'], epoch=checkpoint['epoch'], mode=mode)
                    atomic_json(directory / 'summary.json', summary)
            final_ready = args.run / 'training/models/final/evaluation_ready.json'
            done = final_ready.is_file() and not busy and not jobs
            atomic_json(root / 'status.json', dict(status='completed' if done else 'running',
                       pid=os.getpid(), updated_at=datetime.now(timezone.utc).isoformat(),
                       checkpoints=[dict(step=c['step'], epoch=c['epoch']) for c in published],
                       completed_shards=completed,
                       running={gpu: dict(pid=value['process'].pid, key=value['key'])
                                for gpu, value in busy.items()}, pending_shards=len(jobs)))
            if done:
                return
            time.sleep(15)


if __name__ == '__main__':
    main()
