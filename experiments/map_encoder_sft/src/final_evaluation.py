"""Give the study's 028 evaluation GPU to finished-endpoint OOD batches.

Reuse the existing evaluator, collector and shard ownership protocol. Training
and the 4090 ID dispatcher continue throughout. Credentials arrive via stdin.
"""
import argparse
from collections import Counter
import copy
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from types import SimpleNamespace

from filelock import FileLock
from grounded_llm.core.files import atomic_json
from experiments.llm_task_sft.src.bridge import connect
from experiments.map_sft_baselines.src.evaluation_queue import ExistingWorker, existing_jobs
from experiments.map_sft_baselines.src.shard_pool import ShardPool

QUEUE = 'experiments.map_sft_baselines.src.queue'


def final_config(base, policy, records):
    counts = Counter(row['cohort'] for row in records)
    if len(records) != policy['tasks'] or dict(counts) != policy['cohort_counts']:
        raise ValueError('Final evaluation cohort differs from the accepted study')
    if len({row['trajectory_id'] for row in records}) != len(records):
        raise ValueError('Final evaluation has duplicate queries')
    config = copy.deepcopy(base)
    config['data']['tasks'] = base['generalization']['task_pool']
    config['evaluation'].update(tasks=policy['tasks'], modes=policy['modes'], probe_tasks=0)
    config['execution']['evaluation_gpus'] = [policy['gpu']]
    return config


def completed(root, baseline, step, config):
    path = root / 'evaluation' / baseline / f'checkpoint-{step}' / 'summary.json'
    if not path.exists():
        return False
    row = json.loads(path.read_text())
    return row.get('step') == step and all(
        row.get('metrics', {}).get(mode, {}).get('tasks') == config['evaluation']['tasks']
        for mode in config['evaluation']['modes'])


def stop_dispatcher(pid, root):
    """Stop only a verified dispatcher, leaving its independent shard workers."""
    path = Path(f'/proc/{pid}/cmdline')
    if not path.exists() or not path.read_bytes():
        return
    argv = path.read_bytes().decode().split('\0')
    if QUEUE not in argv or '--root' not in argv or argv[argv.index('--root') + 1] != str(root):
        raise RuntimeError('PID is not this study evaluation dispatcher')
    os.kill(pid, signal.SIGTERM)
    for _ in range(100):
        if not path.exists() or not path.read_bytes():
            return
        time.sleep(.1)
    raise RuntimeError('Evaluation dispatcher did not exit')


def spawn(argv, code, log, credentials=None):
    env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
               TOKENIZERS_PARALLELISM='false')
    with log.open('a') as stream:
        process = subprocess.Popen(argv, cwd=code, env=env,
            stdin=subprocess.PIPE if credentials is not None else subprocess.DEVNULL,
            stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
    if credentials is not None:
        process.stdin.write(json.dumps(credentials).encode())
        process.stdin.close()
    return process


def drain_id(root, base, profile, credentials, policy, status):
    """Finish/publish any outer shard, including one between Popen and snapshot."""
    state = json.loads((root / 'evaluation_queue.json').read_text())
    stop_dispatcher(state['pid'], root)
    jump, target = connect(profile, credentials)
    files = target.open_sftp()
    files.get_channel().settimeout(60)
    pool = ShardPool(profile['target_run'], files)
    try:
        jobs = {str(job[-1]): (pid, job) for _, pid, job in
                existing_jobs(SimpleNamespace(root=root), base)}
        for item in state.get('active', {}).values():
            job = list(item['job']); job[-2:] = map(Path, job[-2:])
            jobs.setdefault(str(job[-1]), (item['pid'], tuple(job)))
        preparing = state.get('preparing')
        if preparing and str(preparing['job'][-1]) not in jobs:
            job = list(preparing['job']); job[-2:] = map(Path, job[-2:])
            owner = pool.owner(job)
            if owner:
                if owner['node'] != state['node']:
                    raise RuntimeError('Prepared claim belongs to another node')
                pool.release(job, owner)
        active = []
        for pid, job in jobs.values():
            owner = pool.owner(job)
            if owner is None:
                if pool.exists(pool.output(job) / 'summary.json'):
                    continue
                raise RuntimeError('ID worker is missing its claim')
            if owner['node'] != state['node']:
                raise RuntimeError('ID worker claim belongs to another node')
            proc = Path(f'/proc/{pid}/cmdline')
            worker = ExistingWorker(pid, job[-1]) if proc.exists() and proc.read_bytes() else None
            active.append((worker, job, owner))
        while active:
            status('draining_id_shard', remaining=len(active))
            for item in list(active):
                worker, job, owner = item
                if worker is not None and worker.poll() is None:
                    continue
                if not (Path(job[-1]) / 'summary.json').exists():
                    raise RuntimeError('ID worker exited without a complete shard; claim retained')
                pool.publish(job, owner)
                active.remove(item)
            if active:
                time.sleep(policy['drain_poll_seconds'])
    finally:
        files.close(); target.close(); jump.close()
    atomic_json(root / 'evaluation_queue.json', dict(state, status='paused_for_final_evaluation',
        active={}, preparing=None, updated=time.time()))


def run(args, credentials):
    root = args.root.resolve()
    policy = json.loads(args.policy.read_text())
    base = json.loads((root / 'config.json').read_text())
    final = root / policy['output_subdirectory']
    data = args.data.resolve() / policy['data_subdirectory']
    config = final_config(base, policy, json.loads((data / 'evaluation.json').read_text()))
    final.mkdir(exist_ok=True)
    atomic_json(final / 'config.json', config)
    profile = json.loads((root / 'bridge_profile.json').read_text())
    script_code = Path(__file__).resolve().parents[3]
    controller = final / 'controller.json'
    def status(phase, **updates):
        atomic_json(controller, dict(pid=os.getpid(), phase=phase, updated=time.time(), **updates))
    # Preserve the actual ID launch arguments; training remains on its own snapshot.
    original = json.loads((root / 'evaluation_queue.json').read_text())
    argv_path = Path(f"/proc/{original['pid']}/cmdline")
    original_argv = [v for v in argv_path.read_bytes().decode().split('\0') if v]
    original_code = Path(f"/proc/{original['pid']}/cwd").resolve()
    if '--resume-workers' not in original_argv:
        original_argv.append('--resume-workers')
    id_paused = False
    evaluator = None
    with FileLock(str(final / 'controller.lock'), timeout=0):
        try:
            while True:
                endpoints = {}
                for baseline in base['baselines']:
                    path = root / 'training' / baseline / 'finished.json'
                    if not path.exists():
                        continue
                    finished = json.loads(path.read_text())
                    if finished['status'] not in ('completed', 'stopped'):
                        raise RuntimeError('Training did not produce an accepted endpoint')
                    step = finished['step']
                    ck = path.parent / f'checkpoint-{step}'
                    if not all((ck / n).is_file() for n in ('weights.pt', 'evaluation_ready.json', 'state.json')):
                        raise RuntimeError('Finished endpoint lacks complete inference files')
                    endpoints[baseline] = step
                pending = {b:s for b,s in endpoints.items() if not completed(final,b,s,config)}
                if not pending:
                    if len(endpoints) == len(base['baselines']):
                        status('completed', endpoints=endpoints)
                        return
                    status('waiting_for_training', completed_endpoints=endpoints)
                    time.sleep(policy['poll_seconds'])
                    continue
                # Set before drain so exceptions still restore the original ID dispatcher.
                id_paused = True
                drain_id(root, base, profile, credentials, policy, status)
                for baseline, step in pending.items():
                    dest = final / 'models' / baseline / f'checkpoint-{step}'
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    source = root / 'training' / baseline / dest.name
                    if not dest.exists():
                        dest.symlink_to(source)
                argv = [sys.executable, '-u', '-m', QUEUE, 'evaluate', '--config',
                    str(final / 'config.json'), '--root', str(final), '--data', str(data),
                    '--model-path', str(args.model_path), '--q-checkpoint', str(data/'blocks/qmap.pt'),
                    '--node-id', '028-map-encoder-final', '--gpus', str(policy['gpu']), '--resume-workers']
                evaluator = spawn(argv, script_code, final / 'evaluation.log')
                while not all(completed(final,b,s,config) for b,s in pending.items()):
                    if evaluator.poll() is not None:
                        raise RuntimeError('Final evaluation dispatcher exited; inspect retained claims')
                    status('evaluating_final', endpoints=pending, evaluator_pid=evaluator.pid)
                    time.sleep(policy['poll_seconds'])
                # Completed summaries cover all modes; no unfinished shard remains in this batch.
                state = json.loads((final / 'evaluation_queue.json').read_text())
                while state.get('active') or state.get('preparing'):
                    time.sleep(policy['poll_seconds'])
                    state = json.loads((final / 'evaluation_queue.json').read_text())
                stop_dispatcher(evaluator.pid, final)
                evaluator = None
                process = spawn(original_argv, original_code, root/'logs/evaluate_028_after_final.log', credentials)
                runtime = json.loads((root/'evaluation_runtime.json').read_text())
                atomic_json(root/'evaluation_runtime.json',dict(runtime,local_pid=process.pid))
                id_paused = False
        except Exception as error:
            status('failed', error=str(error))
            # A running final dispatcher keeps GPU ownership; never restart ID over it.
            if id_paused and (evaluator is None or evaluator.poll() is not None):
                spawn(original_argv, original_code, root/'logs/evaluate_028_after_final.log', credentials)
            raise


def main():
    parser = argparse.ArgumentParser(__doc__)
    for name in ('root', 'data', 'model-path', 'policy'):
        parser.add_argument('--'+name, type=Path, required=True)
    run(parser.parse_args(), json.load(sys.stdin))


if __name__ == '__main__':
    main()
