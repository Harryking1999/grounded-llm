"""Shared-pool evaluation workers; preserve live workers across queue changes."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from filelock import FileLock
from grounded_llm.core.files import atomic_json
from .shard_pool import ShardPool

MODULE = 'experiments.map_sft_baselines.src.evaluate'


def worker_gpu(slot):
    return int(str(slot).split(':')[0])


def free_worker_slot(active, gpu, limit):
    for index in range(limit):
        slot = str(gpu) if index == 0 else f'{gpu}:{index}'
        if slot not in active:
            return slot
    return None


def training_completed(path):
    """Only successful final export releases an explicitly assigned GPU group."""
    if not path.exists():
        return False
    state = json.loads(path.read_text())
    if state.get('status') != 'completed':
        raise RuntimeError('Training did not complete successfully; evaluation handoff blocked')
    return True


def evaluation_completed(path):
    """Require all configured modes and cases in an exact checkpoint summary."""
    if not path.exists():
        return False
    config = json.loads((path.parents[3] / 'config.json').read_text())
    from .contract import cohort_size, evaluation_modes
    summary = json.loads(path.read_text())
    step = int(path.parent.name.split('-')[-1])
    baseline = path.parent.parent.name
    return (summary.get('step') == step and
            all(summary.get('metrics', {}).get(mode, {}).get('tasks') == cohort_size(config, mode)
                for mode in evaluation_modes(config, baseline)))


def released_training_gpus(path, gpus):
    """Fail closed until the training scheduler has assigned pending work."""
    if not path.exists():
        return []
    state = json.loads(path.read_text())
    if state.get('pending') or state.get('status') not in ('running', 'finished'):
        return []
    occupied = {gpu for item in state.get('active', {}).values() for gpu in item['gpus']}
    return [gpu for gpu in gpus if gpu not in occupied]


class ExistingWorker:
    def __init__(self, pid, output, completion_marker='summary.json'):
        self.pid, self.output = pid, output
        self.completion_marker = completion_marker
        self.path = Path(f'/proc/{pid}/cmdline')
        self.identity = self.path.read_bytes()

    def poll(self):
        try:
            if self.path.read_bytes() == self.identity:
                return None
        except FileNotFoundError:
            pass
        return 0 if (self.output / self.completion_marker).exists() else 1


def existing_jobs(args, config):
    """Use actual worker arguments, including checkpoint, rather than stale PIDs."""
    from .contract import steps_per_epoch
    for path in Path('/proc').glob('[0-9]*/cmdline'):
        try:
            argv = path.read_bytes().decode().split('\0')
            if MODULE not in argv:
                continue
            def value(flag):
                return argv[argv.index(flag) + 1]
            output = Path(value('--out'))
            if not output.is_relative_to(args.root / 'evaluation'):
                continue
            # Mixed-task shards spawn a task evaluator below their output.
            # Only the outer process owns the GPU and the shard claim.
            if len(output.relative_to(args.root / 'evaluation').parts) != 3:
                continue
            checkpoint = Path(value('--checkpoint'))
            step = int(checkpoint.name.split('-')[-1])
            job = (step, step / steps_per_epoch(config), value('--baseline'), value('--mode'),
                   int(value('--start')), int(value('--stop')), checkpoint, output)
            environment = dict(item.split('=', 1) for item in (path.parent / 'environ').read_bytes().decode().split('\0') if '=' in item)
            yield int(environment['CUDA_VISIBLE_DEVICES']), int(path.parent.name), job
        except (FileNotFoundError, ProcessLookupError):
            continue


def run(args, config):
    from .queue import evaluation_jobs, collect
    gpus = [int(gpu) for gpu in args.gpus.split(',')] if args.gpus else config['execution']['evaluation_gpus']
    training_queue = getattr(args, 'training_queue', None)
    released_config = config['execution'].get('after_training_evaluation', {})
    if training_queue and not args.gpus:
        gpus = released_config['gpus']
    if len(set(gpus)) != len(gpus):
        raise ValueError('Evaluation GPUs must be distinct')
    workers_per_gpu = getattr(args, 'workers_per_gpu', 1)
    worker_memory = getattr(args, 'worker_memory_gib', None)
    poll_seconds = getattr(args, 'poll_seconds', None)
    poll_seconds = config['execution']['poll_seconds'] if poll_seconds is None else poll_seconds
    if workers_per_gpu < 1 or poll_seconds <= 0:
        raise ValueError('Worker count and polling interval must be positive')
    if workers_per_gpu > 1 and (worker_memory is None or worker_memory <= 0):
        raise ValueError('Concurrent workers require an explicit positive memory allowance')
    root = args.root.resolve()
    args.root = root
    root.mkdir(parents=True, exist_ok=True)
    # Node IDs only label owners; the same coordinator directory arbitrates every node.
    status = root / 'evaluation_queue.json'
    with FileLock(str(root / 'evaluation_queue.lock'), timeout=0):
        prerequisites = getattr(args, 'after_evaluation', None)
        while prerequisites and not all(evaluation_completed(path) for path in prerequisites):
            atomic_json(status, dict(status='waiting_for_evaluation', pid=os.getpid(),
                updated=time.time(), prerequisites=[str(path) for path in prerequisites],
                gpus=gpus, active={}, failed=[]))
            time.sleep(poll_seconds)
        completion = getattr(args, 'after_training', None)
        if completion:
            try:
                while not training_completed(completion):
                    atomic_json(status, dict(status='waiting_for_training', pid=os.getpid(),
                        updated=time.time(), completion=str(completion), gpus=gpus, active={}, failed=[]))
                    time.sleep(config['execution']['poll_seconds'])
            except Exception as error:
                atomic_json(status, dict(status='failed', pid=os.getpid(), updated=time.time(),
                    active={}, failed=[dict(error=str(error))]))
                raise
        jump = target = files = None
        if args.coordination_profile:
            import paramiko
            from experiments.llm_task_sft.src.bridge import connect
            profile = json.loads(args.coordination_profile.read_text())
            credentials = json.load(sys.stdin)
            jump, target = connect(profile, credentials)
            files = paramiko.SFTPClient.from_transport(target.get_transport(), window_size=64*1024*1024, max_packet_size=1024*1024)
            pool = ShardPool(profile['target_run'], files)
        else:
            pool = ShardPool(root)
        active, failed, preparing = {}, [], None

        def snapshot(state='running', pending=0):
            atomic_json(status, dict(status=state, node=args.node_id, pid=os.getpid(), updated=time.time(),
                active={str(gpu):dict(pid=item[0].pid, job=[str(x) if isinstance(x, Path) else x for x in item[1]],
                                     owner=item[3], gpu=worker_gpu(gpu)) for gpu,item in active.items()},
                workers_per_gpu=workers_per_gpu, worker_memory_gib=worker_memory,
                poll_seconds=poll_seconds,
                pending_shards=pending, failed=failed, preparing=preparing))

        try:
            previous = json.loads(status.read_text()) if status.exists() else {}
            workers = list(existing_jobs(args, config))
            if workers and not args.resume_workers:
                raise RuntimeError('Live workers exist; restart queue with --resume-workers to adopt them')
            if args.resume_workers:
                live_pids = {pid for _,pid,_ in workers}
                old_preparing = previous.get('preparing')
                if old_preparing:
                    job = list(old_preparing['job']);job[-2:] = map(Path, job[-2:])
                    owner = old_preparing['owner']
                    live_outputs = {str(item[-1][-1]) for item in workers}
                    if owner['node'] != args.node_id:
                        raise RuntimeError('Preparing claim belongs to another node')
                    if str(job[-1]) not in live_outputs and pool.owner(job):
                        pool.release(job, owner)
                # Explicit restart may finish publishing a completed, exited worker.
                for record in previous.get('active', {}).values():
                    if 'owner' not in record or record['pid'] in live_pids:
                        continue
                    job = list(record['job']);job[-2:] = map(Path, job[-2:])
                    owner = record['owner']
                    if owner['node'] != args.node_id:
                        raise RuntimeError('Previous queue belongs to a different node')
                    if not pool.owner(job):
                        continue
                    if (job[-1] / 'summary.json').exists():
                        pool.publish(job, owner)
                    else:
                        # Only explicit restart and a dead local process permit release.
                        pool.release(job, owner)
                for gpu,pid,job in workers:
                    slot = free_worker_slot(active, gpu, workers_per_gpu)
                    if gpu not in gpus or slot is None:
                        raise RuntimeError('Live worker GPU conflicts with the new queue')
                    owner = pool.owner(job)
                    if owner:
                        if owner['node'] != args.node_id or owner['pid'] not in (None, pid):
                            raise RuntimeError('Live worker claim belongs to another owner')
                        pool.set_pid(job, owner, pid)
                    else:
                        owner = pool.acquire(job, args.node_id, pid)
                        if not owner:
                            # A worker just finished while its arguments were read.
                            if (job[-1] / 'summary.json').exists():
                                continue
                            raise RuntimeError('Cannot reserve the shard of an existing worker')
                    active[slot] = (ExistingWorker(pid, job[-1]), job, None, owner)
                snapshot()
            while True:
                for gpu,item in list(active.items()):
                    process,job,log,owner = item
                    code = process.poll()
                    if code is None:
                        continue
                    if log:
                        log.close()
                    if code:
                        failed.append(dict(pid=process.pid, job=str(job[2:6]), code=code))
                        snapshot('failed')
                        raise RuntimeError('Evaluation worker failed; claim retained for explicit recovery')
                    pool.publish(job, owner)
                    del active[gpu]
                jobs = evaluation_jobs(root, config, pool.checkpoints())
                if training_queue:
                    jobs = [job for job in jobs if job[3] in released_config['modes']]
                busy = {str(item[1][-1]) for item in active.values()}
                available = released_training_gpus(training_queue, gpus) if training_queue else gpus
                for gpu in available:
                    slot = free_worker_slot(active, gpu, workers_per_gpu)
                    if slot is None:
                        continue
                    memory = subprocess.check_output(['nvidia-smi',f'--id={gpu}','--query-gpu=memory.used,memory.free','--format=csv,noheader,nounits'],text=True)
                    used, free = map(int, memory.strip().split(','))
                    own_workers = sum(worker_gpu(key) == gpu for key in active)
                    # Never share a GPU occupied by another queue. Extra workers
                    # can share only this dispatcher's already-owned GPU.
                    if (not own_workers and used >= 1000) or (worker_memory is not None and free < worker_memory * 1024):
                        continue
                    selected = pool.acquire_next([job for job in jobs if str(job[-1]) not in busy], args.node_id)
                    if selected:
                        job, owner = selected
                        preparing = dict(job=[str(x) if isinstance(x,Path) else x for x in job],owner=owner)
                        snapshot('preparing_checkpoint',len(jobs))
                        pool.pull_checkpoint(job)
                        step,epoch,base,mode,start,stop,checkpoint,out = job
                        out.mkdir(parents=True,exist_ok=True)
                        log = (out/'worker.log').open('a')
                        command = [sys.executable,'-u','-m',MODULE,'--config',str(args.config),'--data',str(args.data),
                            '--model-path',str(args.model_path),'--q-checkpoint',str(args.q_checkpoint),
                            '--checkpoint',str(checkpoint),'--out',str(out),'--baseline',base,'--mode',mode,
                            '--start',str(start),'--stop',str(stop)]
                        process = subprocess.Popen(command,env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu)),stdout=log,stderr=subprocess.STDOUT)
                        pool.set_pid(job, owner, process.pid)
                        active[slot] = (process,job,log,owner)
                        preparing = None
                        busy.add(str(out));snapshot(pending=len(jobs))
                if not files:
                    collect(root, config)
                snapshot('running' if active else 'waiting_for_checkpoints',len(jobs))
                time.sleep(poll_seconds)
        except Exception as error:
            failed.append(dict(error=str(error)))
            snapshot('failed')
            raise
        finally:
            if files: files.close()
            if target: target.close()
            if jump: jump.close()
