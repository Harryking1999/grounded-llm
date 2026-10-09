"""Plan, detach, and inspect existing experiment commands from a runtime profile."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

from .trajectory_queue import atomic_json

MODULE = 'experiments.flamingo_map_reader.src.'
ROLES = ('training', 'evaluation', 'auxiliary_evaluation')


def build_plan(contract_path, profile, role):
    contract = json.loads(contract_path.read_text(encoding='utf-8'))
    spec = contract[role]
    code = Path(profile['code_root'])
    run = Path(profile['workspace_root']) / 'runs' / contract['run']
    configs = code / 'experiments/flamingo_map_reader/configs'
    command = [profile['python'], '-m']
    environment = dict(OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
                       TOKENIZERS_PARALLELISM='false')
    if role == 'training':
        gpus = spec['gpus'].split(',')
        if len(set(gpus)) != len(gpus) or not all(gpus):
            raise ValueError('Training GPUs must be distinct')
        if len(gpus) * spec['per_device_batch_size'] != spec['global_batch_size']:
            raise ValueError('GPU count and per-device batch differ from the global batch')
        source = json.loads((contract_path.parent / contract['source_training_contract']).read_text())
        if len(gpus) != source['training']['world_size']:
            raise ValueError('A topology change needs its explicit migration contract')
        command += ['torch.distributed.run', '--standalone',
                    f'--nproc_per_node={len(gpus)}', '-m', MODULE+'train',
                    '--config', str(configs / contract['source_training_contract']),
                    '--manifest', str(run / 'data/manifest.json'),
                    '--q-checkpoint', profile['q_checkpoint'],
                    '--out', str(run / contract['continuation_output']),
                    '--resume', profile.get('resume_checkpoint', str(run / contract['source_checkpoint'])),
                    '--epoch-extension', str(configs / contract['epoch_extension']),
                    '--batch-size', str(spec['per_device_batch_size'])]
        environment['CUDA_VISIBLE_DEVICES'] = spec['gpus']
    else:
        command += [MODULE+'blocks_checkpoint_queue', '--run', str(run),
                    '--gpus', spec['gpus'], '--queue-name', spec['queue_name'],
                    '--shard-modulo', str(spec['shard_modulo']),
                    '--shard-remainders', ','.join(map(str, spec['shard_remainders'])),
                    '--continuation-models', str(run / contract['continuation_output'] / 'models')]
        for flag in ('cache_map_kv', 'terminal_tasks'):
            if spec.get(flag):
                command.append('--'+flag.replace('_', '-'))
        if profile.get('source_path_map'):
            command += ['--source-path-map', profile['source_path_map']]
    command += ['--model-path', profile['model_path']]
    native_status = (run / contract['continuation_output'] / 'status.json' if role == 'training'
                     else run / 'evaluation_half_epoch' / (
                         'status.json' if spec['queue_name'] == 'main' else f"status.{spec['queue_name']}.json"))
    return dict(command=command, cwd=str(code), environment=environment,
                log=profile.get('log', str(run / 'logs' / f'{role}.log')),
                status_path=str(run / f'runner.{role}.json'), native_status=str(native_status), role=role)


def existing_worker(command):
    """Recognize legacy launches too; never interrupt or adopt an existing worker."""
    module = MODULE + ('train' if MODULE+'train' in command else 'blocks_checkpoint_queue')
    output_flag = '--out' if module.endswith('.train') else '--run'
    output = command[command.index(output_flag)+1]
    for path in Path('/proc').glob('[0-9]*/cmdline'):
        if path.parent.name == str(os.getpid()):
            continue
        try:
            argv = path.read_bytes().decode().split('\0')
        except (OSError, UnicodeError):
            continue
        if module in argv and output_flag in argv and argv[argv.index(output_flag)+1] == output:
            return int(path.parent.name)
    return None


def supervise(plan):
    from filelock import FileLock
    path = Path(plan['status_path'])
    path.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(path.with_suffix('.lock')), timeout=0):
        pid = existing_worker(plan['command'])
        if pid:
            raise RuntimeError(f'Existing worker {pid} already owns this run; use status')
        state = dict(status='running', pid=os.getpid(), **plan)
        try:
            child = subprocess.Popen(plan['command'], cwd=plan['cwd'],
                                     env=dict(os.environ, **plan['environment']))
            state.update(worker_pid=child.pid, started_at=datetime.now(timezone.utc).isoformat())
            atomic_json(path, state)
            result = child.wait()
            state.update(status='completed' if result == 0 else 'failed', returncode=result)
        except Exception as error:
            state.update(status='failed', error=str(error))
            raise
        finally:
            state['updated_at'] = datetime.now(timezone.utc).isoformat()
            atomic_json(path, state)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('plan', 'start', 'status'))
    parser.add_argument('--contract', required=True, type=Path)
    parser.add_argument('--profile', required=True, type=Path,
                        help='Ignored runtime JSON: paths, optional resume_checkpoint, never credentials')
    parser.add_argument('--role', required=True, choices=ROLES)
    parser.add_argument('--supervise', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    profile = json.loads(args.profile.read_text(encoding='utf-8'))
    plan = build_plan(args.contract, profile, args.role)
    if args.supervise:
        supervise(plan)
    elif args.action == 'plan':
        print(json.dumps(plan, indent=2))
    elif args.action == 'status':
        path = Path(plan['status_path'])
        native = Path(plan['native_status'])
        state = dict(runner=json.loads(path.read_text()) if path.exists() else None,
                     workload=json.loads(native.read_text()) if native.exists() else None,
                     detected_pid=existing_worker(plan['command']))
        print(json.dumps(state, indent=2))
    else:
        if os.name != 'posix':
            parser.error('Detached GPU jobs must be started on the Linux execution node')
        pid = existing_worker(plan['command'])
        if pid:
            parser.error(f'Existing worker {pid} already owns this run; use status')
        log = Path(plan['log'])
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open('a') as handle:
            process = subprocess.Popen(
                [sys.executable, '-m', MODULE+'experiment_runner', 'start',
                 '--contract', str(args.contract.resolve()), '--profile', str(args.profile.resolve()),
                 '--role', args.role, '--supervise'], cwd=plan['cwd'],
                stdout=handle, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                start_new_session=True)
        print(json.dumps(dict(pid=process.pid, log=str(log), status=plan['status_path'])))


if __name__ == '__main__':
    main()
