"""Sync evaluation artifacts through a staging SSH host; never launch GPU jobs."""

import argparse
import getpass
import json
import os
from pathlib import Path
import posixpath
import shlex
import subprocess
import sys
import tarfile
import time

from .blocks_checkpoint_queue import checkpoints, owns_shard
from .trajectory_queue import atomic_json


def exists(files, path):
    try:
        files.stat(str(path))
        return True
    except FileNotFoundError:
        return False


def import_shard(files, folder, output):
    """A failed case transfer must never publish a completion summary."""
    if (output/'summary.json').exists():
        return False
    if not all(exists(files, folder+'/'+name) for name in ('cases.jsonl', 'summary.json')):
        return False
    output.mkdir(parents=True, exist_ok=True)
    for name in ('cases.jsonl', 'summary.json'):
        temporary = output/(name+'.part')
        files.get(folder+'/'+name, str(temporary))
        temporary.replace(output/name)
    return True


def missing_owned_shards(source_run, step, contract, training, evaluation):
    spec = contract['auxiliary_evaluation']
    shard_tasks = training['evaluation']['shard_tasks']
    for task in evaluation['tasks'].values():
        if task.get('filter') and not spec.get('terminal_tasks'):
            continue
        if task['split'] != 'test' and not spec.get('terminal_tasks'):
            continue
        split = 'initial_goal' if task.get('filter') else task['split']
        for mode in task.get('modes', [task.get('mode')]):
            for start in range(0, task['tasks'], shard_tasks):
                if not owns_shard(start, shard_tasks, spec['shard_modulo'], spec['shard_remainders']):
                    continue
                stop = min(start+shard_tasks, task['tasks'])
                summary = source_run / f'evaluation_half_epoch/step-{step}/{split}_{mode}/{start:05d}_{stop:05d}/summary.json'
                if not summary.is_file():
                    return True
    return False


class Bridge:
    def __init__(self, profile, contract_path, credentials):
        # Paramiko belongs to the separate transfer environment, not the ML environment.
        import paramiko
        self.profile = profile
        self.contract = json.loads(contract_path.read_text())
        self.training = json.loads((contract_path.parent / self.contract['source_training_contract']).read_text())
        self.evaluation = json.loads((contract_path.parent / self.contract['evaluation_contract']).read_text())
        self.source = Path(profile['source_workspace']) / 'runs' / self.contract['run']
        self.target_run = posixpath.join(profile['target_workspace'], 'runs', self.contract['run'])
        self.credentials = credentials
        self.connections = []

        def connect(spec, password, sock=None):
            connection = paramiko.SSHClient()
            self.connections.append(connection)
            connection.load_host_keys(profile['known_hosts'])
            connection.set_missing_host_key_policy(paramiko.RejectPolicy())
            connection.connect(spec['host'], port=spec.get('port', 22), username=spec['user'],
                               password=password, sock=sock, allow_agent=False,
                               look_for_keys=False, timeout=20)
            connection.get_transport().set_keepalive(60)
            return connection

        try:
            self.jump = connect(profile['jump'], credentials['jump'])
            channel = self.jump.get_transport().open_channel('direct-tcpip',
                (profile['target']['host'], profile['target'].get('port', 22)), ('127.0.0.1', 0))
            self.target = connect(profile['target'], credentials['target'], channel)
            self.jump_files = self.jump.open_sftp()
            self.target_files = self.target.open_sftp()
            self.execute('mkdir -p '+shlex.quote(self.target_run+'/evaluation_half_epoch'))
            self.jump_command('mkdir -p '+shlex.quote(profile['stage_dir']))
        except Exception:
            self.close()
            raise

    def close(self):
        for connection in reversed(self.connections):
            connection.close()

    @staticmethod
    def command(connection, command):
        _, stdout, stderr = connection.exec_command(command)
        output = stdout.read().decode()
        error = stderr.read().decode()
        code = stdout.channel.recv_exit_status()
        if code:
            raise RuntimeError(f'Remote command failed ({code}): {error[-600:]}')
        return output

    def execute(self, command):
        return self.command(self.target, command)

    def jump_command(self, command):
        return self.command(self.jump, command)

    def stage_and_copy(self, source, destination):
        staged = posixpath.join(self.profile['stage_dir'], source.name)
        with source.open('rb') as incoming, self.jump_files.open(staged+'.part', 'wb') as outgoing:
            outgoing.set_pipelined(True)
            while block := incoming.read(1024*1024):
                outgoing.write(block)
        if exists(self.jump_files, staged):
            self.jump_files.remove(staged)
        self.jump_files.rename(staged+'.part', staged)
        self.execute('mkdir -p '+shlex.quote(posixpath.dirname(destination)))
        target = self.profile['target']
        command = ['scp', '-o', 'StrictHostKeyChecking=yes', '-P', str(target.get('port', 22)),
                   staged, f"{target['user']}@{target['host']}:{destination}.part"]
        channel = self.jump.get_transport().open_session()
        channel.get_pty()
        channel.exec_command(shlex.join(command))
        buffer, sent = b'', False
        try:
            while not channel.exit_status_ready():
                if channel.recv_ready():
                    buffer = (buffer+channel.recv(65536))[-65536:]
                    if b'password:' in buffer.lower() and not sent:
                        channel.send(self.credentials['target']+'\n')
                        sent = True
                time.sleep(.1)
            if channel.recv_exit_status():
                raise RuntimeError('Staging-to-target copy failed; inspect SSH connectivity')
        finally:
            channel.close()
        self.execute('mv -f '+shlex.quote(destination+'.part')+' '+shlex.quote(destination))

    def synchronize(self):
        copied, imported = [], []
        continuation = self.contract['continuation_output']+'/models'
        # Reuse the queue's publication/half-epoch selection; no second scheduler.
        selected = checkpoints(self.source/'training/models', self.source/continuation)
        for checkpoint in selected:
            relative = checkpoint['path'].relative_to(self.source).as_posix()
            final = relative == continuation+'/final'
            ready = self.target_run+'/'+relative+'/evaluation_ready.json'
            if exists(self.target_files, ready) or not (final or missing_owned_shards(
                    self.source, checkpoint['step'], self.contract, self.training, self.evaluation)):
                continue
            for name in ('adapter.pt', 'evaluation_ready.json'):
                self.stage_and_copy(checkpoint['path']/name, self.target_run+'/'+relative+'/'+name)
            copied.append(checkpoint['step'])
        root = self.source/'evaluation_half_epoch'
        markers = [path for path in root.glob('step-*/*/*/summary.json')
                   if '.interrupted.' not in str(path) and not exists(self.target_files,
                       self.target_run+'/evaluation_half_epoch/'+path.relative_to(root).as_posix())]
        if markers:
            bundle = Path(self.profile['status']).parent/'completion_markers.tar'
            bundle.parent.mkdir(parents=True, exist_ok=True)
            with tarfile.open(bundle, 'w') as archive:
                for path in markers:
                    archive.add(path, arcname=path.relative_to(root).as_posix())
            destination = self.target_run+'/transfer/completion_markers.tar'
            self.stage_and_copy(bundle, destination)
            self.execute('tar -xf '+shlex.quote(destination)+' -C '+shlex.quote(self.target_run+'/evaluation_half_epoch'))
        remote = self.target_run+'/evaluation_half_epoch'
        for step in self.target_files.listdir(remote):
            if not step.startswith('step-') or not step[5:].isdigit():
                continue
            for mode in ('test_rollout', 'test_reference', 'initial_goal_reference', 'test_no_solution_reference'):
                if not exists(self.target_files, remote+'/'+step+'/'+mode):
                    continue
                for shard in self.target_files.listdir(remote+'/'+step+'/'+mode):
                    if '.interrupted.' in shard or not shard.replace('_', '').isdigit():
                        continue
                    relative = Path(step)/mode/shard
                    folder = remote+'/'+relative.as_posix()
                    output = root/relative
                    if import_shard(self.target_files, folder, output):
                        imported.append(relative.as_posix())
        return dict(copied_checkpoints=copied, imported_shards=imported)

    def queue_states(self):
        auxiliary = self.contract['auxiliary_evaluation']['queue_name']
        with self.target_files.open(self.target_run+f'/evaluation_half_epoch/status.{auxiliary}.json') as handle:
            remote = json.load(handle)
        local = json.loads((self.source/'evaluation_half_epoch/status.json').read_text())
        if any(state['status'] == 'failed' for state in (local, remote)):
            raise RuntimeError('An evaluation queue failed; inspect its existing log')
        return local['status'], remote['status']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', type=Path, required=True, help='Ignored paths/SSH profile, without passwords')
    parser.add_argument('--contract', type=Path, required=True)
    parser.add_argument('--once', action='store_true')
    parser.add_argument('--detach', action='store_true')
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    profile = json.loads(args.profile.read_text())
    status = Path(profile['status'])
    status.parent.mkdir(parents=True, exist_ok=True)
    credentials = json.load(sys.stdin) if args.worker else {
        'jump': getpass.getpass('Staging host password: '),
        'target': getpass.getpass('Target host password: ')}
    if args.detach and not args.worker:
        if os.name != 'posix':
            parser.error('Start detached transfer on the Linux source node')
        log = status.with_suffix('.log')
        command = [sys.executable, '-m', __spec__.name, '--profile', str(args.profile.resolve()),
                   '--contract', str(args.contract.resolve()), '--worker']
        if args.once:
            command.append('--once')
        with log.open('a') as handle:
            child = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=handle,
                                     stderr=subprocess.STDOUT, start_new_session=True)
            child.stdin.write(json.dumps(credentials).encode())
            child.stdin.close()
        print(json.dumps(dict(pid=child.pid, log=str(log), status=str(status))))
        return
    from filelock import FileLock
    with FileLock(str(status.with_suffix('.lock')), timeout=0):
        bridge = None
        try:
            bridge = Bridge(profile, args.contract, credentials)
            while True:
                result = bridge.synchronize()
                local, remote = bridge.queue_states()
                complete = local == remote == 'completed'
                state = dict(status='completed' if complete else 'running', pid=os.getpid(),
                             updated_at=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                             primary_status=local, auxiliary_status=remote, **result)
                atomic_json(status, state)
                print(json.dumps(state), flush=True)
                if complete or args.once:
                    break
                time.sleep(profile.get('interval_seconds', 600))
        except Exception as error:
            message = str(error)
            for value in credentials.values():
                if value:
                    message = message.replace(value, '[redacted]')
            atomic_json(status, dict(status='failed', pid=os.getpid(), error=message))
            # Avoid an unredacted traceback from authentication libraries.
            print(message, file=sys.stderr)
            raise SystemExit(1)
        finally:
            if bridge:
                bridge.close()


if __name__ == '__main__':
    main()
