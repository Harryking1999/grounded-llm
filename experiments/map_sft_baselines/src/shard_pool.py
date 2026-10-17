"""One authoritative shard directory, accessed locally or through SFTP.

Claims never expire on a timer: a slow worker must not lose its ownership.
Completed cases are transferred before the completion marker is published.
"""
import json
import uuid
from pathlib import Path, PurePosixPath


class ShardPool:
    def __init__(self, root, files=None):
        self.root = PurePosixPath(root) if files else Path(root)
        self.files = files

    def exists(self, path):
        if not self.files:
            return path.exists()
        try:
            self.files.stat(str(path))
            return True
        except FileNotFoundError:
            return False

    def read(self, path):
        if not self.files:
            return json.loads(path.read_text())
        with self.files.open(str(path)) as stream:
            return json.load(stream)

    def mkdirs(self, path):
        if not self.files:
            path.mkdir(parents=True, exist_ok=True)
            return
        if self.exists(path):
            return
        self.mkdirs(path.parent)
        try:
            self.files.mkdir(str(path))
        except OSError:
            if not self.exists(path):
                raise

    def write(self, path, value):
        temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.part')
        if not self.files:
            temporary.write_text(json.dumps(value))
            temporary.replace(path)
        else:
            with self.files.open(str(temporary), 'w') as stream:
                stream.write(json.dumps(value))
            self.files.posix_rename(str(temporary), str(path))

    def output(self, job):
        step, _, base, mode, start, stop, _, _ = job
        return self.root / 'evaluation' / base / f'checkpoint-{step}' / f'{mode}_{start:04d}_{stop:04d}'

    def claim_path(self, job):
        output = self.output(job)
        return output.with_name(output.name + '.worker.claim')

    def acquire_next(self, jobs, node):
        """Round-robin baselines across nodes; preserve priority within each baseline.

        Only claim selection holds this short shared lock, never checkpoint transfer
        or evaluation. A crashed dispatcher requires explicit lock recovery.
        """
        self.mkdirs(self.root)
        lock = self.root / 'evaluation_dispatch.lock'
        try:
            if self.files:
                self.files.mkdir(str(lock))
            else:
                lock.mkdir()
        except OSError:
            if self.exists(lock):
                return None
            raise
        try:
            cursor = self.root / 'evaluation_dispatch.json'
            previous = self.read(cursor).get('baseline') if self.exists(cursor) else None
            baselines = sorted({job[2] for job in jobs})
            if previous in baselines:
                index = baselines.index(previous) + 1
                baselines = baselines[index:] + baselines[:index]
            for baseline in baselines:
                for job in jobs:
                    if job[2] != baseline:
                        continue
                    owner = self.acquire(job, node)
                    if owner:
                        self.write(cursor, dict(baseline=baseline, node=node))
                        return job, owner
            return None
        finally:
            if self.files:
                self.files.rmdir(str(lock))
            else:
                lock.rmdir()

    def acquire(self, job, node, pid=None):
        output, path = self.output(job), self.claim_path(job)
        self.mkdirs(output.parent)
        if self.exists(output / 'summary.json'):
            return None
        try:
            if self.files:
                self.files.mkdir(str(path))
            else:
                path.mkdir()
        except OSError:
            if self.exists(path):
                return None
            raise
        owner = dict(node=node, token=uuid.uuid4().hex, pid=pid)
        self.write(path / 'owner.json', owner)
        if self.exists(output / 'summary.json'):
            self.release(job, owner)
            return None
        return owner

    def owner(self, job):
        path = self.claim_path(job) / 'owner.json'
        return self.read(path) if self.exists(path) else None

    def verify(self, job, owner):
        actual = self.owner(job)
        if not actual or actual['token'] != owner['token'] or actual['node'] != owner['node']:
            raise RuntimeError('Shard ownership changed; refusing to publish or release')

    def set_pid(self, job, owner, pid):
        self.verify(job, owner)
        owner['pid'] = pid
        self.write(self.claim_path(job) / 'owner.json', owner)

    def release(self, job, owner):
        self.verify(job, owner)
        path = self.claim_path(job)
        if self.files:
            self.files.remove(str(path / 'owner.json'))
            self.files.rmdir(str(path))
        else:
            (path / 'owner.json').unlink()
            path.rmdir()

    def publish(self, job, owner):
        self.verify(job, owner)
        local = Path(job[-1])
        summary = json.loads((local / 'summary.json').read_text())
        step, _, base, mode, start, stop, _, _ = job
        cases = [json.loads(line) for line in (local / 'cases.jsonl').read_text().splitlines()]
        if (summary['baseline'] != base or summary['mode'] != mode or
                summary['checkpoint']['step'] != step or len(cases) != stop - start or
                len({case['trajectory_id'] for case in cases}) != stop - start):
            raise ValueError('Completed shard identity or case count differs from its claim')
        output = self.output(job)
        if self.files:
            self.mkdirs(output)
            # Keep the claim throughout transfer; summary is the only completion marker.
            for name in ('cases.jsonl', 'summary.json'):
                destination = output / name
                temporary = destination.with_name(name + '.' + owner['token'] + '.part')
                self.files.put(str(local / name), str(temporary))
                self.files.posix_rename(str(temporary), str(destination))
        elif local.resolve() != Path(output).resolve():
            raise ValueError('Local coordinator output must be the authoritative directory')
        self.release(job, owner)

    def checkpoints(self):
        if not self.files:
            return [self.read(p) for p in (self.root / 'models').glob('B*/checkpoint-*/evaluation_ready.json')
                    if not p.with_name('weights_retired.json').exists()]
        result = []
        for base in self.files.listdir(str(self.root / 'models')):
            if not base.startswith('B'):
                continue
            for name in self.files.listdir(str(self.root / 'models' / base)):
                marker = self.root / 'models' / base / name / 'evaluation_ready.json'
                if (name.startswith('checkpoint-') and self.exists(marker) and
                        not self.exists(marker.with_name('weights_retired.json'))):
                    result.append(self.read(marker))
        return result

    def pull_checkpoint(self, job):
        if not self.files:
            return
        step, _, base, _, _, _, checkpoint, _ = job
        checkpoint = Path(checkpoint)
        if (checkpoint / 'evaluation_ready.json').exists():
            return
        checkpoint.mkdir(parents=True, exist_ok=True)
        source = self.root / 'models' / base / f'checkpoint-{step}'
        for name in ('weights.pt', 'state.json', 'evaluation_ready.json'):
            temporary = checkpoint / (name + '.part')
            self.files.get(str(source / name), str(temporary))
            temporary.replace(checkpoint / name)
