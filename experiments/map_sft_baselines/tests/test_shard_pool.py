import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import tempfile
import unittest
from experiments.map_sft_baselines.src.shard_pool import ShardPool


class LocalSFTP:
    """Exercise the same remote publication order with real filesystem operations."""
    def __init__(self):
        self.published = []
        self.fail_cases = False
    def stat(self, path): return Path(path).stat()
    def open(self, path, mode='r'): return Path(path).open(mode)
    def mkdir(self, path): Path(path).mkdir()
    def remove(self, path): Path(path).unlink()
    def rmdir(self, path): Path(path).rmdir()
    def posix_rename(self, source, target):
        Path(source).replace(target); self.published.append(Path(target).name)
    def put(self, source, target):
        if self.fail_cases and Path(source).name=='cases.jsonl':
            raise OSError('simulated interrupted case transfer')
        Path(target).write_bytes(Path(source).read_bytes())


class PoolTests(unittest.TestCase):
    def job(self, root):
        return (8,0.5,'B0','RO',0,2,Path(root)/'models/B0/checkpoint-8',
                Path(root)/'evaluation/B0/checkpoint-8/RO_0000_0002')

    def test_concurrent_nodes_only_one_claims(self):
        with tempfile.TemporaryDirectory() as folder:
            pool=ShardPool(folder);job=self.job(folder)
            with ThreadPoolExecutor(max_workers=8) as executor:
                owners=list(executor.map(lambda i:pool.acquire(job,f'node-{i}'),range(8)))
            self.assertEqual(sum(owner is not None for owner in owners),1)
            owner=next(owner for owner in owners if owner)
            pool.release(job,owner)
            self.assertIsNotNone(pool.acquire(job,'next-node'))

    def test_foreign_owner_cannot_release_and_completed_cannot_claim(self):
        with tempfile.TemporaryDirectory() as folder:
            pool=ShardPool(folder);job=self.job(folder);owner=pool.acquire(job,'a')
            with self.assertRaises(RuntimeError):pool.release(job,dict(node='b',token=owner['token']))
            self.assertIsNone(pool.acquire(job,'b'))
            pool.release(job,owner)
            job[-1].mkdir();(job[-1]/'summary.json').write_text('{}')
            self.assertIsNone(pool.acquire(job,'b'))

    def test_remote_publish_cases_before_summary_and_keep_claim_on_failure(self):
        with tempfile.TemporaryDirectory() as authority, tempfile.TemporaryDirectory() as worker:
            files=LocalSFTP();pool=ShardPool(authority,files);job=self.job(worker)
            owner=pool.acquire(job,'aux')
            job[-1].mkdir(parents=True)
            (job[-1]/'cases.jsonl').write_text('\n'.join(json.dumps(dict(trajectory_id=i)) for i in range(2)))
            (job[-1]/'summary.json').write_text(json.dumps(dict(baseline='B0',mode='RO',checkpoint={'step':8})))
            files.fail_cases=True
            with self.assertRaises(OSError):pool.publish(job,owner)
            self.assertFalse(pool.exists(pool.output(job)/'summary.json'))
            self.assertIsNone(pool.acquire(job,'other'))
            files.fail_cases=False;pool.publish(job,owner)
            self.assertEqual(files.published[-2:],['cases.jsonl','summary.json'])
            self.assertFalse(pool.exists(pool.claim_path(job)))
            self.assertIsNone(pool.acquire(job,'other'))

    def test_wrong_case_count_never_publishes_completion(self):
        with tempfile.TemporaryDirectory() as authority, tempfile.TemporaryDirectory() as worker:
            pool=ShardPool(authority,LocalSFTP());job=self.job(worker);owner=pool.acquire(job,'aux')
            job[-1].mkdir(parents=True)
            (job[-1]/'cases.jsonl').write_text(json.dumps(dict(trajectory_id=1)))
            (job[-1]/'summary.json').write_text(json.dumps(dict(baseline='B0',mode='RO',checkpoint={'step':8})))
            with self.assertRaises(ValueError):pool.publish(job,owner)
            self.assertFalse(pool.exists(pool.output(job)/'summary.json'))


if __name__=='__main__':unittest.main()
