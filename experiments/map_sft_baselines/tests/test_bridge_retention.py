import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import Mock, patch
from experiments.map_sft_baselines.src import bridge
from experiments.map_sft_baselines.src.retention import dispatch_lock, prune_completed, has_capacity
from experiments.map_sft_baselines.src.shard_pool import ShardPool
from experiments.map_sft_baselines.tests.test_shard_pool import LocalSFTP


class Files(LocalSFTP):
    def listdir(self, path):
        return [p.name for p in Path(path).iterdir()]

    def listdir_attr(self, path):
        return [SimpleNamespace(filename=p.name, st_size=p.stat().st_size) for p in Path(path).iterdir()]

    def get(self, source, destination):
        Path(destination).write_bytes(Path(source).read_bytes())


class BridgeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.local = self.root / 'source'
        self.remote = self.root / 'target'
        self.remote.mkdir()
        self.files = Files()
        self.config = json.loads((Path(__file__).parents[1] / 'configs/study.json').read_text())
        self.policy = json.loads((Path(__file__).parents[1] / 'configs/bridge_retention.json').read_text())
        self.profile = dict(run=str(self.local), run_path=self.local, target_run=str(self.remote),
                            baselines=['B0'], early_stop=self.config['early_stop'])
        self.meta = dict(baseline='B0', step=250000, epoch=20.)
        self.row = dict(step=250000, epoch=20., metrics={m: dict(tasks=510, arrival=134) for m in ['RO', 'TF']})
        self.source = self.local / 'training/B0/checkpoint-250000'
        self.payload = self.remote / 'models/B0/checkpoint-250000'
        self.dump(self.source / 'evaluation_ready.json', self.meta)
        self.dump(self.source / 'state.json', self.meta)
        (self.source / 'weights.pt').write_bytes(b'weights')

    def dump(self, path, data):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data))

    def publish_ready(self):
        self.dump(self.payload / 'evaluation_ready.json', self.meta)
        (self.payload / 'weights.pt').write_bytes(b'weights')

    def prune(self):
        return prune_completed(self.files, self.profile, self.config,
                               {('B0', 'checkpoint-250000'): self.source / 'weights.pt'})

    def test_scores_return_before_upload_failure_and_keep_old_points(self):
        old = dict(self.row, step=6250, epoch=.5)
        self.dump(self.local / 'results/B0/curve.json', [old])
        self.dump(self.remote / 'results/B0/curve.json', [self.row])
        progress = {}
        with patch.object(bridge, 'execute', side_effect=lambda *a: self.payload.mkdir(parents=True)), \
                patch.object(bridge, 'publish', side_effect=OSError('Disk quota exceeded')):
            with self.assertRaises(OSError):
                bridge.cycle(self.files, None, self.profile, self.config, self.policy, progress)
        self.assertEqual(json.loads((self.local / 'results/B0/curve.json').read_text()), [old, self.row])
        self.assertEqual(progress['phase'], 'weights')
        self.assertIn('scores_synced_at', progress)

    def test_delete_only_after_complete_scores_mirrored_and_no_claim(self):
        self.publish_ready()
        summary = self.remote / 'evaluation/B0/checkpoint-250000/summary.json'
        self.dump(summary, self.row)
        self.assertEqual(self.prune(), [])  # Scores not yet mirrored.
        self.dump(self.local / 'results/B0/curve.json', [self.row])
        claim = summary.parent / 'RO_0000_0032.worker.claim'
        claim.mkdir()
        self.assertEqual(self.prune(), [])
        claim.rmdir()
        self.assertEqual(len(self.prune()), 1)
        self.assertFalse((self.payload / 'weights.pt').exists())
        self.assertTrue((self.payload / 'evaluation_ready.json').exists())
        self.assertTrue(summary.exists())
        self.assertEqual(ShardPool(self.remote).checkpoints(), [])
        self.assertEqual(ShardPool(str(self.remote), self.files).checkpoints(), [])
        # Keeping the ready marker also prevents the bridge republishing it.
        with patch.object(bridge, 'publish') as publish:
            bridge.cycle(self.files, None, self.profile, self.config, self.policy, {})
            publish.assert_not_called()

    def test_incomplete_metrics_or_missing_archive_never_delete(self):
        self.publish_ready()
        row = dict(self.row, metrics={'RO': dict(tasks=510, arrival=134)})
        self.dump(self.remote / 'evaluation/B0/checkpoint-250000/summary.json', row)
        self.dump(self.local / 'results/B0/curve.json', [row])
        self.assertEqual(self.prune(), [])
        (self.source / 'weights.pt').unlink()
        self.assertEqual(self.prune(), [])
        self.assertTrue((self.payload / 'weights.pt').exists())

    def test_dispatch_lock_prevents_cleanup(self):
        self.publish_ready()
        self.dump(self.remote / 'evaluation/B0/checkpoint-250000/summary.json', self.row)
        self.dump(self.local / 'results/B0/curve.json', [self.row])
        (self.remote / 'evaluation_dispatch.lock').mkdir()
        self.assertEqual(self.prune(), [])
        self.assertTrue((self.payload / 'weights.pt').exists())

    def test_dispatch_lock_excludes_other_owner_and_releases_after_error(self):
        for files in (None, self.files):
            with self.subTest(local=files is None):
                pool = ShardPool(str(self.remote), files)
                with self.assertRaisesRegex(RuntimeError, 'interrupted'):
                    with dispatch_lock(pool) as acquired:
                        self.assertTrue(acquired)
                        with dispatch_lock(pool) as second:
                            self.assertFalse(second)
                        self.assertTrue(pool.exists(pool.root / 'evaluation_dispatch.lock'))
                        raise RuntimeError('interrupted')
                with dispatch_lock(pool) as acquired:
                    self.assertTrue(acquired)
                self.assertFalse(pool.exists(pool.root / 'evaluation_dispatch.lock'))

    def test_capacity_and_interrupted_upload_recovery(self):
        self.payload.mkdir(parents=True)
        (self.payload / 'weights.pt.part').write_bytes(b'partial')
        self.assertFalse(has_capacity(self.files, self.profile, 'B0', 7, self.policy))
        self.prune()
        self.assertTrue(has_capacity(self.files, self.profile, 'B0', 7, self.policy))
        self.assertFalse(has_capacity(self.files, self.profile, 'B0', self.policy['maximum_weight_bytes'] + 1, self.policy))

    def test_partial_file_removed_on_write_failure(self):
        def fail(source, dest):
            Path(dest).write_bytes(b'partial')
            raise OSError('quota')
        self.payload.mkdir(parents=True)
        with patch.object(self.files, 'put', side_effect=fail):
            with self.assertRaises(OSError):
                bridge.publish(self.files, self.source / 'weights.pt', str(self.payload / 'weights.pt'))
        self.assertFalse((self.payload / 'weights.pt.part').exists())

    def test_source_retention_keeps_best_latest_and_unscored(self):
        markers=[]
        for step in [6250,12500,18750,25000]:
            path=self.local/'training/B0'/f'checkpoint-{step}'
            self.dump(path/'evaluation_ready.json',dict(baseline='B0',step=step,epoch=step/12500))
            (path/'weights.pt').write_bytes(b'weights')
            markers.append(path/'evaluation_ready.json')
        # 6250 is best, 25000 latest; 18750 has no returned complete scores.
        rows=[dict(self.row,step=6250),dict(self.row,step=12500,metrics={m:dict(tasks=510,arrival=1) for m in ['RO','TF']})]
        self.dump(self.local/'results/B0/curve.json',rows)
        deleted=bridge.prune_source(self.profile,self.config,markers)
        self.assertEqual([r['step'] for r in deleted],[12500])
        for step in [6250,18750,25000]:
            self.assertTrue((self.local/'training/B0'/f'checkpoint-{step}'/'weights.pt').exists())

    def test_failure_status_and_reconnect_after_space_returns(self):
        config = self.root / 'config.json'
        self.dump(config, self.config)
        profile = self.root / 'profile.json'
        self.dump(profile, {k: v for k, v in dict(self.profile, config=str(config), interval_seconds=1).items() if k != 'run_path'})
        states = []
        original = bridge.atomic_json
        def write(path, value):
            if path.name == 'bridge_status.json': states.append(dict(value))
            original(path, value)
        clients = Mock(), Mock()
        def healthy(files, target, p, config, policy, progress):
            progress.update(phase='idle', status='running')
        with patch.object(bridge, 'connect', side_effect=[OSError('quota'), clients]) as connect, \
                patch.object(bridge.paramiko.SFTPClient, 'from_transport', return_value=Mock()), \
                patch.object(bridge, 'cycle', side_effect=healthy), \
                patch.object(bridge, 'atomic_json', side_effect=write), patch.object(bridge.time, 'sleep'):
            bridge.run(profile, {}, cycles=2)
        self.assertEqual(connect.call_count, 2)
        failed = next(s for s in states if s['status'] == 'retrying')
        self.assertEqual(failed['phase'], 'connect')
        self.assertEqual(failed['error'], 'quota')
        self.assertIn('retry_at', failed)
        history = [json.loads(line) for line in (self.local / 'bridge_failures.jsonl').read_text().splitlines()]
        self.assertEqual(history[0]['error'], 'quota')
        self.assertEqual(states[-1]['status'], 'running')


if __name__ == '__main__':
    unittest.main()
