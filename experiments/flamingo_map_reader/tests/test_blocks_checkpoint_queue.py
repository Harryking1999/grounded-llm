"""Check selection of published half-epoch weights without launching workers."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments.flamingo_map_reader.src.blocks_checkpoint_queue import (
    checkpoints, gpu_modes, validate_resume_contract, evaluation_modes,
    owns_shard, queue_file, shard_remainders, validate_auxiliary_partition,
    main, next_assignment, claim_shard, ExistingProcess, owns_job,
)


class CheckpointSelectionTest(unittest.TestCase):
    def test_independent_no_solution_tasks_are_only_scheduled_at_final(self):
        manifest = dict(records=[dict(split='test')]*510 + [dict(split='test_no_solution')]*100)
        ordinary = evaluation_modes(dict(path=Path('checkpoint-6875')), manifest)
        final = evaluation_modes(dict(path=Path('final')), manifest)
        self.assertEqual(ordinary, [('test', 'rollout', 510), ('test', 'reference', 510)])
        self.assertEqual(final, [*ordinary, ('test_no_solution', 'reference', 100)])

    def publish(self, root, name, step, epoch, adapter=True):
        directory = root / name
        directory.mkdir()
        (directory / 'evaluation_ready.json').write_text(json.dumps(dict(step=step, epoch=epoch)))
        if adapter:
            (directory / 'adapter.pt').touch()
        return directory

    def test_half_epoch_selection_and_latest_first(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.publish(root, 'checkpoint-0', 0, 0)
            self.publish(root, 'checkpoint-1375', 1375, 0.1)
            self.publish(root, 'checkpoint-6875', 6875, 0.5)
            self.publish(root, 'checkpoint-13750', 13750, 1.0)
            self.publish(root, 'checkpoint-20625', 20625, 1.5, adapter=False)
            self.assertEqual([c['step'] for c in checkpoints(root)], [13750, 6875])

    def test_optimizer_boundary_overshoot_is_allowed_once(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.publish(root, 'checkpoint-430', 430, 0.5)
            self.publish(root, 'checkpoint-431', 431, 0.5 + 1/860)
            self.publish(root, 'checkpoint-432', 432, 0.5 + 2/860)
            self.assertEqual([c['step'] for c in checkpoints(root)], [431, 430])

    def test_final_deduplicates_same_numbered_weights(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            final = self.publish(root, 'final', 137500, 10.0)
            self.publish(root, 'checkpoint-137500', 137500, 10.0)
            selected = checkpoints(root)
            self.assertEqual(len(selected), 1)
            self.assertEqual(selected[0]['path'], final)

    def test_continuation_keeps_source_final_and_prioritizes_newer_weights(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, continuation = root/'source', root/'continuation'
            source.mkdir()
            old_final = self.publish(source, 'final', 125000, 10.0)
            self.publish(source, 'checkpoint-125000', 125000, 10.0)
            self.assertEqual(checkpoints(source, continuation)[0]['path'], old_final)
            continuation.mkdir()
            newer = self.publish(continuation, 'checkpoint-131250', 131250, 10.5)
            selected = checkpoints(source, continuation)
            self.assertEqual([c['path'] for c in selected], [newer, old_final])
            self.assertTrue((old_final/'adapter.pt').is_file())


class QueueMigrationTest(unittest.TestCase):
    def test_override_transfers_early_backlog_without_changing_recent_or_other_shards(self):
        rules=[dict(steps=[6250,62500], indices=[5], queues=['main','extra'])]
        for step, index, platform, auxiliary in [(6250,5,True,False), (62500,5,True,False),
                (68750,5,False,True), (6250,6,False,True), (6250,0,True,False)]:
            self.assertEqual(owns_job(step,index*64,64,9,[0,1,2,3,4],'main',rules),platform)
            self.assertEqual(owns_job(step,index*64,64,9,[5,6,7,8],'aux',rules),auxiliary)

    def test_shared_claim_excludes_second_worker_and_preserves_completed_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)/'step-1/test_rollout/00000_00064'
            lock = claim_shard(output)
            self.assertIsNotNone(lock)
            self.assertIsNone(claim_shard(output))
            lock.release()
            second = claim_shard(output)
            self.assertIsNotNone(second)
            second.release()
            output.mkdir()
            (output/'summary.json').write_text('{}')
            self.assertIsNone(claim_shard(output))

    def test_overlapping_pool_requires_explicit_shared_locking_on_both_queues(self):
        primary = dict(run='same', shard_modulo=9, shard_remainders=[0,1,2,3,4],
                       shared_shard_pool=True, flexible_modes=True)
        validate_auxiliary_partition(primary, dict(primary, queue_name='extra'))
        with self.assertRaises(ValueError):
            validate_auxiliary_partition(primary, dict(primary, shared_shard_pool=False))

    def test_adopted_worker_identity_and_completion_are_verified(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            argv = ['python', '-m', 'experiments.flamingo_map_reader.src.trajectory_eval',
                    '--out', str(output)]
            identity = ('\0'.join(argv)+'\0').encode()
            with patch('pathlib.Path.read_bytes', return_value=identity):
                worker = ExistingProcess(123, output)
                self.assertIsNone(worker.poll())
            with patch('pathlib.Path.read_bytes', side_effect=FileNotFoundError):
                self.assertEqual(worker.poll(), 1)
                (output/'summary.json').write_text('{}')
                self.assertEqual(worker.poll(), 0)
            with patch('pathlib.Path.read_bytes', return_value=b'other\0process\0'):
                with self.assertRaises(ValueError):
                    ExistingProcess(123, output)

    def test_single_gpu_keeps_newest_first_across_both_modes(self):
        assignments = gpu_modes(['4'])
        self.assertEqual(assignments, {'4': 'both'})
        reference = (None, 'test', 'reference')
        rollout = (None, 'test', 'rollout')
        jobs = [reference, rollout]
        self.assertEqual(next_assignment(jobs, assignments['4'], True), 0)
        jobs.pop(0)
        self.assertEqual(next_assignment(jobs, assignments['4'], True), 0)
        self.assertIsNone(next_assignment([], assignments['4'], True))

    def test_source_final_does_not_complete_queue_before_continuation_final(self):
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary)
            (run/'data').mkdir()
            (run/'data/manifest.json').write_text(json.dumps(dict(config=dict(task='blocks'),
                records=[dict(split='test')]*510 + [dict(split='test_no_solution')]*100)))
            final=run/'training/models/final'
            final.mkdir(parents=True)
            (final/'adapter.pt').touch()
            (final/'evaluation_ready.json').write_text(json.dumps(dict(step=125000,epoch=10.0)))
            for split_mode,tasks in [('test_reference',510),('test_rollout',510),('test_no_solution_reference',100)]:
                mode=run/'evaluation_half_epoch/step-125000'/split_mode
                mode.mkdir(parents=True)
                (mode/'summary.json').write_text('{}')
                for start in range(0,tasks,64):
                    shard=mode/f'{start:05d}_{min(start+64,tasks):05d}'
                    shard.mkdir()
                    (shard/'summary.json').write_text('{}')
            argv=['queue','--run',str(run),'--model-path','model','--gpus','0,1',
                  '--continuation-models',str(run/'training_extension_20epoch/models')]
            with patch('sys.argv',argv), patch(
                    'experiments.flamingo_map_reader.src.blocks_checkpoint_queue.time.sleep',
                    side_effect=InterruptedError), self.assertRaises(InterruptedError):
                main()
            status=json.loads((run/'evaluation_half_epoch/status.json').read_text())
            self.assertEqual(status['status'],'running')
            self.assertEqual(status['pending_shards'],0)

    def test_main_waits_for_auxiliary_final_shards(self):
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary)
            data = run / 'data'
            data.mkdir()
            (data / 'manifest.json').write_text(json.dumps(dict(config=dict(task='blocks'),
                records=[dict(split='test')]*510 + [dict(split='test_no_solution')]*100)))
            final = run / 'training/models/final'
            final.mkdir(parents=True)
            (final / 'adapter.pt').touch()
            (final / 'evaluation_ready.json').write_text(json.dumps(dict(step=137500, epoch=10.0)))
            root = run / 'evaluation_half_epoch/step-137500'
            for mode in ('rollout', 'reference'):
                for start in range(0, 510, 64):
                    if not owns_shard(start, 64, 3, (0, 1)):
                        continue
                    out = root / f'test_{mode}/{start:05d}_{min(start+64, 510):05d}'
                    out.mkdir(parents=True)
                    (out / 'summary.json').write_text('{}')
            negative = root / 'test_no_solution_reference'
            negative.mkdir()
            (negative / 'summary.json').write_text('{}')
            for start, stop in ((0, 64), (64, 100)):
                out = negative / f'{start:05d}_{stop:05d}'
                out.mkdir()
                (out / 'summary.json').write_text('{}')
            # All main-owned work is done. The four auxiliary test shards are missing.
            argv = ['queue', '--run', str(run), '--model-path', 'model',
                    '--shard-modulo', '3', '--shard-remainders', '0,1']
            with patch('sys.argv', argv), patch(
                    'experiments.flamingo_map_reader.src.blocks_checkpoint_queue.time.sleep',
                    side_effect=InterruptedError), self.assertRaises(InterruptedError):
                main()
            status = json.loads((run / 'evaluation_half_epoch/status.json').read_text())
            self.assertEqual(status['status'], 'running')
            self.assertEqual(status['running'], {})
            self.assertEqual(status['pending_shards'], 0)

    def test_two_node_partitions_cover_main_and_final_no_solution_once(self):
        primary = shard_remainders(3, '0,1')
        auxiliary = shard_remainders(3, '2')
        for tasks in (510, 100):
            covered = []
            for start in range(0, tasks, 64):
                owners = [owns_shard(start, 64, 3, values) for values in (primary, auxiliary)]
                self.assertEqual(sum(owners), 1)
                covered.extend(range(start, min(start + 64, tasks)))
            self.assertEqual(covered, list(range(tasks)))

    def test_auxiliary_queue_rejects_overlapping_or_different_partition(self):
        primary = dict(run='same', tasks=510, shard_modulo=3, shard_remainders=[0, 1])
        auxiliary = dict(primary, shard_remainders=[2])
        validate_auxiliary_partition(primary, auxiliary)
        for changed in (dict(auxiliary, shard_remainders=[1, 2]),
                        dict(auxiliary, shard_modulo=4), dict(auxiliary, tasks=100)):
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                validate_auxiliary_partition(primary, changed)

    def test_two_gpu_auxiliary_has_both_modes_and_separate_runtime_files(self):
        self.assertEqual(gpu_modes(['2', '3']), {'2': 'rollout', '3': 'reference'})
        root = Path('run')
        for name in ('status.json', 'contract.json', '.queue.lock'):
            self.assertEqual(queue_file(root, name, 'main'), root / name)
            self.assertNotEqual(queue_file(root, name, 'main'), queue_file(root, name, 'aux'))
        with self.assertRaises(ValueError):
            queue_file(root, 'status.json', '../other')

    def test_invalid_shard_remainders_are_rejected(self):
        for modulo, values in ((0, '0'), (3, '0,0'), (3, '-1'), (3, '3')):
            with self.subTest(modulo=modulo, values=values), self.assertRaises(ValueError):
                shard_remainders(modulo, values)

    def test_projection_cache_changes_execution_but_keeps_generation_contract(self):
        previous = dict(run='same-run', tasks=510, modes=['rollout', 'reference'],
                        variants=1, cache_map_kv=False)
        current = dict(previous, cache_map_kv=True)
        validate_resume_contract(previous, current)
        with self.assertRaises(ValueError):
            validate_resume_contract(previous, dict(current, variants=2))

    def test_three_gpu_resume_preserves_experiment(self):
        previous = dict(run='same-run', tasks=510, modes=['rollout', 'reference'],
                        gpus=['0', '1', '2', '3'], gpu_modes=gpu_modes(['0', '1', '2', '3']))
        current = dict(previous, gpus=['0', '1', '2'], gpu_modes=gpu_modes(['0', '1', '2']))
        validate_resume_contract(previous, current)
        self.assertEqual(list(current['gpu_modes'].values()).count('rollout'), 2)
        self.assertEqual(list(current['gpu_modes'].values()).count('reference'), 1)
        for changed in (dict(current, run='other-run'), dict(current, tasks=100),
                        dict(current, modes=['reference'])):
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                validate_resume_contract(previous, changed)

    def test_duplicate_gpu_cannot_own_two_workers(self):
        with self.assertRaises(ValueError):
            gpu_modes(['0', '1', '2', '2'])


if __name__ == '__main__':
    unittest.main()
