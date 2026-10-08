"""Resume the orchestration without rebuilding data or repeating finished shards."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from experiments.flamingo_map_reader.src import retrain_blocks


class RetrainResumeTest(unittest.TestCase):
    def test_fresh_kv_run_reuses_only_one_failure_copy_without_resuming_weights(self):
        configs = Path(__file__).resolve().parents[1]/'configs'
        original = json.loads((configs/'blocks_ffn_failure_batch4.json').read_text())
        config = json.loads((configs/'blocks_kv_restart.json').read_text())
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root/'source.json'
            source.write_text(json.dumps(dict(config=original, audit={}, q_checkpoint='map.pt', records=[
                dict(split='train', variants=6),
                dict(split='train', variants=1, start='1', goal='1'),
                dict(split='train', variants=6, no_solution=True)])))
            config_path = root/'config.json'
            config_path.write_text(json.dumps(config))
            commands = []
            class Process:
                def __init__(self, command, **kwargs):
                    commands.append(command)
                def wait(self):
                    return 0
            argv = ['retrain_blocks', '--config', str(config_path), '--prepared-manifest', str(source),
                    '--source-manifest', 'unused', '--model-path', 'model', '--out', str(root/'run'),
                    '--training-only']
            with mock.patch('sys.argv', argv), mock.patch.object(retrain_blocks.subprocess, 'Popen', Process), \
                    mock.patch.object(Path, 'symlink_to'):
                retrain_blocks.main()
            manifest = json.loads((root/'run/data/manifest.json').read_text())
            self.assertEqual([r['variants'] for r in manifest['records']], [6, 1, 1])
            self.assertEqual(manifest['audit']['training_numbered_trajectories'], 8)
            self.assertEqual(manifest['config']['map']['memory_mode'], 'address_key_state_value')
            self.assertTrue(all('--resume' not in c for c in commands))

    def test_two_gpu_continuation_reuses_newest_two_rank_checkpoint_and_leaves_evaluation_separate(self):
        configs = Path(__file__).resolve().parents[1]/'configs'
        config = json.loads((configs/'blocks_ffn_failure_batch4.json').read_text())
        topology = configs/'blocks_batch4_two_gpu_resume.json'
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)/'run'
            config_path = Path(directory)/'config.json'
            config_path.write_text(json.dumps(config))
            (run/'data').mkdir(parents=True)
            (run/'data/manifest.json').write_text(json.dumps(dict(config=config, audit={}, q_checkpoint='map.pt',
                records=[dict(split='test'), dict(split='test_no_solution')])))
            models = run/'training/models'
            models.mkdir(parents=True)
            (run/'training/config.json').write_text(json.dumps(dict(batch_size=2)))
            for number, ranks in ((86, 4), (172, 2)):
                checkpoint = models/f'checkpoint-{number}'
                checkpoint.mkdir()
                for filename in ('adapter.pt', 'optimizer.pt', 'scheduler.pt', 'trainer_state.json',
                                 'evaluation_ready.json', *(f'rng_state_{i}.pth' for i in range(ranks))):
                    (checkpoint/filename).touch()
            commands = []
            class Process:
                def __init__(self, command, **kwargs):
                    commands.append(command)
                def wait(self):
                    return 0
            argv = ['retrain_blocks', '--config', str(config_path), '--source-manifest', 'unused',
                '--model-path', 'model', '--out', str(run), '--resume', '--resume-topology', str(topology),
                '--training-only']
            with mock.patch('sys.argv', argv), mock.patch.object(retrain_blocks.subprocess, 'Popen', Process):
                retrain_blocks.main()
            self.assertEqual(len(commands), 1)
            command = commands[0]
            self.assertIn('--nproc_per_node=2', command)
            self.assertEqual(command[command.index('--batch-size')+1], '2')
            self.assertEqual(command[command.index('--resume')+1], str(models/'checkpoint-172'))
            self.assertEqual(command[command.index('--resume-topology')+1], str(topology))
            self.assertEqual(json.loads((run/'status.json').read_text())['status'], 'training_completed')

    def test_resume_passes_complete_checkpoint_and_retries_only_incomplete_shards(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = root/'run'
            config = dict(task='blocks', training=dict(world_size=4),
                          evaluation=dict(shard_tasks=64, numbering_variants=1))
            config_path = root/'config.json'
            config_path.write_text(json.dumps(config))
            (run/'data').mkdir(parents=True)
            manifest = dict(config=config, audit={}, q_checkpoint='map.pt',
                            records=[dict(split=split) for split in ('test', 'test_no_solution')])
            (run/'data/manifest.json').write_text(json.dumps(manifest))
            (run/'training').mkdir()
            (run/'training/config.json').write_text(json.dumps(dict(batch_size=16)))
            checkpoint = run/'training/models/checkpoint-86'
            checkpoint.mkdir(parents=True)
            for name in ('adapter.pt', 'optimizer.pt', 'scheduler.pt', 'trainer_state.json',
                         'evaluation_ready.json', *(f'rng_state_{i}.pth' for i in range(4))):
                (checkpoint/name).touch()
            incomplete = checkpoint.with_name('checkpoint-172')
            incomplete.mkdir()
            (incomplete/'evaluation_ready.json').touch()
            completed = run/'evaluation/test_reference_00000_00001'
            completed.mkdir(parents=True)
            (completed/'summary.json').write_text('{}')
            (completed/'cases.jsonl').write_text('{"preserved": true}\n')
            partial = run/'evaluation/test_rollout_00000_00001'
            partial.mkdir()
            (partial/'cases.jsonl').write_text('partial output')
            calls = []

            class Process:
                def __init__(self, command, **kwargs):
                    self.command = command
                    calls.append(command)

                def wait(self):
                    if any(arg.endswith('.train') for arg in self.command):
                        final = run/'training/models/final'
                        final.mkdir()
                        (final/'adapter.pt').touch()
                        (final/'evaluation_ready.json').write_text('{}')
                    else:
                        output = Path(self.command[self.command.index('--out')+1])
                        output.mkdir()
                        (output/'cases.jsonl').write_text('{"retried": true}\n')
                        (output/'summary.json').write_text('{}')
                    return 0

            argv = ['retrain_blocks', '--config', str(config_path), '--source-manifest', 'unused',
                    '--model-path', 'model', '--out', str(run), '--resume']
            with mock.patch('sys.argv', argv), mock.patch.object(retrain_blocks.subprocess, 'Popen', Process), \
                 mock.patch('experiments.flamingo_map_reader.src.trajectory_eval.aggregate',
                            side_effect=lambda cases, manifest: dict(count=len(cases))):
                retrain_blocks.main()
                self.assertEqual(len(calls), 3)
                training = calls[0]
                self.assertEqual(training[training.index('--resume')+1], str(checkpoint))
                self.assertEqual(training[training.index('--batch-size')+1], '16')
                self.assertEqual((completed/'cases.jsonl').read_text(), '{"preserved": true}\n')
                self.assertEqual((partial/'cases.jsonl').read_text(), '{"retried": true}\n')
                self.assertEqual(json.loads((run/'status.json').read_text())['status'], 'completed')
                calls.clear()
                # A later resume has nothing to train or evaluate again.
                retrain_blocks.main()
                self.assertFalse(calls)


if __name__ == '__main__':
    unittest.main()
