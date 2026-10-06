"""Resume the orchestration without rebuilding data or repeating finished shards."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from experiments.flamingo_map_reader.src import retrain_blocks


class RetrainResumeTest(unittest.TestCase):
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
