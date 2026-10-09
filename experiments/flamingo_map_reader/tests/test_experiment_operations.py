"""Exercise launch contract reuse and incomplete cross-machine publication."""

import json
from pathlib import Path
import tempfile
import unittest
import sys
from unittest.mock import patch

from experiments.flamingo_map_reader.src.artifact_bridge import import_shard, missing_owned_shards
from experiments.flamingo_map_reader.src.experiment_runner import build_plan, existing_worker, supervise

CONFIGS = Path(__file__).resolve().parents[1]/'configs'
CONTRACT = CONFIGS/'blocks_kv_two_node_continuation.json'


class LaunchPlanTest(unittest.TestCase):
    def profile(self, root):
        return dict(workspace_root=str(root/'workspace'), code_root=str(root/'source'),
                    python='/env/bin/python', model_path='/models/base', q_checkpoint='/maps/q.pt')

    def test_four_gpu_continuation_uses_original_source_and_extension(self):
        plan = build_plan(CONTRACT, self.profile(Path('/runtime')), 'training')
        command = plan['command']
        self.assertIn('--nproc_per_node=4', command)
        self.assertEqual(command[command.index('--batch-size')+1], '1')
        self.assertTrue(command[command.index('--resume')+1].replace('\\', '/').endswith('training/models/checkpoint-125000'))
        self.assertTrue(command[command.index('--epoch-extension')+1].endswith('blocks_kv_twenty_epoch_extension.json'))
        self.assertNotIn('--resume-topology', command)
        self.assertEqual(plan['environment']['CUDA_VISIBLE_DEVICES'], '0,1,2,3')

    def test_queues_keep_disjoint_shards_and_auxiliary_relocation(self):
        profile = self.profile(Path('/runtime'))
        profile['source_path_map'] = '/runtime/relocation.json'
        for role, expected in [('evaluation', '0,1'), ('auxiliary_evaluation', '2')]:
            command = build_plan(CONTRACT, profile, role)['command']
            self.assertEqual(command[command.index('--shard-modulo')+1], '3')
            self.assertEqual(command[command.index('--shard-remainders')+1], expected)
            self.assertIn('--cache-map-kv', command)
            self.assertIn('--terminal-tasks', command)
            self.assertIn('/runtime/relocation.json', command)

    def test_runtime_resume_checkpoint_keeps_budget_and_output(self):
        profile = self.profile(Path('/runtime'))
        checkpoint = '/runtime/workspace/runs/blocks_kv_restart_36c213c/training_extension_20epoch/models/checkpoint-145000'
        profile['resume_checkpoint'] = checkpoint
        plan = build_plan(CONTRACT, profile, 'training')
        command = plan['command']
        self.assertEqual(command[command.index('--resume')+1], checkpoint)
        self.assertTrue(command[command.index('--out')+1].replace('\\', '/').endswith('training_extension_20epoch'))
        self.assertIn('--epoch-extension', command)

    def test_wrong_global_batch_is_rejected_before_launch(self):
        with tempfile.TemporaryDirectory() as directory:
            contract = json.loads(CONTRACT.read_text())
            contract['training']['per_device_batch_size'] = 2
            path = Path(directory)/'contract.json'
            path.write_text(json.dumps(contract))
            with self.assertRaisesRegex(ValueError, 'global batch'):
                build_plan(path, self.profile(Path(directory)), 'training')

    def test_existing_legacy_worker_is_detected_without_matching_other_runs(self):
        plan = build_plan(CONTRACT, self.profile(Path('/runtime')), 'training')
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'12345/cmdline'
            path.parent.mkdir()
            path.write_bytes(('\0'.join(plan['command'])+'\0').encode())
            with patch('pathlib.Path.glob', return_value=[path]):
                self.assertEqual(existing_worker(plan['command']), 12345)
                command = list(plan['command'])
                command[command.index('--out')+1] = '/other/run'
                self.assertIsNone(existing_worker(command))

    def test_supervisor_records_failed_child_without_claiming_completion(self):
        with tempfile.TemporaryDirectory() as directory:
            plan = dict(command=[sys.executable, '-c', 'raise SystemExit(3)'], cwd=directory,
                        environment={}, status_path=str(Path(directory)/'status.json'))
            with patch('experiments.flamingo_map_reader.src.experiment_runner.existing_worker', return_value=None):
                supervise(plan)
            state = json.loads(Path(plan['status_path']).read_text())
            self.assertEqual(state['status'], 'failed')
            self.assertEqual(state['returncode'], 3)
            self.assertIsInstance(state['worker_pid'], int)


class LocalSFTP:
    def __init__(self, fail=None):
        self.fail = fail
        self.copied = []

    def stat(self, path):
        return Path(path).stat()

    def get(self, source, destination):
        if Path(source).name == self.fail:
            Path(destination).write_text('interrupted')
            raise OSError('Transfer interrupted')
        self.copied.append(Path(source).name)
        Path(destination).write_bytes(Path(source).read_bytes())


class TransferPublicationTest(unittest.TestCase):
    def test_partial_cases_never_publish_summary_and_completed_shards_are_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root/'source'
            source.mkdir()
            (source/'cases.jsonl').write_text('case\n')
            (source/'summary.json').write_text('{}')
            output = root/'output'
            with self.assertRaises(OSError):
                import_shard(LocalSFTP(fail='cases.jsonl'), str(source), output)
            self.assertFalse((output/'summary.json').exists())
            files = LocalSFTP()
            self.assertTrue(import_shard(files, str(source), output))
            self.assertEqual(files.copied, ['cases.jsonl', 'summary.json'])
            (source/'cases.jsonl').write_text('changed')
            self.assertFalse(import_shard(files, str(source), output))
            self.assertEqual((output/'cases.jsonl').read_text(), 'case\n')

    def test_auxiliary_completion_does_not_require_other_nodes_shards(self):
        contract = json.loads(CONTRACT.read_text())
        training = json.loads((CONFIGS/contract['source_training_contract']).read_text())
        evaluation = json.loads((CONFIGS/contract['evaluation_contract']).read_text())
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            self.assertTrue(missing_owned_shards(run, 125000, contract, training, evaluation))
            # Auxiliary owns ordinary shards 2 and 5; each terminal task has only two shards.
            for mode in ('reference', 'rollout'):
                for start in (128, 320):
                    shard = run/f'evaluation_half_epoch/step-125000/test_{mode}/{start:05d}_{start+64:05d}'
                    shard.mkdir(parents=True)
                    (shard/'summary.json').write_text('{}')
            self.assertFalse(missing_owned_shards(run, 125000, contract, training, evaluation))


if __name__ == '__main__':
    unittest.main()
