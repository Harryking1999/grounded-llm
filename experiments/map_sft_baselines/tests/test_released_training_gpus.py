import json
from pathlib import Path
import tempfile
import unittest

from experiments.map_sft_baselines.src.evaluation_queue import (
    released_training_gpus, training_completed, evaluation_completed, worker_gpu, free_worker_slot)


class ReleasedTrainingGPUsTest(unittest.TestCase):
    def test_evaluation_handoff_requires_complete_exact_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'config.json').write_text(json.dumps(dict(
                baselines={'B2': dict(real_map=True)},
                evaluation=dict(tasks=3500, modes=['RO'], probe_tasks=0))))
            path = root / 'evaluation/B2/checkpoint-150000/summary.json'
            path.parent.mkdir(parents=True)
            self.assertFalse(evaluation_completed(path))
            for step, tasks, expected in [(150000, 3499, False), (125000, 3500, False),
                                           (150000, 3500, True)]:
                path.write_text(json.dumps(dict(step=step, metrics={'RO': dict(tasks=tasks)})))
                self.assertEqual(evaluation_completed(path), expected)

    def test_concurrent_slots_keep_physical_gpu_and_enforce_capacity(self):
        active = {'1': object(), '2': object()}
        self.assertIsNone(free_worker_slot(active, 1, 1))
        slot = free_worker_slot(active, 1, 2)
        self.assertEqual(slot, '1:1')
        self.assertEqual(worker_gpu(slot), 1)
        active[slot] = object()
        self.assertIsNone(free_worker_slot(active, 1, 2))
        self.assertEqual(free_worker_slot(active, 3, 2), '3')
        del active['1']
        self.assertEqual(free_worker_slot(active, 1, 2), '1')
        self.assertEqual(worker_gpu('2'), 2)

    def test_final_export_is_required_for_handoff(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'finished.json'
            self.assertFalse(training_completed(path))
            path.write_text(json.dumps(dict(status='stopped', step=123)))
            with self.assertRaisesRegex(RuntimeError, 'did not complete'):
                training_completed(path)
            path.write_text(json.dumps(dict(status='completed', step=50000)))
            self.assertTrue(training_completed(path))

    def test_training_priority_and_released_lanes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'training_queue.json'
            gpus = list(range(8))
            self.assertEqual(released_training_gpus(path, gpus), [])
            state = dict(status='running', pending=['B4'], active={'3': dict(gpus=[6, 7])})
            path.write_text(json.dumps(state))
            self.assertEqual(released_training_gpus(path, gpus), [])
            state.update(pending=[], active={'0': dict(gpus=[0, 1]), '1': dict(gpus=[2, 3]), '2': dict(gpus=[4, 5])})
            path.write_text(json.dumps(state))
            self.assertEqual(released_training_gpus(path, gpus), [6, 7])
            state.update(status='finished', active={})
            path.write_text(json.dumps(state))
            self.assertEqual(released_training_gpus(path, gpus), gpus)


if __name__ == '__main__':
    unittest.main()
