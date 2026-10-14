import json
from pathlib import Path
import tempfile
import unittest

from experiments.map_sft_baselines.src.evaluation_queue import released_training_gpus


class ReleasedTrainingGPUsTest(unittest.TestCase):
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
