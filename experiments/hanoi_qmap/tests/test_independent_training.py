import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from experiments.hanoi_qmap.src.data import build
from experiments.hanoi_qmap.src.environment import Hanoi
from experiments.hanoi_qmap.src.model import QMap, MapReader
from experiments.hanoi_qmap.src.run import make_environment, train
from experiments.puzzle_qmap.src.data import build as historical_build
from experiments.puzzle_qmap.src.run import train as historical_train


class IndependentHanoiTrainingTest(unittest.TestCase):
    def test_split_preserves_data_update_and_checkpoint_readout(self):
        env = Hanoi(3)
        config = json.loads((Path(__file__).parents[1] / 'configs/study.json').read_text())
        config = copy.deepcopy(config)
        config['model'] = dict(dimension=8, hidden=16)
        config['training'].update(steps=1, batch_size=16, decision_batch_size=8,
                                  eval_every=1, cpu_threads=1)
        config['evaluation'].update(decision_count=16, rollout_count=16)
        with tempfile.TemporaryDirectory() as temp:
            old, new = Path(temp) / 'historical', Path(temp) / 'hanoi'
            old.mkdir(); new.mkdir()
            prior, prior_summary = historical_build(env, config, old)
            data, summary = build(env, config, new)
            self.assertEqual(prior_summary, summary)
            self.assertEqual(prior.keys(), data.keys())
            for key in data:
                np.testing.assert_array_equal(prior[key], data[key])
            with contextlib.redirect_stdout(io.StringIO()):
                before = historical_train(env, config, prior, old, 'cpu', 'test')
                after = train(env, config, data, new, 'cpu', 'test')
            self.assertEqual(before['final'], {k: v for k, v in after['final'].items() if k != 'standard_towers'})
            self.assertEqual(after['final']['standard_towers']['oracle']['solved'], 6)
            a = torch.load(old / 'best.pt', weights_only=False)['model']
            b = torch.load(new / 'best.pt', weights_only=False)['model']
            for key in a:
                torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)
            frozen = MapReader(env, old / 'best.pt')
            model = QMap(env.feature_dim, **config['model'], cap=env.diameter)
            model.load_state_dict(a)
            torch.testing.assert_close(frozen.encode(np.arange(env.size)),
                model.encode(torch.as_tensor(env.features(np.arange(env.size)))), rtol=0, atol=0)

    def test_hanoi_entry_rejects_cube_task(self):
        with self.assertRaises(ValueError):
            make_environment('cube2_qtm')
        with self.assertRaises(ValueError):
            QMap(9, 8, 16, 7, symmetry='cube_axis_permutation_average')


if __name__ == '__main__':
    unittest.main()
