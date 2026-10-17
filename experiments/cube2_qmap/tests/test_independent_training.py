import unittest
import contextlib
import io
import tempfile
from pathlib import Path
import numpy as np
import torch
from experiments.cube2_qmap.src.model import QMap
from experiments.cube2_qmap.src.relation_data import state_split, pair_keys
from experiments.cube2_qmap.src.run import make_environment
from experiments.cube2_qmap.src.environment import Cube2
from experiments.cube2_qmap.src.relation_data import build
from experiments.puzzle_qmap.src.data import build as historical_build
from experiments.puzzle_qmap.src.data import state_split as historical_split, pair_keys as historical_keys


class IndependentCubeTrainingTest(unittest.TestCase):
    def test_cube_entry_rejects_hanoi_task(self):
        with self.assertRaises(ValueError):
            make_environment('hanoi3')

    def test_split_identity_and_model_state_keys(self):
        rows = np.array([[0, 5], [5, 0], [300, 120]])
        np.testing.assert_array_equal(state_split(rows), historical_split(rows))
        np.testing.assert_array_equal(pair_keys(rows, 1000), historical_keys(rows, 1000))
        model = QMap(192, 128, 256, 14, metric='linf', symmetry='cube_axis_permutation_average')
        self.assertTrue(all(k.startswith('network.') for k in model.state_dict()))
        self.assertEqual(model.encode(torch.zeros(1, 192)).shape, (1, 128))

    def test_cube_relation_builder_preserves_frozen_split(self):
        env = Cube2(metric='QTM_URF_6')
        config = dict(seed=15, data=dict(pairs_train=500, pairs_validation=40, pairs_test=40,
                      decisions_train=500, decisions_validation=40, decisions_test=40),
                      augmentation=dict(kind=env.symmetry_kind, enabled=True, copies=1, max_attempts=128))
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with contextlib.redirect_stdout(io.StringIO()):
                env.build_oracle(root / 'oracle.npy')
            old, new = root / 'old', root / 'new'
            old.mkdir(); new.mkdir()
            a, sa = historical_build(env, config, old)
            b, sb = build(env, config, new)
            self.assertEqual(sa, sb)
            self.assertEqual(a.keys(), b.keys())
            for key in a:
                np.testing.assert_array_equal(a[key], b[key])


if __name__ == '__main__':
    unittest.main()
