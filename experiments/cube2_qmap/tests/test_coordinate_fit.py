import unittest

import numpy as np
import torch

from grounded_llm.tasks.cube2.environment import Cube2
from grounded_llm.tasks.cube2.model import QMap
from experiments.cube2_qmap.src.coordinate_fit import initialize, subset_batch
from experiments.cube2_qmap.src.full_state_joint import batch_losses


class CoordinateFitChecks(unittest.TestCase):
    def test_cached_stars_match_original_loss_and_parameter_gradients(self):
        # Duplicate and reordered decisions must preserve labels and all candidates.
        rng = np.random.default_rng(17)
        physical = rng.integers(0,19,size=(5,7,2)); physical[:,1:,1] = physical[:,0,None,1]
        full = dict(pairs=physical.reshape(-1,2),labels=rng.random(35).astype(np.float32),
            truth=rng.random((5,6)).astype(np.float32))
        ids = np.array([4,1,4,0]); states,subset = subset_batch(full,ids)
        old = dict(pairs=physical[ids].reshape(-1,2),labels=full['labels'].reshape(5,7)[ids].ravel(),
            children=physical[ids,1:,0],goals=physical[ids,0,1],truth=full['truth'][ids])
        features = torch.randn(19,192); model = QMap(192,128,16,14,'linf')
        cfg = {'training':{'temperature':1.0}}
        a = sum(batch_losses(model,features,old,cfg))
        ga = torch.autograd.grad(a,tuple(model.parameters()))
        b = sum(batch_losses(model,features[states],subset,cfg))
        gb = torch.autograd.grad(b,tuple(model.parameters()))
        torch.testing.assert_close(a,b)
        for x,y in zip(ga,gb):
            torch.testing.assert_close(x,y,atol=1e-6,rtol=1e-5)

    def test_release_retains_axis_equivariance_and_other_raw_coordinates(self):
        env = Cube2(metric='QTM_URF_6')
        model = QMap(192,128,16,14,'linf',symmetry='cube_axis_permutation_average')
        bits = torch.as_tensor(env.features(np.array([1,824,12345])))
        before = model.network(bits).detach()
        initialize(model,'release126')
        after = model.network(bits).detach()
        torch.testing.assert_close(before[:,:126],after[:,:126])
        torch.testing.assert_close(before[:,127],after[:,127])
        self.assertEqual(float(after[:,126].abs().sum()),0)
        for axis in range(6):
            changed = torch.as_tensor(env.features(env.axis_symmetry(np.array([1,824,12345]),axis)))
            torch.testing.assert_close(model.encode(changed),
                model.encode(bits)[:,model.axis_output_action[axis]],atol=1e-6,rtol=1e-5)


if __name__ == '__main__':
    unittest.main()
