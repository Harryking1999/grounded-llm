import copy
import unittest

import numpy as np
import torch

from experiments.cube2_qmap.src.cycle_repair import ExitGuard, current_cycles, guarded_adam_step, split_cycle_losses
from experiments.cube2_qmap.src.cycle_supervision import cycle_pack
from experiments.cube2_qmap.tests.test_full_state_joint import Ring
from experiments.puzzle_qmap.src.model import QMap


class CycleRepairTests(unittest.TestCase):
    def test_mining_tracks_current_policy_and_excludes_protected_stars(self):
        pointer = np.arange(12)
        pointer[8], pointer[9] = 9, 8
        allowed = np.ones(12, dtype=bool)
        np.testing.assert_array_equal(current_cycles(pointer, allowed), [[8, 9]])
        pointer[8] = 7
        self.assertEqual(len(current_cycles(pointer, allowed)), 0)
        pointer[8] = 9
        allowed[9] = False
        self.assertEqual(len(current_cycles(pointer, allowed)), 0)

    def test_guard_damps_actual_adam_displacement(self):
        model = torch.nn.Linear(1, 1, bias=False)
        model.weight.data.zero_()
        optimizer = torch.optim.Adam(model.parameters(), lr=1.)
        (-model.weight.sum()).backward()
        scale = guarded_adam_step(model, optimizer, lambda: model.weight.item() <= .13,
                                  [1., .5, .25, .125, .0625])
        self.assertEqual(scale, .125)
        self.assertAlmostEqual(model.weight.item(), .125, places=6)
        self.assertEqual(int(optimizer.state[model.weight]["step"]), 1)

    def test_rejection_restores_existing_adam_moments_and_step(self):
        model = torch.nn.Linear(1, 1, bias=False)
        optimizer = torch.optim.Adam(model.parameters(), lr=.1)
        model.weight.sum().backward()
        optimizer.step()
        optimizer.zero_grad()
        before = model.weight.detach().clone()
        state = copy.deepcopy(optimizer.state[model.weight])
        (-model.weight.sum()).backward()
        scale = guarded_adam_step(model, optimizer, lambda: False, [1., .5])
        self.assertEqual(scale, 0.)
        torch.testing.assert_close(model.weight, before)
        for key, value in state.items():
            torch.testing.assert_close(optimizer.state[model.weight][key], value)

    def test_direction_correction_can_preserve_progress_when_scaling_rejects(self):
        model = torch.nn.Linear(2, 1, bias=False)
        model.weight.data.zero_()
        optimizer = torch.optim.Adam(model.parameters(), lr=.1)
        (-model.weight.sum()).backward()

        def projector(before, delta):
            # The first component must stay fixed; the second remains free.
            result = [d.clone() for d in delta]
            result[0][0, 0] = 0.
            return result

        scale = guarded_adam_step(model, optimizer, lambda: abs(model.weight[0, 0].item()) < 1e-8,
                                  [1., .5], projector)
        self.assertEqual(scale, 1.)
        self.assertAlmostEqual(model.weight[0, 0].item(), 0.)
        self.assertGreater(model.weight[0, 1].item(), .09)

    def test_exit_derivative_correction_passes_finite_guard(self):
        env = Ring()
        env.depth = env.distance(np.arange(env.size), 0)

        class ScalarCoordinates(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.q = torch.nn.Parameter(torch.tensor(env.depth, dtype=torch.float32)[:, None])

            def encode(self, features):
                return features @ self.q

            def distance(self, a, b):
                return (a - b).abs().amax(-1)

        model = ScalarCoordinates()
        guard = ExitGuard(model, env, [8], [1.9], "cpu")
        optimizer = torch.optim.Adam(model.parameters(), lr=1.)
        (-model.q[7] + .1 * model.q[20]).sum().backward()
        scale = guarded_adam_step(model, optimizer, guard.accepts, [1., .5],
                                  lambda before, delta: guard.project(before, delta, 4, .0001))
        self.assertEqual(scale, 1.)
        self.assertGreater(guard.last_projection_corrections, 0)
        self.assertTrue(guard.accepts())
        self.assertLess(model.q[20].item(), 19.1)

    def test_exit_and_relation_losses_have_independent_denominators(self):
        env = Ring()
        env.size = 40
        env.depth = env.distance(np.arange(env.size), 0)
        batch = cycle_pack(env, [[8, 9], [14, 15]], np.zeros(env.size, dtype=bool), np.random.default_rng(1))
        model = QMap(env.size, 8, 12, 20, "linf")
        features = torch.eye(env.size)[batch["states"]]
        config = {"training": {"temperature": 1.}}
        before = split_cycle_losses(model, features, batch, config)
        repeated = dict(batch)
        nonexit = ~(batch["states"][batch["pairs"]] == 0).any(-1)
        repeated["pairs"] = np.r_[batch["pairs"], batch["pairs"][nonexit]]
        repeated["labels"] = np.r_[batch["labels"], batch["labels"][nonexit]]
        after = split_cycle_losses(model, features, repeated, config)
        for family in before:
            for a, b in zip(before[family], after[family]):
                torch.testing.assert_close(a, b)
        sum(sum(values) for values in before.values()).backward()
        self.assertGreater(sum(float(p.grad.abs().sum()) for p in model.parameters()), 0)


if __name__ == "__main__":
    unittest.main()
