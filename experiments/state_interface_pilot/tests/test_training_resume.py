"""Optimizer/shuffle restoration and conservative convergence stopping."""
import copy
from pathlib import Path
import tempfile
import unittest
import torch

from grounded_llm.training import train, joint_loss_plateau


class TrainingTests(unittest.TestCase):
    def test_validation_plateau_alone_does_not_stop_training(self):
        spec = dict(minimum_epochs=5, checks=3, relative_min_improvement=0.005)
        history = [dict(epoch=i + 1, loss=1 / (i + 1), validation_loss=1.0,
                        validation_correct=0, validation_total=10) for i in range(5)]
        self.assertFalse(joint_loss_plateau(history, spec))
        for row in history:
            row['loss'] = 0.4
        self.assertTrue(joint_loss_plateau(history, spec))
        history[-1]['validation_correct'] = 1
        self.assertFalse(joint_loss_plateau(history, spec))

    def test_resumed_optimizer_and_shuffle_match_uninterrupted_training(self):
        spec = dict(initial_microbatch_size=2, report_batch_size=3, epochs=4,
                    shuffle_seed=18, selection='validation', validation_interval=1,
                    learning_rate=0.01, betas=[0.9, 0.999], epsilon=1e-8,
                    weight_decay=0.0, gradient_clip_norm=1.0, save_training_state=True)

        class Interface:
            def __init__(self, epochs):
                self.config = {'training': dict(spec, epochs=epochs)}
                self.model = torch.nn.Linear(1, 1).requires_grad_(False)
            def losses(self, items, adapter):
                x = torch.tensor(items, dtype=torch.float32).reshape(-1, 1)
                return (adapter(x).flatten() - x.flatten() * 2).square()

        torch.manual_seed(17)
        initial = torch.nn.Linear(1, 1)
        full, partial, resumed = [copy.deepcopy(initial) for _ in range(3)]
        data = [0.1, 0.3, 0.9, 1.2, 1.7, 2.0, 2.5]
        validate = lambda *_: (0, len(data))
        with tempfile.TemporaryDirectory() as root:
            paths = [Path(root) / name for name in ('full', 'partial', 'resumed')]
            for path in paths:
                path.mkdir()
            train(Interface(4), full, data, data, paths[0], validate)
            train(Interface(2), partial, data, data, paths[1], validate)
            result = train(Interface(4), resumed, data, data, paths[2], validate,
                           resume=paths[1] / 'latest.pt')
            self.assertEqual(result['resumed_from_epoch'], 2)
            self.assertEqual(result['sample_presentations'], 4 * len(data))
            for expected, actual in zip(full.parameters(), resumed.parameters()):
                torch.testing.assert_close(expected, actual, rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
