from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from experiments.blocks_distance_map.src.model import BoardEncoder, distance as blocks_distance
from experiments.cube2_qmap.src.environment import Cube2
from experiments.puzzle_qmap.src.data import build, state_split
from experiments.puzzle_qmap.src.model import QMap, distance, soft_rank_loss


class PureQTest(unittest.TestCase):
    def test_encoder_reuses_blocks_and_has_no_other_parameters(self):
        torch.manual_seed(11)
        old = BoardEncoder(128, 256, 7, "directed_sum")
        torch.manual_seed(11)
        new = QMap(100, 128, 256, 7)
        features = torch.randn(8, 100)
        torch.testing.assert_close(old.encode(features), new.encode(features), rtol=0, atol=0)
        self.assertTrue(all(key.startswith("network.") for key in new.state_dict()))
        self.assertFalse(hasattr(new, "transition"))

    def test_distance_is_exact_bidirectional_average(self):
        a, b = torch.randn(10, 128), torch.randn(10, 128)
        expected = (blocks_distance(a, b, "directed_sum") + blocks_distance(b, a, "directed_sum")) / 2
        torch.testing.assert_close(distance(a, b), expected, rtol=0, atol=0)
        torch.testing.assert_close(distance(a, b), distance(b, a), rtol=0, atol=0)
        torch.testing.assert_close(distance(a, a), torch.zeros(10))

    def test_soft_ranking_matches_blocks_formula_and_orders_nonoptimal_choices(self):
        scores = torch.tensor([[.1, 1.2, 2.9]], requires_grad=True)
        truth = torch.tensor([[0, 1, 3]])
        valid = torch.ones_like(truth, dtype=torch.bool)
        expected = -(torch.softmax(-truth.float(), dim=1) * torch.log_softmax(-scores, dim=1)).sum(1).mean()
        loss = soft_rank_loss(scores, truth, valid, 1.0)
        torch.testing.assert_close(loss, expected)
        # Both predictions keep the same optimal choice; only the other ordering changes.
        reversed_others = scores.detach()[:, [0, 2, 1]]
        self.assertLess(float(loss.detach()), float(soft_rank_loss(reversed_others, truth, valid, 1.0)))
        padded = torch.cat((scores, torch.tensor([[999.0]])), dim=1)
        masked_loss = soft_rank_loss(padded, torch.tensor([[0, 1, 3, 99]]), torch.tensor([[True, True, True, False]]), 1.0)
        torch.testing.assert_close(loss, masked_loss)
        masked_loss.backward()
        self.assertTrue(torch.isfinite(scores.grad).all())

    def test_cube_symmetry_labels_edges_and_augmented_split(self):
        env = Cube2(metric="QTM_URF_6")
        rng = np.random.default_rng(15)
        states, goals, transform = rng.integers(env.size, size=(3, 1000))
        changed = env.symmetry(states, transform)
        self.assertTrue(np.all(changed < env.size))
        np.testing.assert_array_equal(env.symmetry(env.successors(states), transform[:, None]),
                                      env.successors(changed))
        config = {
            "seed": 15,
            "data": {"pairs_train": 500, "pairs_validation": 40, "pairs_test": 40,
                     "decisions_train": 500, "decisions_validation": 40, "decisions_test": 40},
            "augmentation": {"kind": env.symmetry_kind, "enabled": True, "copies": 2,
                             "max_attempts": 128},
        }
        with tempfile.TemporaryDirectory() as tmp:
            env.build_oracle(Path(tmp) / "oracle.npy")
            np.testing.assert_array_equal(env.distance(states, goals),
                                          env.distance(changed, env.symmetry(goals, transform)))
            data, summary = build(env, config, Path(tmp))
        for name, split in (("train", 0), ("validation", 1), ("test", 2)):
            pairs = data[f"pairs_{name}"]
            np.testing.assert_array_equal(env.distance(pairs[:, 0], pairs[:, 1]), pairs[:, 2])
            self.assertTrue(np.all(state_split(pairs[:, :2]) == split))
        self.assertTrue(np.all(state_split(data["children_train"]) == 0))
        self.assertTrue(np.all(state_split(data["tasks_train"]) == 0))
        self.assertGreater(summary["augmentation"]["pairs"]["changed_rows"], 0)
        self.assertGreater(summary["augmentation"]["decisions"]["changed_rows"], 0)


if __name__ == "__main__":
    unittest.main()
