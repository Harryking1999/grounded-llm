import unittest

import numpy as np
import torch

from experiments.external_map_interface.src.blocks_dynamics import BlocksDynamics
from experiments.external_map_interface.src.blocks_diagnostics import (
    candidate_diagnostic, rollout_diagnostic)
from experiments.external_map_interface.src.train_blocks_dynamics import split_transitions


class TinyTask:
    placements = [(1, {"id": 0}), (2, {"id": 1})]

    def from_grid(self, grid):
        return int(grid)

    def apply(self, mask, action, initial):
        tile = self.placements[action["id"]][0]
        if tile & mask != tile:
            raise ValueError("illegal")
        return mask ^ tile


class BlocksDynamicsTest(unittest.TestCase):
    def model(self):
        model = BlocksDynamics(4, 2, {"state_dim": 1, "action_dim": 1,
                                      "hidden_dim": 1, "q_init_std": 1.})
        with torch.no_grad():
            model.q.weight.copy_(torch.tensor([[0.], [1.], [2.], [3.]]))
            model.action.weight.zero_()
            model.displacement[0].weight.copy_(torch.tensor([[1., 0.]]))
            model.displacement[0].bias.zero_()
            model.displacement[2].weight.fill_(1.)
            model.displacement[2].bias.zero_()
        return model

    def test_same_action_can_have_different_displacements(self):
        model = self.model()
        current = torch.tensor([[1.], [2.]])
        delta = model.predict(current, torch.tensor([0, 0])) - current
        self.assertEqual(delta.tolist(), [[1.], [2.]])
        self.assertEqual(model.action.num_embeddings, 2)
        # The score uses predicted Q (6), not the true next-state table row.
        score = model.candidate_distances(torch.tensor([3.]), torch.tensor([0]), torch.tensor([0.]))
        self.assertEqual(score.item(), 6.)

    def test_split_keeps_connected_training_states_and_actions(self):
        rows = np.array([[0, 0, 1], [0, 1, 2], [1, 1, 3], [2, 0, 3], [0, 2, 3]])
        train, heldout = split_transitions(rows, 4, .4, 7)
        self.assertTrue(len(heldout))
        self.assertFalse(set(train) & set(heldout))
        self.assertEqual(set(train) | set(heldout), set(range(len(rows))))
        self.assertEqual(set(rows[train, 1]), {0, 1, 2})
        reached = {0}
        for _ in range(4):
            for source, _, dest in rows[train]:
                if source in reached or dest in reached:
                    reached.update((source, dest))
        self.assertEqual(reached, {0, 1, 2, 3})

    def test_rollout_never_resets_prediction_to_table(self):
        model = self.model()
        rows = np.array([[3, 0, 2], [3, 1, 1], [2, 1, 0], [1, 0, 0]])
        stats = rollout_diagnostic(model, TinyTask(), {"grid": 3}, [0, 1, 2, 3], rows, 2, 0)
        # q0=3 -> prediction 6 -> prediction 12, actual final q=0.
        self.assertEqual(stats[2]["mse_to_table_q"], 144.)
        self.assertEqual(stats[2]["with_reference_q"], 2)

    def test_bad_legal_branch_is_diagnosed_without_labeling_missing_paths_dead(self):
        class BranchTask:
            placements = [(3, {}), (5, {}), (12, {})]

        # 15 -> 12 -> 0 is a witness; 15 -> 10 is a nonempty immediate dead end.
        states = [0, 10, 12, 15]
        rows = np.array([[3, 0, 2], [3, 1, 1], [2, 2, 0]])
        model = BlocksDynamics(4, 3, {"state_dim": 2, "action_dim": 2,
                                      "hidden_dim": 4, "q_init_std": 1.})
        result = candidate_diagnostic(model, BranchTask(), states, rows)
        self.assertEqual(result["sampled_transitions_to_immediate_dead"], 1)
        self.assertEqual(result["sampled_states_with_witnessed_goal_path"], 3)
        self.assertEqual(result["same_source_dead_vs_solvable_candidate_pairs"], 1)


if __name__ == "__main__":
    unittest.main()
