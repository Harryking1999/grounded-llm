"""Small exact-state checks for the imported tree evaluation tools."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from experiments.blocks_distance_map.src import tree_breadth_eval, tree_distance_eval, tree_fixed_eval
from experiments.blocks_distance_map.src.oracle import DistanceOracle


class ExactTestScorer:
    identity = {"kind": "test_only_exact_distance"}

    def scores(self, masks, goal):
        oracle = DistanceOracle()
        return np.asarray([value if (value := oracle.distance(mask, goal)) >= 0 else 100
                           for mask in masks])


class TreeEvaluationTests(unittest.TestCase):
    def test_distance_pairs_retain_their_source_board(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            np.savez(root / "base.npz", train_board_rows=[1], ood_board_rows=[7])
            (root / "config.json").write_text(json.dumps({"board_seed": 1}))
            (root / "comparison.json").write_text(json.dumps({
                "board_rows": [9, 10], "skip_fresh_boards": 1}))
            boards = [(7, 7, ()), (8, 7, ()), (9, 63, ()), (10, 511, ())]
            with patch.object(tree_distance_eval, "official_boards", return_value=([], boards)), \
                    patch.object(tree_distance_eval, "sample_paths",
                                 side_effect=lambda board, count, rng: [[board[1], 7, 0]]), \
                    redirect_stdout(io.StringIO()):
                tree_distance_eval.build(root / "base.npz", root / "config.json",
                                         root / "unused.h5", root / "comparison.json",
                                         root / "pairs.npz")
            with np.load(root / "pairs.npz", allow_pickle=False) as saved:
                self.assertEqual(len(saved["pair_board"]), len(saved["test_ood_board"]))
                self.assertEqual(set(saved["pair_board"]), {0, 1})
                for pair, board in zip(saved["test_ood_board"], saved["pair_board"]):
                    state_masks = {int(saved["states"][pair[0]]), int(saved["states"][pair[1]])}
                    self.assertTrue(state_masks <= ({0, 7, 63} if board == 0 else {0, 7, 511}))

    def test_breadth_scores_first_step_and_goal_groups(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tasks = [dict(source="63", goal="0", true_steps=2, goal_type="empty", source_cells=6),
                     dict(source="63", goal="56", true_steps=1, goal_type="ordinary", source_cells=6)]
            (root / "tasks.json").write_text(json.dumps({"split": "sealed", "tasks": tasks}))
            with patch.object(tree_breadth_eval, "QScorer", return_value=ExactTestScorer()), \
                    redirect_stdout(io.StringIO()):
                result = tree_breadth_eval.score(root / "tasks.json", root / "unused.pt",
                                                 root / "scores.json", device="cpu")
            self.assertEqual(result["reached"], 2)
            self.assertEqual(result["first_step"], dict(decisions=2, reachable=2, shortest=2))
            self.assertEqual(result["by_goal_type"]["empty"]["tasks"], 1)
            self.assertEqual(result["by_goal_type"]["ordinary"]["tasks"], 1)

    def test_fixed_scores_keep_clear_and_nonempty_separate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "comparison.json").write_text(json.dumps({
                "task_rows": [dict(source="63", goal="56", true_steps=1)]}))
            (root / "clear.json").write_text(json.dumps({
                "rows": [dict(source="63", true_steps=2)]}))
            with patch.object(tree_fixed_eval, "QScorer", return_value=ExactTestScorer()), \
                    redirect_stdout(io.StringIO()):
                result = tree_fixed_eval.evaluate(root / "comparison.json", root / "clear.json",
                                                  root / "unused.pt", root / "scores.json", device="cpu")
            for kind in ("nonempty", "clear"):
                self.assertEqual(result["results"][kind]["tasks"], 1)
                self.assertEqual(result["results"][kind]["outcomes"]["reached"], 1)


if __name__ == "__main__":
    unittest.main()
