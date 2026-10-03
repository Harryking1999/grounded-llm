"""Risk checks: consistent numbering, exact cache replay and metric denominators."""

import json
import multiprocessing
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from experiments.flamingo_map_reader.src import prepare_trajectories
from experiments.flamingo_map_reader.src.sft import greedy_demonstration
from experiments.flamingo_map_reader.src.trajectory_protocol import numbering_plans, renumber_demonstration
from experiments.flamingo_map_reader.src.trajectory_dataset import prepare_record, PreparedTrajectoryDataset, load_record
from experiments.flamingo_map_reader.src.trajectory_metrics import score_turn, summarize_turns, numeric_distances
from experiments.flamingo_map_reader.src.trajectory_queue import checkpoint_jobs
from test_graph import line_graph
from test_timeline import ByteChatTemplate


class TrajectoryProtocolTest(unittest.TestCase):
    def setUp(self):
        self.env, self.qmap = line_graph()
        self.demo = greedy_demonstration(self.env, self.qmap, 1, 2, rng=np.random.default_rng(7))
        self.config = dict(task="graph", distance_precision=4, maximum_sequence_tokens=16384,
                           data=dict(numbering_variants=6), training=dict(epochs=3))

    def test_numbering_keeps_physical_actions_q_and_targets_together(self):
        plans = numbering_plans(self.demo, 6, 42)
        self.assertEqual(len(plans), 2)
        self.assertEqual({p[0] for p in plans}, {(0, 1), (1, 0)})
        for plan in plans:
            demo = renumber_demonstration(self.demo, plan, "graph", 4)
            self.assertEqual(demo.executed_path, self.demo.executed_path)
            turn = demo.turns[0]
            self.assertEqual(turn.step.execute(self.env, turn.chosen_id)[1], 2)
            for i, action in enumerate(turn.step.candidate_actions):
                torch.testing.assert_close(turn.step.map_batch.vectors[0, i + 2],
                    torch.tensor(self.qmap.q[1] + self.qmap.v[action], dtype=torch.float32))
            scored = score_turn(turn.step, turn.answer_text)
            self.assertTrue(scored["exact_ranking"])
            self.assertTrue(scored["action_map_minimum"])
            self.assertEqual(scored["distance_absolute_error"], 0)
            self.assertNotIn("Current map distance", turn.user_text)
            self.assertTrue(score_turn(demo.turns[-1].step, demo.turns[-1].answer_text,
                                       demo.turns[-1].answer_text)["summary_correct"])

    def test_cache_repeats_exact_tokens_and_switches_map_after_action_boundary(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "trajectories").mkdir()
            record = prepare_record(self.demo, dict(split="train", sample_seed=4), self.config,
                                    ByteChatTemplate(), root / "trajectories/task.pt")
            dataset = PreparedTrajectoryDataset(root / "manifest.json", [record], self.config)
            self.assertEqual(len(dataset), 2)
            encoded, steps = dataset[0]
            self.assertEqual(encoded, dataset[0][0])
            text = bytes(encoded.input_ids).decode()
            end = text.index("</action>") + len("</action>")
            self.assertTrue(all(i == 0 for i in encoded.token_map_ids[:end]))
            next_turn = text.index("<user>[Environment update]", end)
            self.assertTrue(all(i == 1 for i in encoded.token_map_ids[next_turn:]))
            self.assertTrue(steps[-1].done)
            replay = load_record(root / "trajectories", record, self.config)
            self.assertEqual(replay.executed_path, self.demo.executed_path)

    def test_missing_numbers_are_not_zero_error_and_false_done_counts(self):
        step = self.demo.turns[0].step
        row = score_turn(step, "<done/>")
        summary = summarize_turns([row])
        self.assertEqual(summary["distance_invalid_turn_rate"], 1.)
        self.assertIsNone(summary["distance_mae_on_valid"])
        self.assertEqual(summary["premature_done_rate"], 1.)
        self.assertEqual(summary["exact_ranking_rate"], 0.)
        self.assertIsNone(numeric_distances("Current map distance to goal: nan.\nCandidate map distances to goal: 1: 0; 2: 1.", 2))

    @unittest.skipUnless(hasattr(os, "fork"), "forked preparation is Linux-only")
    def test_forked_pool_prepares_in_order_with_swapped_arguments_caught(self):
        """Argument order, fork inheritance and payload pickling all fail silently."""
        original = prepare_trajectories.prepare_one
        prepare_trajectories.prepare_one = lambda record, index: dict(index=index, record=record)
        try:
            records = [dict(split="train", start=start) for start in range(12)]
            jobs = [(record, index) for index, record in enumerate(records)]
            with multiprocessing.get_context("fork").Pool(3) as pool:
                payloads = list(pool.imap(prepare_trajectories.prepare_star, jobs, chunksize=1))
        finally:
            prepare_trajectories.prepare_one = original
        self.assertEqual([p["index"] for p in payloads], list(range(12)))
        self.assertEqual([p["record"]["start"] for p in payloads], list(range(12)))

    def test_final_queue_covers_all_test_tasks_and_no_map_controls(self):
        config = dict(self.config, evaluation=dict(train_diagnostic_tasks=256, numbering_variants=6, shard_tasks=64))
        manifest = dict(config=config, records=[dict(split=s) for s in ("train", "validation", "test") for _ in range(129)])
        jobs = checkpoint_jobs("path", Path("/run/final"), manifest, Path("/run"))
        self.assertEqual(len(jobs), 18)
        test = [j for j in jobs if "/test_reference_map/" in j["key"]]
        self.assertEqual(len(test), 3)
        self.assertTrue(test[-1]["key"].endswith("00128_00129"))


if __name__ == "__main__":
    unittest.main()
