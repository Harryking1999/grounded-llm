"""Risk checks: consistent numbering, exact cache replay and metric denominators."""

import json
import multiprocessing
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from experiments.flamingo_map_reader.src import prepare_trajectories
from experiments.flamingo_map_reader.src.sft import decision_text, greedy_demonstration, reported_candidates
from experiments.flamingo_map_reader.src.trajectory_protocol import numbering_plans, renumber_demonstration
from experiments.flamingo_map_reader.src.trajectory_dataset import prepare_record, PreparedTrajectoryDataset, load_record
from experiments.flamingo_map_reader.src.trajectory_metrics import score_turn, summarize_turns
from experiments.flamingo_map_reader.src.trajectory_queue import (
    MODULE, available_gpus, checkpoint_jobs, checkpoint_progress, clear_partial_shard,
    completed_shards, live_children, resume_point, still_running)
from test_graph import line_graph
from test_timeline import ByteChatTemplate


class TrajectoryProtocolTest(unittest.TestCase):
    def setUp(self):
        self.env, self.qmap = line_graph()
        self.demo = greedy_demonstration(self.env, self.qmap, 1, 2, rng=np.random.default_rng(7))
        self.config = dict(task="graph", maximum_sequence_tokens=16384,
                           data=dict(numbering_variants=6), training=dict(epochs=3))

    def test_numbering_keeps_physical_actions_q_and_targets_together(self):
        plans = numbering_plans(self.demo, 6, 42)
        self.assertEqual(len(plans), 2)
        self.assertEqual({p[0] for p in plans}, {(0, 1), (1, 0)})
        for plan in plans:
            demo = renumber_demonstration(self.demo, plan, "graph")
            self.assertEqual(demo.executed_path, self.demo.executed_path)
            turn = demo.turns[0]
            self.assertEqual(turn.step.execute(self.env, turn.chosen_id)[1], 2)
            for i, action in enumerate(turn.step.candidate_actions):
                torch.testing.assert_close(turn.step.map_batch.vectors[0, i + 2],
                    torch.tensor(self.qmap.q[1] + self.qmap.v[action], dtype=torch.float32))
            scored = score_turn(turn.step, turn.answer_text)
            self.assertTrue(scored["exact_ranking"])
            self.assertTrue(scored["action_map_minimum"])
            self.assertNotIn("Current map distance", turn.user_text)
            # Only the ordering is supervised; no distance is ever written out.
            self.assertNotIn("Current map distance to goal:", turn.answer_text)
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

    def test_missing_ranking_is_not_zero_error_and_false_done_counts(self):
        step = self.demo.turns[0].step
        row = score_turn(step, "<done/>")
        summary = summarize_turns([row])
        self.assertEqual(summary["premature_done_rate"], 1.)
        self.assertEqual(summary["exact_ranking_rate"], 0.)
        self.assertEqual(summary["current_relation_total"], len(step.candidate_actions))
        self.assertEqual(summary["current_relation_correct"], 0)

    def test_narrowed_answer_names_only_the_nearest_candidates(self):
        """Every legal move stays in the environment; only the answer narrows."""
        step = SimpleNamespace(done=False, candidate_actions=tuple(range(12)),
            current_map_distance=5., candidate_map_distances=tuple(float(10 + i) for i in range(12)),
            map_minimal_candidates=(1,))
        self.assertEqual(reported_candidates(step, 10), tuple(range(1, 11)))
        answer = decision_text(step, 1, 10)
        self.assertEqual(len(answer.splitlines()), 4)
        scored = score_turn(step, answer, reported=10)
        self.assertTrue(scored["exact_ranking"])
        self.assertEqual(scored["candidates"], 12)
        self.assertEqual(scored["current_relation_correct"], 10)
        # The same answer no longer covers the task once every candidate must be ranked.
        self.assertFalse(score_turn(step, answer)["valid_ranking"])

    def test_tied_distances_still_respect_the_cap(self):
        """Ties must not stretch the answer past the cap the parser enforces."""
        step = SimpleNamespace(done=False, candidate_actions=tuple(range(12)),
            current_map_distance=5., candidate_map_distances=(7.,) * 12,
            map_minimal_candidates=tuple(range(1, 13)))
        named = reported_candidates(step, 10)
        self.assertEqual(named, tuple(range(1, 11)))
        scored = score_turn(step, decision_text(step, 1, 10), reported=10)
        self.assertTrue(scored["exact_ranking"])
        self.assertTrue(scored["closest_candidate_set_exact"])

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

    def test_restart_reuses_finished_work_instead_of_redoing_it(self):
        """A restarted queue must not re-prepare, re-smoke or re-evaluate."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            models = root / "path/training/models"
            for step in (0, 128, 8064):
                (models / f"checkpoint-{step}").mkdir(parents=True)
                (models / f"checkpoint-{step}/evaluation_ready.json").write_text("{}")
            # A crash between saves leaves a directory Trainer never published,
            # and the closing "final" copy is published too but is not a step.
            (models / "checkpoint-8192").mkdir()
            (models / "final").mkdir()
            (models / "final/evaluation_ready.json").write_text("{}")
            self.assertEqual(resume_point(models).name, "checkpoint-8064")
            self.assertIsNone(resume_point(root / "absent/models"))

            evaluation = root / "evaluation"
            shard = evaluation / "path/checkpoint-0/train_reference_map/00000_00002"
            shard.mkdir(parents=True)
            (shard / "summary.json").write_text("{}")
            # The aggregate the queue writes sits one level up and is not a shard.
            (evaluation / "path/checkpoint-0/train_reference_map/summary.json").write_text("{}")
            self.assertEqual(completed_shards(evaluation),
                             {"path/checkpoint-0/train_reference_map/00000_00002"})

            partial = evaluation / "blocks/checkpoint-128/train_reference_map/00000_00064"
            partial.mkdir(parents=True)
            (partial / "cases.jsonl").write_text("{}\n")
            clear_partial_shard(partial)
            self.assertFalse(partial.exists())
            clear_partial_shard(shard)
            self.assertTrue(shard.exists())

    def test_restart_orders_the_newest_checkpoint_first_and_adopts_live_children(self):
        """A reorder must not cost the run: newest reading first, no second start."""
        names = ["checkpoint-0", "checkpoint-128", "checkpoint-8064", "checkpoint-49000", "final"]
        order = sorted((Path(name) for name in names), key=checkpoint_progress, reverse=True)
        self.assertEqual([path.name for path in order],
                         ["final", "checkpoint-49000", "checkpoint-8064", "checkpoint-128", "checkpoint-0"])

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shard = "blocks/checkpoint-128/train_reference_map/00000_00064"
            running = [subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)", *arguments])
                       for arguments in ([MODULE + "train", "--out", str(root / "path/training")],
                                         [MODULE + "trajectory_eval", "--out", str(root / "evaluation" / shard)],
                                         [MODULE + "trajectory_eval", "--out", "/elsewhere/evaluation"])]
            dead = subprocess.Popen([sys.executable, "-c", "pass"])
            dead.wait()
            try:
                adopted = live_children(dict(running={
                    "path/train": dict(pid=running[0].pid, gpu=0, log="trainer"),
                    shard: dict(pid=running[1].pid, gpu=2, log="shard"),
                    # Wrong module for the key, and a foreign output directory.
                    "blocks/train": dict(pid=running[2].pid, gpu=1, log="elsewhere"),
                    "path/smoke": dict(pid=dead.pid, gpu=0, log="gone")}), root)
                self.assertEqual(set(adopted), {"path/train", shard})
                self.assertEqual(adopted["path/train"]["pid"], running[0].pid)
                self.assertEqual(live_children(None, root), {})
                # Adopted work stays adopted only while /proc still has the pid.
                self.assertTrue(still_running(running[0].pid))
                self.assertFalse(still_running(dead.pid))
            finally:
                for process in (*running, dead):
                    process.kill()
                    process.wait()

    def test_an_adopted_child_keeps_its_gpu_out_of_the_pool(self):
        """Only the queue's own process table was consulted, which double-booked
        GPUs 2 and 3 the first time a restart adopted live evaluations."""
        tasks = {"path": dict(gpu=0, phase="train"), "blocks": dict(gpu=1, phase="train")}
        shard = dict(gpu=3, pid=1713)
        self.assertEqual(available_gpus([0, 1, 2, 3], tasks, [shard]), [2])
        # A finished trainer releases its card to evaluation.
        trained = {"path": dict(gpu=0, phase="trained"), "blocks": dict(gpu=1, phase="train")}
        self.assertEqual(available_gpus([0, 1, 2, 3], trained, [shard]), [0, 2])
        # So does a job that never had one.
        self.assertEqual(available_gpus([0, 1, 2, 3], trained, [dict(gpu=None)]), [0, 2, 3])

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
