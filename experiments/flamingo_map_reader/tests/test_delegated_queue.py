"""A shared-storage worker must wait for final and leave other jobs untouched."""

import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from experiments.flamingo_map_reader.src import trajectory_queue as queue


class DelegatedQueueTest(unittest.TestCase):
    def test_waits_for_final_and_runs_only_owned_missing_shards(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            config = dict(training=dict(epochs=5), checkpoint=dict(early_steps=[]),
                          evaluation=dict(train_diagnostic_tasks=1, numbering_variants=6, shard_tasks=64))
            for task in ("path", "blocks"):
                data = root / task / "data"
                data.mkdir(parents=True)
                (data / "manifest.json").write_text(json.dumps(dict(
                    config=config, records=[dict(split=s) for s in ("train", "validation", "test")])))
            def publish(task, name, step, epoch):
                directory = root / task / "training/models" / name
                directory.mkdir(parents=True)
                (directory / "evaluation_ready.json").write_text(json.dumps(dict(step=step, epoch=epoch)))
            publish("path", "final", 3, 3)
            # The numbered end-of-budget save gets no battery; final must arrive later.
            publish("blocks", "checkpoint-5", 5, 5)
            (root / "queue_status.json").write_text('{"pid": 999}')
            done = root / "evaluation/blocks/checkpoint-4/validation_reference_map/00000_00001"
            publish("blocks", "checkpoint-4", 4, 4)
            done.mkdir(parents=True)
            (done / "summary.json").write_text("{}")
            (done / "cases.jsonl").write_text("{}\n")
            launched = []
            def launch(command, **kwargs):
                launched.append(command)
                output = Path(command[command.index("--out") + 1])
                output.mkdir(parents=True)
                (output / "summary.json").write_text("{}")
                (output / "cases.jsonl").write_text("{}\n")
                return SimpleNamespace(pid=100 + len(launched), poll=lambda: 0)
            def advance(_):
                if not (root / "blocks/training/models/final").exists():
                    publish("blocks", "final", 5, 5)
            argv = ["queue", "--run-root", str(root), "--model-path", "unused",
                    "--evaluation-only", "--gpus", "0", "1", "2",
                    "--status-file", str(root / "delegate_status.json"),
                    "--include-prefix", "path/final/test_rollout_map/",
                    "--include-prefix", "blocks/final/validation_reference_map/",
                    "--include-prefix", "blocks/checkpoint-4/validation_reference_map/"]
            evaluator = SimpleNamespace(aggregate=lambda cases, manifest: dict(cases=len(cases)))
            with patch.object(sys, "argv", argv), patch.object(queue.subprocess, "Popen", launch), \
                 patch.object(queue.time, "sleep", advance), patch.dict(sys.modules, {
                     "experiments.flamingo_map_reader.src.trajectory_eval": evaluator}):
                queue.main()
            self.assertEqual(len(launched), 2)
            outputs = [Path(c[c.index("--out") + 1]).relative_to(root / "evaluation").as_posix() for c in launched]
            self.assertEqual(set(outputs), {"path/final/test_rollout_map/00000_00001",
                                            "blocks/final/validation_reference_map/00000_00001"})
            self.assertTrue(all(queue.MODULE + "trajectory_eval" in c for c in launched))
            self.assertEqual(json.loads((root / "queue_status.json").read_text()), dict(pid=999))
            self.assertTrue(json.loads((root / "delegate_status_completion.json").read_text())["success"])
            self.assertFalse((root / "completion.json").exists())


if __name__ == "__main__":
    unittest.main()
