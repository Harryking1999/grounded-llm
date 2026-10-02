"""Prevent premature stopping or decision-mask leakage in continued SFT."""

from pathlib import Path
import json
import unittest
import tempfile
from types import SimpleNamespace

from experiments.flamingo_map_reader.src.convergence import ConvergenceCheck, PlateauTracker, with_decision_mask
from experiments.flamingo_map_reader.src.transcript import EncodedTrajectory
from experiments.flamingo_map_reader.src.train import check_resume_target, fixed_warmup_arguments
from transformers import TrainingArguments


class ByteTokenizer:
    def decode(self, ids, **kwargs):
        return bytes(ids).decode("ascii")

    def __call__(self, text, **kwargs):
        return {"input_ids": list(text.encode("ascii")),
                "offset_mapping": [(i, i + 1) for i in range(len(text))]}


class ConvergenceTest(unittest.TestCase):
    def spec(self):
        return json.loads((Path(__file__).parents[1] / "configs/kv_convergence.json").read_text())

    def test_transformers_does_not_reexpand_explicit_warmup_with_new_budget(self):
        kwargs = fixed_warmup_arguments({"warmup_fraction": .05}, 256)
        args = TrainingArguments(output_dir="unused", report_to=[], **kwargs)
        for budget in (256, 8192, 16384):
            self.assertEqual(args.get_warmup_steps(budget), 13)

    def test_plateau_requires_decision_loss_to_stop_improving(self):
        tracker = PlateauTracker(self.spec())
        for step in range(256, 1664, 128):
            self.assertFalse(tracker.update(step, {"training_ce": .1,
                "decision_ce": 2.0 - step / 2000}))
        for step in range(1664, 2304, 128):
            stopped = tracker.update(step, {"training_ce": .1, "decision_ce": 1.2})
        self.assertTrue(stopped)

    def test_budget_extension_preserves_history_and_does_not_double_count_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            rows = [dict(step=256, training_ce=.2, decision_ce=.8),
                    dict(step=384, training_ce=.1, decision_ce=.5),
                    dict(step=512, training_ce=.12, decision_ce=.55)]
            (Path(directory)/"convergence.jsonl").write_text(
                ''.join(json.dumps(row) + '\n' for row in rows))
            check = ConvergenceCheck(self.spec(), directory, [])
            self.assertEqual(check.best_step, 384)
            self.assertEqual(check.best_training, .1)
            self.assertEqual(check.tracker.stale, dict(training_ce=1, decision_ce=1))
            control = SimpleNamespace()
            self.assertIs(check.check(SimpleNamespace(global_step=512), control), control)
            self.assertEqual(check.tracker.stale, dict(training_ce=1, decision_ce=1))

    def test_resume_accepts_an_interrupted_run_only_at_its_last_measurement(self):
        spec = json.loads((Path(__file__).parents[1]
                           / "configs/path_readout_expanded_resume.json").read_text())
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            (output / "models/checkpoint-12288").mkdir(parents=True)
            resume = output / "models/checkpoint-12288"
            (output / "convergence.jsonl").write_text(json.dumps(
                dict(step=12288, training_ce=.06, decision_ce=.70)) + "\n")
            (output / "convergence_status.json").write_text(json.dumps(
                dict(step=12288, status="running")))
            check_resume_target(output, resume, 12288, spec)
            with self.assertRaises(ValueError):
                check_resume_target(output, resume, 11264, spec)
            (output / "convergence_status.json").write_text(json.dumps(
                dict(status="plateau", completed_step=12288)))
            with self.assertRaises(ValueError):
                check_resume_target(output, resume, 12288, spec)

    def test_extension_still_requires_the_exhausted_budget_step(self):
        spec = json.loads((Path(__file__).parents[1]
                           / "configs/kv_convergence_extension.json").read_text())
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            (output / "models/checkpoint-8192").mkdir(parents=True)
            (output / "convergence_status.json").write_text(json.dumps(
                dict(status="budget_reached_not_converged", completed_step=8192)))
            check_resume_target(output, output / "models/checkpoint-8192", 8192, spec)
            with self.assertRaises(ValueError):
                check_resume_target(output, output / "models/checkpoint-8192", 8064, spec)

    def test_plateau_does_not_call_loss_regression_converged(self):
        tracker = PlateauTracker(self.spec())
        tracker.update(256, {"training_ce": .1, "decision_ce": .5})
        for step in range(384, 2048, 128):
            self.assertFalse(tracker.update(step, {"training_ce": .3, "decision_ce": 1.5}))

    def test_mask_ignores_prompt_format_and_copied_action(self):
        prompt = "<closer>99</closer>"
        answer = "Map-distance ranking to the goal, closest to farthest: 2 < 1.\n<action>2</action>"
        ids = list((prompt + answer).encode("ascii"))
        encoded = EncodedTrajectory(ids, [-100] * len(prompt) + ids[len(prompt):],
                                    [0] * len(ids), len(answer))
        marked, _ = with_decision_mask((encoded, []), ByteTokenizer())
        selected = bytes(i for i, marked in zip(ids, marked.focus_mask) if marked).decode()
        self.assertEqual(selected, "2 < 1")
        short = "<closer>42</closer>"
        encoded = EncodedTrajectory(list(short.encode()), list(short.encode()), [0]*len(short), len(short))
        marked, _ = with_decision_mask((encoded, []), ByteTokenizer())
        self.assertEqual(sum(marked.focus_mask), 2)


if __name__ == "__main__":
    unittest.main()
