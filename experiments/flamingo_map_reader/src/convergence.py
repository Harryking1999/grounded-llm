"""Fixed-training-set loss checks for budget extensions of existing SFT runs."""

from dataclasses import replace
import json
import math
from pathlib import Path
import re
import statistics

import torch
from torch.nn import functional as F
from transformers import TrainerCallback


def with_decision_mask(example, tokenizer):
    """Mark gold ranking contents or short choice digits, never prompt tokens."""
    encoded, steps = example
    focus = [False] * len(encoded.labels)
    start = 0
    while start < len(encoded.labels):
        if encoded.labels[start] == -100:
            start += 1
            continue
        end = start
        while end < len(encoded.labels) and encoded.labels[end] != -100:
            end += 1
        ids = encoded.labels[start:end]
        answer = tokenizer.decode(ids, skip_special_tokens=False,
                                  clean_up_tokenization_spaces=False)
        tokenized = tokenizer(answer, add_special_tokens=False,
                              return_offsets_mapping=True)
        if tokenized["input_ids"] != ids:
            raise ValueError("Decision loss mask needs exact answer-token alignment")
        matches = list(re.finditer(
            r"(?m)^Map-distance ranking to the goal, closest to farthest: (.+)\.$",
            answer))
        matches += list(re.finditer(r"<(?:closer|nearest)>([1-9][0-9]*)</", answer))
        spans = [match.span(1) for match in matches]
        for offset, (left, right) in enumerate(tokenized["offset_mapping"]):
            focus[start + offset] = any(left < b and right > a for a, b in spans)
        start = end
    if not any(focus):
        raise ValueError("Training example has no ranking or choice supervision")
    return replace(encoded, focus_mask=focus), steps


class PlateauTracker:
    """Require a plateau in both total CE and decision CE, not formatting alone."""

    def __init__(self, spec):
        self.spec = spec
        self.best = {}
        self.reference = {}
        self.stale = {}

    def update(self, step, metrics):
        for name in self.spec["metrics"]:
            value = metrics[name]
            if not math.isfinite(value):
                raise ValueError(f"Non-finite convergence metric: {name}")
            self.best[name] = min(value, self.best.get(name, math.inf))
            reference = self.reference.get(name)
            threshold = max(self.spec["absolute_improvement"],
                            abs(reference or 0) * self.spec["relative_improvement"])
            if reference is None or value < reference - threshold:
                self.reference[name], self.stale[name] = value, 0
            else:
                self.stale[name] += 1
        stable = all(metrics[name] <= self.best[name] + max(
            self.spec["maximum_final_loss_regression_absolute"],
            self.best[name] * self.spec["maximum_final_loss_regression_fraction"])
            for name in self.spec["metrics"])
        return (step >= self.spec["minimum_total_steps"] and stable and
                all(self.stale[name] >= self.spec["patience_evaluations"]
                    for name in self.spec["metrics"]))


class ConvergenceCheck(TrainerCallback):
    """Evaluate unchanged training examples and persist the selection evidence."""

    def __init__(self, spec, output, examples):
        self.spec, self.output, self.examples = spec, Path(output), examples
        self.tracker = PlateauTracker(spec)
        self.trainer = None
        self.best_training = math.inf
        self.best_step = None
        self.converged = False
        self.last_step = None
        history = self.output / "convergence.jsonl"
        if history.exists():
            for line in history.read_text(encoding="utf-8").splitlines():
                row = json.loads(line)
                self.tracker.update(row["step"], row)
                if row["training_ce"] < self.best_training:
                    self.best_training, self.best_step = row["training_ce"], row["step"]
                self.last_step = row["step"]

    @torch.inference_mode()
    def measure(self):
        trainer = self.trainer
        model = trainer.model
        was_training = model.training
        model.eval()
        total, decision = [], []
        try:
            for example in self.examples:
                inputs = trainer._prepare_inputs(trainer.data_collator([example]))
                mask = inputs.pop("focus_mask")[:, 1:]
                labels = inputs["labels"][:, 1:]
                selected = mask & (labels != -100)
                with trainer.compute_loss_context_manager():
                    outputs = model(**inputs)
                total.append(float(outputs.loss))
                logits = outputs.logits[:, :-1][selected].float()
                decision.append(float(F.cross_entropy(logits, labels[selected])))
                del outputs, inputs, logits
        finally:
            model.train(was_training)
        return {"training_ce": statistics.mean(total),
                "decision_ce": statistics.mean(decision)}

    def check(self, state, control):
        # A budget extension resumes exactly at the last measured checkpoint.
        # Do not count that point twice toward the patience window.
        if self.last_step == state.global_step:
            return control
        metrics = self.measure()
        self.converged = self.tracker.update(state.global_step, metrics)
        if metrics["training_ce"] < self.best_training:
            self.best_training = metrics["training_ce"]
            self.best_step = state.global_step
            self.trainer.save_model(str(self.output / "models/best_loss"))
        row = dict(step=state.global_step, epoch=state.epoch, **metrics,
                   stale=dict(self.tracker.stale), best_step=self.best_step,
                   plateau=self.converged)
        with (self.output / "convergence.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")
        status = dict(row, status="plateau" if self.converged else "running",
                      selected_checkpoint="models/best_loss/adapter.pt")
        (self.output / "convergence_status.json").write_text(
            json.dumps(status, indent=2) + "\n", encoding="utf-8")
        print("CONVERGENCE " + json.dumps(row), flush=True)
        self.last_step = state.global_step
        if self.converged:
            control.should_training_stop = True
            control.should_save = True
        return control

    def on_train_begin(self, args, state, control, **kwargs):
        return self.check(state, control)

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % self.spec["evaluation_every_steps"] == 0:
            return self.check(state, control)
        return control

    def on_train_end(self, args, state, control, **kwargs):
        path = self.output / "convergence_status.json"
        status = json.loads(path.read_text(encoding="utf-8"))
        status["status"] = "plateau" if self.converged else "budget_reached_not_converged"
        status["completed_step"] = state.global_step
        path.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
