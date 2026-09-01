# Experiment harness

Create a study directory only when there is a concrete hypothesis or evaluator. The default shape is:

```text
experiments/<study>/
  README.md          hypothesis, baselines, metrics, current conclusion
  configs/           committed machine-readable formal run contracts
  src/               reusable environment or method implementation
  tests/             targeted correctness tests
```

Generated state belongs outside tracked source:

```text
runs/<run_id>/       resolved config, predictions, metrics, logs, artifacts
```

`runs/` is ignored by Git. Promote only compact, decision-relevant results into the study README or the authoritative project status.

## Minimal run contract

A formal run should identify:

- model and inference settings;
- environment/rule version and dataset split;
- observation encoding and available tools;
- random seed where applicable;
- primary metrics;
- source commit and output directory.

Exploratory smokes do not need production-grade manifests. Record only enough to reproduce the observation that informs the next decision.
