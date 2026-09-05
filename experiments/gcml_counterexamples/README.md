# GCML counterexamples: broad pass@8 pilot

This study tests whether Luna can reliably produce complete legal tilings and
shortest paths across independent task instances. It is a task-selection study
before state-network training. One-shot complete plans are the intended protocol
at this stage; no intermediate feedback or model-side tools are available.

The formal sampling, generation and API settings are in
[`configs/pilot.json`](configs/pilot.json). Outputs, exact references, prompts,
complete API responses and verdicts are stored in the ignored `runs/` directory.
Do not infer a new model result from offline difficulty scores.

## Task selection

- **Official eight-block silhouettes:** sample new original test-slice rows,
  excluding previously tested silhouettes and translation duplicates. Select the
  API cases randomly before computing diagnostic scores.
- **Generated twelve-block silhouettes:** use the original shapes and grid.
  Repeatedly choose a nonoverlapping placement adjacent to the existing silhouette.
  Retain the construction decomposition as a solvability witness. The judge accepts
  every complete legal decomposition without an action-count cap. Construction
  count and exact minimum removals are metadata, not success constraints.
- **Original graph shortest paths:** preserve every distinct pair from the previous
  pilot and add random distinct pairs. Unlike the old prompt, this prompt explicitly
  requires a shortest path. Arrival and optimality are scored separately.
- **Switch-gate paths:** preserve all edges of the original graph. Edges crossing
  distance-layer cuts require an associated switch to be open. Check the gate
  before moving; entering the switch node then flips its bit. Every reentry flips
  it, while starting at a switch does not. Goal success depends on the node. Find a
  shortest path in the complete `(node, mask)` state space. Layouts are selected by
  explicit structural criteria before model evaluation and paired over initial
  switch states; paired masks are not independent layouts.

The twelve-block generator differs from the official data generator, so a difference
between the two block groups cannot be attributed solely to the number of objects.
The generated count is a construction count, not necessarily the minimum number of
removals. Exact minimum lengths and overlap/trap statistics are retained per board.

## Offline analysis and case studies

The exact-cover solver records the minimum number of removals and a witness, or
proves that a residual silhouette cannot be tiled. The current block protocol has
no action-count cap. The judge retains optional cap handling only to audit historical
runs. The model's action count divided by the minimum is an efficiency diagnostic
for successful plans, never an extra success requirement.

For each initial board it examines all legal first removals. A trap can leave an
obvious unsupported cell, or leave every cell covered by some legal placement while
the overall board remains untileable. For the latter, a capped legal-prefix search
records how many additional steps can retain local coverage. This operational
measure does not claim a minimum reasoning depth. Random and largest-first rollout
scores are diagnostic proxies; they do not select the main API cases.

For actual model outputs, the judge records the earliest legal action after which
completion becomes impossible, before/after grids, subsequent states, and the first
illegal action if any. This supports case studies without altering the primary
randomly selected test distribution.

## Metrics

`sample_success_rate` is the successful fraction of individual model calls.
**pass@8** is the fraction of instances with at least one successful result among
their eight independent calls. It is candidate availability under an exact judge,
not evidence that an unaided model would select that candidate.

Path pass@8 requires a shortest valid path because shortestness is requested in all
path conditions. Also report execution-only pass@8. Switch reports are auxiliary
diagnostics and do not cause an otherwise correct path to fail. JSON extraction
tolerates surrounding prose like the previous pilot, while strict JSON compliance
is separately recorded. HTTP failures and incomplete responses are reported and
excluded from complete-response pass@8 groups, never silently called model failures.

Eight samples of one instance are not eight independent task instances. The two
initial masks of a gate layout also share structure. Report the per-case counts and
paired layout results alongside aggregates. A broad pilot is not a population-level
or cross-model generalization claim.

## Running

From the repository root, using Python with `h5py` available for dataset extraction:

```powershell
python experiments/gcml_counterexamples/src/prepare.py
python -m unittest discover -s experiments/gcml_counterexamples/tests -v
$env:GND_API_KEY = '...'
python -u experiments/gcml_counterexamples/src/run.py --condition path_shortest --out experiments/gcml_counterexamples/runs/uncapped/path_shortest
python -u experiments/gcml_counterexamples/src/run.py --condition blocks8 --condition blocks12 --out experiments/gcml_counterexamples/runs/uncapped/blocks
python -u experiments/gcml_counterexamples/src/run.py --condition path_gates1 --condition path_gates2 --out experiments/gcml_counterexamples/runs/uncapped/path_gates
```

The runner records the source commit and complete request body without authorization
headers. It saves each completed sample atomically in the combined `run.json`. It
does not overwrite an existing run or retry failed requests automatically. It checks
one scheduled sample before concurrent calls and stops scheduling on API errors.
After an observed transport interruption, `--continue-from <stopped run.json>` retains
every completed answer, including model failures, and schedules only missing answers.
It rejects changed prompts or model budgets and archives prior transport failures.
Analyze the continuation output alone, since it includes the retained answers.
SSE streaming addresses the gateway timeout observed for long buffered requests; only
the final response object is judged. Inspect actual returned token use and limits:
the earlier endpoint did not echo/enforce the
requested output-token cap as expected.

## Current evidence

The suites have been generated and exact references verified. Targeted checks cover
exhaustive tiny tilings, alternative solutions, budget enforcement, stop-on-illegal
behavior, the historical failure points, gate timing, repeated-node product states,
shortestness, pass@8 grouping, unlimited valid decompositions, and streamed response
completion. The original graph shortest-path batch completed with 116/128 shortest
paths and pass@8 of 16/16. The first capped block batch stopped after gateway 524
errors (four model responses, three API errors); those outputs are historical only.
The uncapped block batch uses new prompts and does not reuse capped model answers.
Switch-gate shortest paths are also being evaluated. Full results will replace this
interim evidence once the batches finish.
