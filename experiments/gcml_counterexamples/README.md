# GCML counterexamples: broad pass@8 pilot

This study tests whether API models can reliably produce complete legal tilings and
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

The offline pool contains 352 distinct candidates. For example, removing shape 0
at `(4,0)` in official row 18221 creates an untileable state while every remaining
cell is still locally covered; a legal continuation can preserve that property for
three more actions. This board was not in the primary API sample and is an offline
trap candidate, not an observed Luna counterexample.

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
Only transport errors and explicit upstream/server errors are eligible to be filled.
A response explicitly truncated by its output-token budget is retained as a final
failed trial, and scheduling continues; unexplained incomplete responses still stop
the run. Budgeted pass@8 includes truncated trials in its eight-attempt denominator,
while completed-response-only pass@8 remains separately available.
Analyze the continuation output alone, since it includes the retained answers.
SSE streaming addresses the gateway timeout observed for long buffered requests; only
the final response object is judged. Inspect actual returned token use and limits:
the earlier endpoint did not echo/enforce the
requested output-token cap as expected.

## Results

### DeepSeek Flash comparison

The user authorized a second model on the same suite. The comparison contract is
[`configs/deepseek_flash.json`](configs/deepseek_flash.json); it overrides the
model, endpoint, output-token cap and request concurrency. All cases,
prompts, rules, eight calls per case, and offline judges remain identical. Run it
with `--api-config experiments/gcml_counterexamples/configs/deepseek_flash.json`
and a separate output directory after setting the corresponding API credential.

DeepSeek's official [thinking-mode documentation](https://api-docs.deepseek.com/guides/thinking_mode/)
maps the requested `medium` to its `high` effort. The first scheduled trial under
the inherited cap exhausted its entire budget on reasoning and returned no answer;
it remains in `runs/deepseek_flash/run.json` as a separate preliminary observation.
Luna's gateway did not enforce that cap. The formal group therefore starts all
samples afresh with a uniformly larger cap, under the updated committed contract,
in `runs/deepseek_flash_full/`. This is not matched compute. Report actual token use
and truncations separately; neither the preliminary trial nor any resampled model
failure is added to the formal eight samples. No formal result is available yet.

### Luna completed pilot

The suites have been generated and exact references verified. Targeted checks cover
exhaustive tiny tilings, alternative solutions, budget enforcement, stop-on-illegal
behavior, the historical failure points, gate timing, repeated-node product states,
shortestness, pass@8 grouping, unlimited valid decompositions, and streamed response
completion. All 512 completed answers across 64 instances have been independently
rejudged. Every instance has exactly eight answers. Stored prompts match the current
protocol, all reference plans execute successfully, and the primary block boards
are distinct after removing translations. Aggregate and per-instance counts are in
[`results/summary.json`](results/summary.json).

| Condition | Successful answers | Valid arrivals (path) | pass@8 instances |
| --- | ---: | ---: | ---: |
| Official eight-block silhouettes | 99/128 (77.34%) | — | 15/16 (93.75%) |
| Generated twelve-block silhouettes | 72/128 (56.25%) | — | 15/16 (93.75%) |
| Original graph | 116/128 | 128/128 | 16/16 |
| One switch | 60/64 | 60/64 | 8/8 |
| Two switches | 40/64 | 55/64 | 7/8 |

Each switch condition contains four independent layouts with two paired initial
states per layout. The completed path records are in
`runs/pilot/path_shortest_compatible/run.json` and
`runs/uncapped/path_gates_reconnect/run.json`. The latter preserves completed
answers from interrupted batches. The completed block record is
`runs/uncapped/blocks_reconnect3/run.json`. Combined analysis and case-study proofs
are in `runs/uncapped/analysis.json`; re-create it using:

```powershell
python experiments/gcml_counterexamples/src/analyze.py --run experiments/gcml_counterexamples/runs/pilot/path_shortest_compatible/run.json --run experiments/gcml_counterexamples/runs/uncapped/path_gates_reconnect/run.json --run experiments/gcml_counterexamples/runs/uncapped/blocks_reconnect3/run.json --render 12
```

Across these run lineages, four connection errors and three explicit server/upstream
failures were archived and filled, keeping all completed successes and model failures.
An earlier rejected-User-Agent path run (128 HTTP 403 errors, no answers) and a capped
block run (four answers, three HTTP 524 errors) remain separate historical incidents.
None of their answers is included in the uncapped block results.

The cap change matters in actual answers: **86/99** successful eight-block plans use
more than eight actions, and **71/72** successful twelve-block plans use more than
twelve. Successful lengths range from 7–11 and 12–17, respectively. These are valid
decompositions. Their outcomes under a different prompt cannot be inferred by simply
imposing the old cap after the fact.

### Block failures and case studies

Final stopping reasons are 26 illegal actions, two dead ends and one premature stop
for eight-block cases; 37 illegal actions, 15 dead ends and four premature stops for
twelve-block cases. Looking earlier in each trajectory finds **30 answers across 16
boards** whose legal moves already made completion impossible: eight answers on seven
official boards and 22 answers on nine generated boards. Thirteen of these answers
later also submit an illegal move. Report both the first irreversible event and the
eventual stopping reason.

All 30 first dead-end states already contain a cell unsupported by any legal shape.
The offline pool contains less obvious traps, but this model run does not establish
that detecting its observed dead ends requires a long lookahead. Even a one-step
state rollout plus a residual-feasibility signal is therefore a concrete useful
target for the next comparison.

- **`blocks8_row18465`, 0/8:** all answers eventually submit illegal actions. In
  sample 8, legal action 3 removes the vertical triple at `(2,4)`, isolating `(2,3)`.
  The preceding state has a six-action completion beginning with shape 2 at `(2,3)`.
  This gives both a stable failed instance and a specific legal-move trap.
- **`blocks12_generated025`, 0/8:** seven answers end on illegal actions and one in
  a dead end. Sample 2 first loses at legal action 9: the horizontal pair at `(5,7)`
  strands `(5,9)`. From the preceding state, a horizontal triple at the same anchor
  starts a five-action valid suffix. Four of this board's answers enter a dead end
  before their eventual stopping point.
- **`blocks8_row19456`, 6/8:** sample 1 reaches a six-cell residual that is solvable
  in two actions, then its legal action 8 isolates `(1,4)`. This is a useful dead-end
  case without labeling the entire board unsolved by the model.
- **`blocks8_row19239`, 7/8:** sample 2 removes `(5,5)` at action 2, then includes
  the same cell in action 3. The state after action 2 is still solvable in five
  actions. This separates plan-state consistency from irreversible planning loss.

Coordinates above are zero-based `(row, column)`. The illegal-action audit finds
56 actions overlapping initially empty cells and seven overlapping previously
removed cells (no mixed cases). Both matter to an observation/state interface, but
the former alone is not evidence of forgetting an earlier action.

### Inference budget actually observed

All responses report the requested Luna model name. The gateway returns a null output
cap for all 512 answers, and one completed answer uses 8,965 output tokens despite
the requested 8,000. Treat the request as an intended setting, not a verified hard
cap. Actual output-token totals/medians/maxima are in the compact summary. Median
output tokens are 1,956 for eight-block cases, 2,846.5 for twelve-block cases, 195 for
original paths, 459.5 for one switch and 967 for two switches. Raw provider response
instructions are retained in the ignored logs; these results describe this gateway
and prompt condition, not all closed frontier models.

### A confirmed two-switch counterexample

For `path_gates2_layout0_mask0`, start 21 and goal 3 with both switches initially
zero, all eight answers fail on a closed gate. The paired initial state `[1, 0]`
has seven shortest answers and eight valid arrivals out of eight. A valid shortest
path from `[0, 0]` is:

```text
node:      21 -> 23 -> 21 -> 22 -> 12 -> 13 -> 2 -> 3
switches:  00    10    10    10    11    11    11   11
```

Bits above are written in switch-index order, not as a binary integer. Entering
23 opens switch 0, which permits entry from 22 into 12; entering 12 opens switch 1.
Five answers instead take `21 -> 23 -> 20 -> 7 -> 8`, then attempt the closed
`8 -> 4` edge. One attempts the other closed edge `10 -> 4`; two attempt `22 -> 12`
before opening switch 0. Every switch report on these eight legal prefixes is
correct (27/27). This localizes the observed problem to using the state to constrain
actions and coordinate subgoals; it does not establish that the model forgot the
switch values. This case supports testing a state-and-transition interface that
informs action evaluation, alongside the simpler baseline of explicit text state
reports already present in this prompt.

## Implications for the next experiment

Keep the current complete-plan protocol and fixed per-instance sampling as the
counterexample baseline. **Do not add sixteen-piece cases in this pilot:** original
eight-piece, generated twelve-piece and two-switch tests already yield 0/8 instances,
and their concrete failure mechanisms are more actionable now than another scale.
The following are proposed follow-ups, not additional
conditions silently included in these results:

- **Observation-to-state encoding:** compare the binary-row input with an equivalent
  indexed coordinate representation on the same boards. The illegal-action audit
  separates initially empty cells from cells removed earlier in the plan. This
  identifies whether an observation encoder or an update mechanism is the more
  useful first component; retain the original-input result as well.
- **Using the state during action evaluation:** in the gate case, the model reports
  the state accurately yet violates a gate. A learned interface should support
  action-conditioned consequences and validity, in addition to state queries.
  Evaluate those predictions separately before attributing planning gains to them.
- **Planning beyond immediate legality:** retain the first irreversible block move,
  its residual board, and a valid alternative from the same preceding state.
  Distinguish these failures from simple premature termination. A state rollout
  that exposes an unsupported cell is a concrete first target; the capped local-
  support proxy is not evidence of a required reasoning depth.
- **Controlled scaling and transfer:** use the same generator for an additional
  eight-versus-twelve comparison before treating their score difference as a pure
  piece-count effect. Keep original graph paths as a successful control and focus
  the next state-dependent tests on gate layouts. Subsequent learned comparisons
  should use held-out boards/layouts and matched information and sampling budgets.
