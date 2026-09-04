# Direct API GCML Blocks Baseline

This is the current Blocks baseline. It calls an OpenAI-compatible text API directly, so the model receives no Codex task, browser, shell, screen, repository, or other agent tool surface. Each prompt contains only the GCML Blocks rules, one complete 10x10 binary grid, and a JSON output contract.

## Setup

- Model: `gpt-5.6-luna`
- Reasoning effort: `medium`
- API style: Responses API through `https://clawnode.top/v1`
- Inputs: GCML `tiling_order_10x10_8obj.h5` test-slice grids
- Sampling: 16 independent API calls; first 8 use rows 18000--18002, second 8 use previously unused rows 18003--18010
- Action budget: at most 8 actions, matching the official 8-object notebook demonstration; this is not the meaning of `pass@16`
- Judge: applies the official eight shape masks to the current grid, stops at the first illegal action, and requires an all-zero final grid
- Output: JSON `actions` with `shape_id`, zero-based `row`/`col`, optional short public rationale, and `final_status`

## Result

`pass@16 = 12/16 = 0.7500`.

- 12 outputs were parseable, legal, and solved within 8 actions.
- 1 output made an illegal out-of-bounds action; it was stopped immediately.
- 3 outputs used only legal actions but left occupied pixels and incorrectly reported `solved`.
- 0 outputs failed JSON parsing.

The complete combined prompts, API responses, model outputs, and verdicts are in `experiments/gcml_direct_api_blocks/runs/clawnode_luna_20260904_16/run.json`. Summary and raw per-case records are intentionally kept together in this one JSON file. The two component runs are retained at `experiments/gcml_direct_api_blocks/runs/clawnode_luna_20260904/run.json` and `experiments/gcml_direct_api_blocks/runs/clawnode_luna_20260904_additional8/run.json`.

## Prompt format

Every API call receives the following English prompt shape. The harness substitutes the complete grid for `{GRID_ROWS}`; no repository, browser, shell, screen, or other agent context is supplied.

```text
Task: Remove reusable shapes from the following 10x10 binary grid until every cell is 0.

Rules: 1 means occupied and 0 means empty. An action is (shape_id, row, col), with zero-based row and column and the shape's top-left anchor. An action is legal only when every 1-cell of the shape overlaps a current 1; a legal action sets those cells to 0. There is no gravity and no inventory limit. Shapes: 0=11/10, 1=10/11, 2=11/01, 3=01/11, 4=1/1, 5=11, 6=1/1/1, 7=111. Use at most 8 actions.

Initial grid rows:
{GRID_ROWS}

Output: return only valid JSON, with this schema: {"actions":[{"shape_id":0,"row":0,"col":0,"rationale":"short public reason"}],"final_status":"solved"}. Include one object per attempted action in order. The rationale is optional but, if present, must be a short auditable reason. Do not include markdown or hidden chain-of-thought.
```

## Reproduce

```powershell
$env:GND_API_KEY = '...'
$env:GND_BASE_URL = 'https://your-compatible-endpoint/v1'
$env:GND_MODEL = 'gpt-5.6-luna'
$env:GND_API_STYLE = 'responses'
$env:GND_REASONING_EFFORT = 'medium'
node experiments/gcml_direct_api_blocks/src/run_blocks.mjs
```

The script accepts JSON arrays through `GCML_GRID_LIST` and `GCML_CASE_GRID` for an explicitly selected official grid subset. It never writes the API key. By default, the combined run file is written under `experiments/gcml_direct_api_blocks/runs/`; `GCML_RUN_DIR` can override this path.
