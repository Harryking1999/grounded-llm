# Direct API Blocks Smoke Test

This harness calls an OpenAI-compatible text API directly. It does not create a Codex task and does not expose browser, shell, screen, repository, or other agent tools to the model. The prompt contains only the GCML Blocks rules, the printed grid, and a compact JSON output contract. The local judge applies the official shape masks and stops at the first illegal action.

## Run

PowerShell example (set the endpoint and model supplied by the key owner):

```powershell
$env:GND_API_KEY = '...'
$env:GND_BASE_URL = 'https://your-compatible-endpoint/v1'
$env:GND_MODEL = 'your-model-id'
$env:GND_API_STYLE = 'responses' # or 'chat'
node experiments/gcml_direct_api_blocks/src/run_blocks.mjs
```

The script also accepts `OPENAI_API_KEY`, `OPENAI_BASE_URL`, and `OPENAI_MODEL`. It never writes the key. Raw responses, prompts, and verdicts are written to the ignored path `runs/gcml_direct_api_blocks/latest/run.json`.

## Completed run

The supplied key was accepted by `https://clawnode.top/v1` for model `gpt-5.6-luna` with reasoning effort `medium`. The initial eight calls completed using the `responses` style, with no agent tool surface. That run is saved at `runs/gcml_direct_api_blocks/clawnode_luna_20260904/run.json`.

- `pass@8 = 6/8 = 0.7500`
- Illegal action: case 04, shape 2 at `(2,9)` reaches outside the 10x10 grid; the model still reported `solved`.
- Legal but unsolved: case 05, two occupied cells at row 6 columns 5--6 remained.
- Other six cases were legal, parseable and solved within 8 actions.

The earlier standard endpoint probe at `https://api.openai.com/v1/models` returned HTTP 401; the successful run used the supplied compatible gateway instead.

The eight-action cap follows the official 8-object notebook demonstration budget; it is a Blocks task budget, not the meaning of `pass@8`. The path32 baseline has no corresponding action cap.

## Extended run

Eight additional, previously unused official test-slice grids (HDF5 rows 18003--18010) were run with the same prompt, model, API style, reasoning effort, and external judge. The additional run scored `6/8`; combined with the initial eight it scores `pass@16 = 12/16 = 0.7500`.

- Original cases 01--08: `6/8`; case 04 was immediately illegal and case 05 was legal but left two pixels.
- New cases 09--16: `6/8`; cases 14 and 15 were legal but left one pixel each (source rows 18008 and 18009).
- Across all 16: 1 illegal action, 3 legal-but-unsolved outputs, 0 parse failures, and 12 solved outputs.

The complete combined responses, prompts, and verdicts are in `runs/gcml_direct_api_blocks/clawnode_luna_20260904_16/run.json`; the eight additional responses are also preserved separately in `runs/gcml_direct_api_blocks/clawnode_luna_20260904_additional8/run.json`. To run another official grid selection, pass JSON arrays through `GCML_GRID_LIST` and `GCML_CASE_GRID`.
