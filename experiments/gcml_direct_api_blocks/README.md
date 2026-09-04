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

The current supplied key returned HTTP 401 from `https://api.openai.com/v1/models`, so its compatible `base_url` and model still need to be supplied before an API run can be accepted as evidence.
