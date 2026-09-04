# Direct API GCML 32-Node Path Baseline

This is the current path32 baseline. It calls an OpenAI-compatible text API directly, so the model receives no Codex task, browser, shell, screen, repository, or other agent tool surface. The prompt contains only the official 32-node graph, a start/goal pair, task rules, and a JSON output contract.

## Setup

- Model: `gpt-5.6-luna`
- Reasoning effort: `medium`
- API style: Responses API
- Graph: the fixed 32-node graph extracted from GCML commit `ff76859b71a2bc2056b50f5e052475351c007f76`
- Cases: eight start/goal pairs from the current path pilot, sampled independently twice for 16 calls
- Path budget: none; path length is unrestricted
- Output: JSON `path` with `from`, `to`, optional short public rationale, and `final_node`
- Judge: follows the official undirected edges, stops at the first illegal edge, and separately computes the shortest distance by BFS

`pass@16` means the fraction of 16 independent API samples whose executed path reaches its goal. Optimality is a separate metric and is not silently folded into execution pass.

## Result

The 16-call run reached the goal in all cases: `pass@16 = 16/16 = 1.0000`. Only 5/16 paths were shortest by independent BFS; all 16 `final_node` reports matched the goal. This pair suite is therefore a useful execution sanity check but too easy to distinguish planning quality.

The complete prompts, API responses, model outputs, and verdicts are in `runs/gcml_direct_api_path32/clawnode_luna_20260904_16/run.json`.

## Reproduce

```powershell
$env:GND_API_KEY = '...'
$env:GND_BASE_URL = 'https://your-compatible-endpoint/v1'
$env:GND_MODEL = 'gpt-5.6-luna'
$env:GND_API_STYLE = 'responses'
$env:GND_REASONING_EFFORT = 'medium'
node experiments/gcml_direct_api_path32/src/run_path32.mjs
```

The script reads the formal contract from `configs/path32_v1.json`. Raw prompts, complete API responses, model outputs, and verdicts are written only under the ignored `runs/` path.
