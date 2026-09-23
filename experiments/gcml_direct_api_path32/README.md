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

The complete prompts, API responses, model outputs, and verdicts are in `experiments/gcml_direct_api_path32/runs/clawnode_luna_20260904_16/run.json`. Summary and raw per-case records are intentionally kept together in this one JSON file.

## Prompt format

Every API call receives the following English prompt shape. The harness substitutes `{START}`, `{GOAL}`, and `{ADJACENCY}`; no repository, browser, shell, screen, or other agent context is supplied.

```text
Task: Find a path in the official 32-node undirected graph from start node {START} to goal node {GOAL}.

Rules: Nodes are numbered 0 through 31. Each move must follow one listed graph edge. There is no step limit. Plan the complete path before answering. The external judge will execute the moves in order from the stated start node and will stop at the first illegal move.

Graph neighbors:
{ADJACENCY}

Output: return only valid JSON with this schema: {"path":[{"from":{START},"to":{NEXT_NODE},"rationale":"short public reason"}],"final_node":{GOAL}}. Use one object per attempted move in order. The rationale is optional but, if present, must be a short auditable reason. Do not include markdown.
```

## Reproduce

```powershell
$env:GND_API_KEY = '...'
$env:GND_BASE_URL = 'https://your-compatible-endpoint/v1'
$env:GND_MODEL = 'gpt-5.6-luna'
$env:GND_API_STYLE = 'responses'
$env:GND_REASONING_EFFORT = 'medium'
node experiments/gcml_direct_api_path32/src/run_path32.mjs
```

The script reads the formal contract from `configs/path32_v1.json`. By default, the single combined summary/raw log is written under `experiments/gcml_direct_api_path32/runs/`; `GCML_RUN_DIR` can override this path. Raw prompts, complete API responses, model outputs, and verdicts are not split into separate files.

## Harder path variants (design only)

The current baseline uses the node itself as the complete state, so an action `(u, v)` directly names the next state. That is deliberately retained as the control, but it is too weak to diagnose state maintenance. Candidate variants below keep the official graph and edge legality while changing the task state:

1. **Ordered checkpoints (recommended first):** define the state as `(node, phase)`, where `phase` is the number of ordered checkpoints already visited. Reaching the goal node before all checkpoints is not success. Visiting a checkpoint out of order does not advance phase, and revisiting the same node can produce a different full state. Choose cases by shortest path in the product graph, with 2--3 checkpoints and a target product-path length around 9--14 edges.
2. **Toggle gates (second):** define the state as `(node, gate_mask)`. Traversing designated existing edges toggles a bit; a gated edge is legal only under its current bit. The same node therefore has different outgoing actions in different gate states. This isolates action-conditioned transition prediction, but needs a separate formal config and solvability check.
3. **Partial observation (defer):** hide the phase or gate mask and reveal only node/edge feedback. This is useful only after the fully observable product-state task is understood; otherwise an information deficit is confounded with state maintenance.

For the first variant, the output contract should expose an auditable public state estimate without requiring hidden chain-of-thought, for example `phase_after` on each move and a final `{node, phase}`. The judge should report execution success, checkpoint-order success, product-graph shortestness, illegal moves, and per-step phase accuracy separately. Do not replace the official baseline config until this variant has been approved as a separate condition.
