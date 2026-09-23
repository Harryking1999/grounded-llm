import { mkdirSync, writeFileSync } from "node:fs";
import { dirname, resolve } from "node:path";

const apiKey = process.env.GND_API_KEY || process.env.OPENAI_API_KEY;
const baseUrl = (process.env.GND_BASE_URL || process.env.OPENAI_BASE_URL || "https://api.openai.com/v1").replace(/\/+$/, "");
const model = process.env.GND_MODEL || process.env.OPENAI_MODEL;
const apiStyle = (process.env.GND_API_STYLE || "responses").toLowerCase();
const reasoningEffort = process.env.GND_REASONING_EFFORT || "medium";
const outDir = resolve(process.env.GCML_RUN_DIR || "experiments/gcml_direct_api_blocks/runs/latest");

if (!apiKey) throw new Error("Set GND_API_KEY or OPENAI_API_KEY in the process environment.");
if (!model) throw new Error("Set GND_MODEL or OPENAI_MODEL in the process environment.");
if (!['responses', 'chat'].includes(apiStyle)) throw new Error("GND_API_STYLE must be responses or chat.");

const defaultGrids = [
  "0000111010/0000011110/0000000011/0000000111/0000001111/0000000100/0000011100/0000110000/0000000000/0000000000",
  "0000000000/0000000000/0000000000/0000000000/0000000000/0111100010/0000111010/0011101110/0000001111/0000000010",
  "0000000000/0000000000/1011000000/1111100000/1110111000/0011001000/0011011000/0001000000/0000000000/0000000000",
];
const grids = process.env.GCML_GRID_LIST ? JSON.parse(process.env.GCML_GRID_LIST) : defaultGrids;
const caseGrid = process.env.GCML_CASE_GRID ? JSON.parse(process.env.GCML_CASE_GRID) : [0, 1, 2, 0, 1, 2, 1, 2];
if (!Array.isArray(grids) || grids.length === 0 || grids.some((grid) => typeof grid !== "string")) {
  throw new Error("GCML_GRID_LIST must be a JSON array of grid strings.");
}
if (!Array.isArray(caseGrid) || caseGrid.length === 0 || caseGrid.some((index) => !Number.isInteger(index) || index < 0 || index >= grids.length)) {
  throw new Error("GCML_CASE_GRID must be a JSON array of valid grid indexes.");
}
const shapes = {
  0: [[0, 0], [0, 1], [1, 0]],
  1: [[0, 0], [1, 0], [1, 1]],
  2: [[0, 0], [0, 1], [1, 1]],
  3: [[0, 1], [1, 0], [1, 1]],
  4: [[0, 0], [1, 0]],
  5: [[0, 0], [0, 1]],
  6: [[0, 0], [1, 0], [2, 0]],
  7: [[0, 0], [0, 1], [0, 2]],
};

function promptFor(grid) {
  const rows = grid.split("/").join("\n");
  return `Task: Remove reusable shapes from the following 10x10 binary grid until every cell is 0.

Rules: 1 means occupied and 0 means empty. An action is (shape_id, row, col), with zero-based row and column and the shape's top-left anchor. An action is legal only when every 1-cell of the shape overlaps a current 1; a legal action sets those cells to 0. There is no gravity and no inventory limit. Shapes: 0=11/10, 1=10/11, 2=11/01, 3=01/11, 4=1/1, 5=11, 6=1/1/1, 7=111. Use at most 8 actions.

Initial grid rows:
${rows}

Output: return only valid JSON, with this schema: {"actions":[{"shape_id":0,"row":0,"col":0,"rationale":"short public reason"}],"final_status":"solved"}. Include one object per attempted action in order. The rationale is optional but, if present, must be a short auditable reason. Do not include markdown or hidden chain-of-thought.`;
}

function responseText(data) {
  if (typeof data.output_text === "string") return data.output_text;
  const output = (data.output || []).flatMap((item) => item.content || []);
  const responseParts = output.filter((part) => part.type === "output_text").map((part) => part.text || "");
  if (responseParts.length) return responseParts.join("");
  const content = data.choices?.[0]?.message?.content;
  if (typeof content === "string") return content;
  if (Array.isArray(content)) return content.map((part) => part.text || "").join("");
  return "";
}

function parseJson(raw) {
  try {
    return JSON.parse(raw);
  } catch {
    const start = raw.indexOf("{");
    const end = raw.lastIndexOf("}");
    if (start >= 0 && end > start) {
      try {
        return JSON.parse(raw.slice(start, end + 1));
      } catch {
        return null;
      }
    }
    return null;
  }
}

function judge(gridText, raw) {
  const parsed = parseJson(raw);
  const actions = parsed?.actions;
  const grid = gridText.split("/").map((row) => row.split("").map(Number));
  const errors = [];
  if (!Array.isArray(actions)) return { parseable: false, legal: false, solved: false, pass: false, actions: null, errors: ["missing actions array"] };
  if (actions.length > 8) errors.push(`action count ${actions.length} exceeds 8`);
  for (let i = 0; i < actions.length; i += 1) {
    const action = actions[i];
    const cells = shapes[action?.shape_id];
    const row = action?.row;
    const col = action?.col;
    if (!cells || !Number.isInteger(row) || !Number.isInteger(col)) {
      errors.push(`step ${i + 1}: malformed action`);
      break;
    }
    const coords = cells.map(([dr, dc]) => [row + dr, col + dc]);
    const bad = coords.filter(([r, c]) => r < 0 || r >= 10 || c < 0 || c >= 10 || grid[r][c] !== 1);
    if (bad.length) {
      errors.push(`step ${i + 1}: illegal overlap at ${bad.map(([r, c]) => `(${r},${c})`).join(", ")}`);
      break;
    }
    for (const [r, c] of coords) grid[r][c] = 0;
  }
  const solved = grid.every((row) => row.every((value) => value === 0));
  const legal = errors.every((error) => !error.includes("illegal overlap") && !error.includes("malformed action"));
  const pass = errors.length === 0 && solved && actions.length <= 8;
  return { parseable: true, legal, solved, pass, actions: actions.length, errors, final_grid: grid.map((row) => row.join("")) };
}

async function callModel(prompt) {
  const body = apiStyle === "responses"
    ? { model, input: prompt, reasoning: { effort: reasoningEffort }, max_output_tokens: 2000 }
    : { model, messages: [{ role: "user", content: prompt }], max_tokens: 2000 };
  const endpoint = `${baseUrl}/${apiStyle === "responses" ? "responses" : "chat/completions"}`;
  const response = await fetch(endpoint, {
    method: "POST",
    headers: { Authorization: `Bearer ${apiKey}`, "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const text = await response.text();
  let data;
  try { data = JSON.parse(text); } catch { data = { raw_http_body: text }; }
  if (!response.ok) throw new Error(`HTTP ${response.status}: ${JSON.stringify(data)}`);
  return { data, text: responseText(data) };
}

const startedAt = new Date().toISOString();
const results = await Promise.all(caseGrid.map((gridIndex, i) => {
  const grid = grids[gridIndex];
  const prompt = promptFor(grid);
  return callModel(prompt).then(({ data, text }) => ({
    case: i + 1,
    grid_index: gridIndex,
    input_grid: grid,
    prompt,
    raw_output: text,
    response: data,
    verdict: judge(grid, text),
  })).catch((error) => ({ case: i + 1, input_grid: grid, prompt, error: String(error), verdict: { pass: false } }));
}));

const passCount = results.filter((result) => result.verdict?.pass).length;
const summary = {
  started_at: startedAt,
  finished_at: new Date().toISOString(),
  model,
  base_url: baseUrl,
  api_style: apiStyle,
  reasoning_effort: reasoningEffort,
  sample_count: results.length,
  pass_count: passCount,
  cases: results,
  pass_at_n: `${passCount}/${results.length}`,
};
mkdirSync(dirname(`${outDir}/run.json`), { recursive: true });
writeFileSync(`${outDir}/run.json`, JSON.stringify(summary, null, 2), "utf8");
console.log(JSON.stringify({
  model,
  api_style: apiStyle,
  pass_at_n: summary.pass_at_n,
  cases: results.map((result) => ({ case: result.case, verdict: result.verdict, error: result.error || null })),
  output: `${outDir}/run.json`,
}, null, 2));
