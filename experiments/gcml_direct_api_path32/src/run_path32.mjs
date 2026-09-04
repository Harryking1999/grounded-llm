import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, resolve } from "node:path";

const apiKey = process.env.GND_API_KEY || process.env.OPENAI_API_KEY;
const baseUrl = (process.env.GND_BASE_URL || process.env.OPENAI_BASE_URL || "https://api.openai.com/v1").replace(/\/+$/, "");
const model = process.env.GND_MODEL || process.env.OPENAI_MODEL;
const apiStyle = (process.env.GND_API_STYLE || "responses").toLowerCase();
const reasoningEffort = process.env.GND_REASONING_EFFORT || "medium";
const configPath = resolve(process.env.GCML_PATH32_CONFIG || "experiments/gcml_direct_api_path32/configs/path32_v1.json");
const outDir = resolve(process.env.GCML_RUN_DIR || "runs/gcml_direct_api_path32/latest");

if (!apiKey) throw new Error("Set GND_API_KEY or OPENAI_API_KEY in the process environment.");
if (!model) throw new Error("Set GND_MODEL or OPENAI_MODEL in the process environment.");
if (!["responses", "chat"].includes(apiStyle)) throw new Error("GND_API_STYLE must be responses or chat.");

const config = JSON.parse(readFileSync(configPath, "utf8"));
const neighbors = Object.fromEntries(Object.entries(config.neighbors).map(([node, adjacent]) => [Number(node), adjacent]));
const nodes = Object.keys(neighbors).map(Number).sort((a, b) => a - b);
if (nodes.length !== 32 || nodes.some((node, index) => node !== index)) throw new Error("Expected neighbor lists for nodes 0 through 31.");
for (const [node, adjacent] of Object.entries(neighbors)) {
  for (const next of adjacent) {
    if (!neighbors[next]?.includes(Number(node))) throw new Error(`Graph edge ${node}-${next} is not symmetric.`);
  }
}
if (!Array.isArray(config.pairs) || config.pairs.length === 0 || !Number.isInteger(config.replicates) || config.replicates < 1) {
  throw new Error("Config must define non-empty pairs and a positive replicate count.");
}

function shortestDistance(start, goal) {
  const distance = new Map([[start, 0]]);
  const queue = [start];
  for (let head = 0; head < queue.length; head += 1) {
    const node = queue[head];
    if (node === goal) return distance.get(node);
    for (const next of neighbors[node]) {
      if (!distance.has(next)) {
        distance.set(next, distance.get(node) + 1);
        queue.push(next);
      }
    }
  }
  return null;
}

function promptFor(start, goal) {
  const adjacency = nodes.map((node) => `${node}: ${neighbors[node].join(", ")}`).join("\n");
  return `Task: Find a path in the official 32-node undirected graph from start node ${start} to goal node ${goal}.

Rules: Nodes are numbered 0 through 31. Each move must follow one listed graph edge. There is no step limit. Plan the complete path before answering. The external judge will execute the moves in order from the stated start node and will stop at the first illegal move.

Graph neighbors:
${adjacency}

Output: return only valid JSON with this schema: {"path":[{"from":${start},"to":0,"rationale":"short public reason"}],"final_node":${goal}}. Use one object per attempted move in order. The rationale is optional but, if present, must be a short auditable reason. Do not include markdown.`;
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
      try { return JSON.parse(raw.slice(start, end + 1)); } catch { return null; }
    }
    return null;
  }
}

function judge(start, goal, raw) {
  const parsed = parseJson(raw);
  const path = parsed?.path;
  if (!Array.isArray(path)) {
    return { parseable: false, execution_pass: false, optimal: false, reported_goal_correct: false, pass: false, moves: null, errors: ["missing path array"] };
  }
  let current = start;
  const errors = [];
  for (let index = 0; index < path.length; index += 1) {
    const step = path[index];
    if (!Number.isInteger(step?.from) || !Number.isInteger(step?.to)) {
      errors.push(`step ${index + 1}: malformed move`);
      break;
    }
    if (step.from !== current) {
      errors.push(`step ${index + 1}: from ${step.from} does not match current node ${current}`);
      break;
    }
    if (!neighbors[current].includes(step.to)) {
      errors.push(`step ${index + 1}: illegal edge ${current}->${step.to}`);
      break;
    }
    current = step.to;
  }
  const executionPass = errors.length === 0 && current === goal;
  const optimal = executionPass && path.length === shortestDistance(start, goal);
  const reportedGoalCorrect = parsed.final_node === goal;
  return {
    parseable: true,
    execution_pass: executionPass,
    optimal,
    reported_goal_correct: reportedGoalCorrect,
    pass: executionPass,
    moves: path.length,
    final_node_actual: current,
    final_node_reported: parsed.final_node ?? null,
    shortest_moves: shortestDistance(start, goal),
    errors,
  };
}

async function callModel(prompt) {
  const body = apiStyle === "responses"
    ? { model, input: prompt, reasoning: { effort: reasoningEffort }, max_output_tokens: 3000 }
    : { model, messages: [{ role: "user", content: prompt }], max_tokens: 3000 };
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

const cases = Array.from({ length: config.replicates }, (_, replicate) => config.pairs.map((pair, pairIndex) => ({
  ...pair,
  pair_index: pairIndex,
  replicate: replicate + 1,
}))).flat();
const startedAt = new Date().toISOString();
const results = await Promise.all(cases.map((testCase, index) => {
  const prompt = promptFor(testCase.start, testCase.goal);
  return callModel(prompt).then(({ data, text }) => ({
    case: index + 1,
    replicate: testCase.replicate,
    pair_index: testCase.pair_index,
    start: testCase.start,
    goal: testCase.goal,
    prompt,
    raw_output: text,
    response: data,
    verdict: judge(testCase.start, testCase.goal, text),
  })).catch((error) => ({
    case: index + 1,
    replicate: testCase.replicate,
    pair_index: testCase.pair_index,
    start: testCase.start,
    goal: testCase.goal,
    prompt,
    error: String(error),
    verdict: { parseable: false, execution_pass: false, optimal: false, reported_goal_correct: false, pass: false },
  }));
}));

const passCount = results.filter((result) => result.verdict?.pass).length;
const optimalCount = results.filter((result) => result.verdict?.optimal).length;
const reportedGoalCount = results.filter((result) => result.verdict?.reported_goal_correct).length;
const summary = {
  started_at: startedAt,
  finished_at: new Date().toISOString(),
  model,
  base_url: baseUrl,
  api_style: apiStyle,
  reasoning_effort: reasoningEffort,
  config_path: configPath,
  source_commit: config.source_commit,
  sample_count: results.length,
  pass_count: passCount,
  optimal_count: optimalCount,
  reported_goal_correct_count: reportedGoalCount,
  pass_at_n: `${passCount}/${results.length}`,
  optimal_rate: `${optimalCount}/${results.length}`,
  cases: results,
};
mkdirSync(dirname(`${outDir}/run.json`), { recursive: true });
writeFileSync(`${outDir}/run.json`, JSON.stringify(summary, null, 2), "utf8");
console.log(JSON.stringify({
  model,
  api_style: apiStyle,
  pass_at_n: summary.pass_at_n,
  optimal_rate: summary.optimal_rate,
  output: `${outDir}/run.json`,
}, null, 2));
