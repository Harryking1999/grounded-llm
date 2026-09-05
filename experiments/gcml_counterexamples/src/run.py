"""Direct text API evaluation. Exact solvers run only after model output arrives."""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime, timezone
import json
import itertools
import os
from pathlib import Path
import random
import statistics
import subprocess
import time
import urllib.error
import urllib.request

import blocks
import path as path_task

ROOT = Path(__file__).resolve().parents[3]


def prompt_for(case):
    if case["condition"].startswith("blocks"):
        budget_text = (f"Use at most {case['budget']} actions. A solution within this budget exists."
                       if case["budget"] is not None else
                       "There is no action-count limit. A complete decomposition exists. You do not need to minimize the number of actions.")
        return f"""Task: Remove reusable shapes from this 10x10 binary grid until every cell is 0.

Rules: 1 means occupied and 0 means empty. An action is (shape_id, row, col), with zero-based row and column and the shape's top-left bounding-box anchor. An action is legal only when every 1-cell of the shape overlaps a current 1; a legal action sets those cells to 0. There is no gravity and no inventory limit. Shapes: 0=11/10, 1=10/11, 2=11/01, 3=01/11, 4=1/1, 5=11, 6=1/1/1, 7=111. {budget_text} Any valid decomposition is accepted; you do not need to recover a particular reference decomposition.

Initial grid rows, from row 0 through row 9 (each character is column 0 through column 9):
{chr(10).join(case['grid'].split('/'))}

Plan the complete sequence before answering. Check that the sequence clears the entire grid without an illegal move. The external judge executes the submitted sequence and stops at the first illegal action; there is no intermediate feedback.

Output only valid JSON with an actions array of objects containing integer shape_id, row, and col, and final_status set to "solved" or "unsolved" according to whether your sequence clears the grid. A short public rationale on an action is optional. Do not include markdown or a reasoning transcript."""
    adjacency = "\n".join(f"{node}: {', '.join(map(str, case['neighbors'][str(node)]))}"
                          for node in range(32))
    rules = "Each move must follow one listed undirected edge. Every move has cost 1."
    state_contract = ""
    if case.get("bits"):
        switches = "; ".join(f"entering node {s['node']} flips switch {s['bit']}" for s in case["switches"])
        masks = [(case["initial_mask"] >> bit) & 1 for bit in range(case["bits"])]
        gates = "\n".join(f"{a}-{b}: requires switch {g['bit']} = 1"
                          for g in case["gates"] for a, b in [g["edge"]])
        rules += f"""
There are {case['bits']} binary switches, indexed 0 through {case['bits'] - 1}; the initial values in that order are {json.dumps(masks)}. A flip changes 0 to 1 or 1 to 0. Switch locations: {switches}.
Before each move, check its gate requirements against the CURRENT switch values. If permitted, move to the destination, THEN flip any switch at the destination. A switch flips on EVERY entry, including revisits. Merely starting at a switch does not flip it. Gates apply in both directions. Edges not listed below have no gate requirements. Reaching node {case['goal']} is the goal; final switch values do not matter. Revisiting a node is allowed and may be necessary because its switch state can differ.
Gated edges:
{gates}"""
        state_contract = " Include switches_after on every move as an array of binary values in switch-index order after arrival and any flip. This report is checked separately from path execution."
    return f"""Task: Find a SHORTEST valid path from node {case['start']} to node {case['goal']} in this 32-node graph. Minimize the total number of edges traversed. A valid path exists.

Rules: Nodes are numbered 0 through 31. {rules}
There is no additional step limit. Any shortest path is accepted.

Graph neighbors:
{adjacency}

Plan the complete path before answering and verify edge legality and shortestness. The external judge executes moves from the stated start and stops at the first illegal move; there is no intermediate feedback.

Output only valid JSON with a path array of objects containing integer from and to fields, and a final_node integer. Use one object per move.{state_contract} A short public rationale is optional. Do not include markdown or a reasoning transcript."""


def extract_text(response):
    if isinstance(response.get("output_text"), str):
        return response["output_text"]
    # Some compatible endpoints return commentary and final messages together.
    # Prefer the explicit final message; retain the entire response in the log.
    messages = [item for item in response.get("output", []) if item.get("type") == "message"]
    finals = [item for item in messages if item.get("channel") == "final"]
    selected = finals or messages
    return "".join(part.get("text", "") for item in selected
                   for part in item.get("content", []) if part.get("type") == "output_text")


def parse_output(raw):
    try:
        return json.loads(raw), True
    except (ValueError, TypeError):
        try:
            return json.loads(raw[raw.index("{"):raw.rindex("}") + 1]), False
        except (ValueError, TypeError):
            return None, False


def read_response(response, streaming):
    if not streaming:
        return json.loads(response.read().decode())
    last_response = None
    event_types = Counter()
    # Responses SSE uses one JSON data line per event. Only complete response
    # objects are used for verdicts; token deltas are never treated as a final answer.
    for line in response:
        line = line.decode("utf-8").strip()
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if payload == "[DONE]":
            break
        event = json.loads(payload)
        event_types[event.get("type", "unknown")] += 1
        if isinstance(event.get("response"), dict):
            last_response = event["response"]
        if event.get("type") in ("response.completed", "response.incomplete", "response.failed"):
            return last_response, dict(event_types)
    if last_response is None:
        raise ValueError("Stream ended without a response object")
    return last_response, dict(event_types)


def evaluate(case, raw):
    output, strict = parse_output(raw)
    if not isinstance(output, dict):
        return {"pass": False, "parseable": False, "strict_json": strict, "failure_type": "parse_error"}
    if case["condition"].startswith("blocks"):
        verdict = blocks.judge(case["grid"], output.get("actions"), case["budget"])
        verdict["reported_status"] = output.get("final_status")
        verdict["reported_status_correct"] = output.get("final_status") == ("solved" if verdict.get("solved") else "unsolved")
    else:
        verdict = path_task.judge(case, output)
    return {**verdict, "parseable": True, "strict_json": strict}


def aggregate(records, cases):
    groups = defaultdict(list)
    by_case = defaultdict(list)
    for record in records:
        groups[record["condition"]].append(record)
        by_case[record["case_id"]].append(record)
    case_lookup = {case["id"]: case for case in cases}
    summaries = {}
    for condition, rows in groups.items():
        case_ids = sorted({r["case_id"] for r in rows})
        complete = [key for key in case_ids if len(by_case[key]) == case_lookup[key]["replicates"]]
        valid8 = [key for key in complete if all(not r.get("api_error") and r.get("response_status") == "completed" for r in by_case[key])]
        passes = sum(r["verdict"].get("pass", False) for r in rows)
        eligible = [r for r in rows if not r.get("api_error") and r.get("response_status") == "completed"]
        failures = Counter(r["verdict"].get("failure_type") or "success" for r in rows)
        summary = {"calls_recorded": len(rows), "completed_responses": len(eligible),
                   "successes": passes, "sample_success_rate_all_attempts": passes / len(rows),
                   "sample_success_rate_completed_responses": sum(r["verdict"].get("pass", False) for r in eligible) / len(eligible) if eligible else None,
                   "complete_case_groups": len(complete), "valid_pass8_case_groups": len(valid8),
                   "pass_at_8": sum(any(r["verdict"].get("pass") for r in by_case[key]) for key in valid8) / len(valid8) if valid8 else None,
                   "solved_case_groups": sum(any(r["verdict"].get("pass") for r in by_case[key]) for key in valid8),
                   "failure_counts": dict(failures),
                   "per_case": {key: {"n": len(by_case[key]), "successes": sum(r["verdict"].get("pass", False) for r in by_case[key])} for key in case_ids}}
        if condition.startswith("path"):
            summary["execution_successes"] = sum(r["verdict"].get("execution_pass", False) for r in rows)
            summary["execution_pass_at_8"] = sum(any(r["verdict"].get("execution_pass") for r in by_case[key]) for key in valid8) / len(valid8) if valid8 else None
        for name in ("output_tokens", "reasoning_tokens", "elapsed_seconds"):
            values = [r[name] for r in eligible if isinstance(r.get(name), (int, float))]
            summary[name] = {"sum": sum(values), "median": statistics.median(values), "max": max(values)} if values else None
        summaries[condition] = summary
    return summaries


def continuation_samples(previous, cases, config):
    """Retain every completed answer, including failures, after a transport stop."""
    for field in ("model", "base_url", "reasoning_effort", "max_output_tokens", "style"):
        if previous["api_config"][field] != config[field]:
            raise ValueError(f"Continuation changes API field {field}")
    lookup = {case["id"]: case for case in cases}
    completed, transport_failures, seen = [], list(previous.get("prior_failed_attempts", [])), set()
    for record in previous["cases"]:
        case = lookup[record["case_id"]]
        key = record["case_id"], record["replicate"]
        if key in seen or not 1 <= record["replicate"] <= case["replicates"]:
            raise ValueError(f"Invalid prior sample {key}")
        seen.add(key)
        if record["request"]["input"] != prompt_for(case):
            raise ValueError(f"Continuation changes prompt for {case['id']}")
        if not record.get("api_error") and record.get("response_status") == "completed":
            completed.append(record)
        else:
            error_code = (record.get("response", {}).get("error") or {}).get("code")
            if not record.get("api_error") and error_code not in ("upstream_error", "server_error"):
                raise ValueError(f"Prior sample {key} is not a confirmed service failure; do not resample budget-limited or unexplained incomplete answers")
            transport_failures.append(record)
    return completed, transport_failures


def call(case, replicate, config, key):
    prompt = prompt_for(case)
    body = {"model": config["model"], "input": prompt,
            "reasoning": {"effort": config["reasoning_effort"]},
            "max_output_tokens": config["max_output_tokens"], "tools": []}
    if config.get("stream"):
        body["stream"] = True
    start = time.monotonic()
    record = {"case_id": case["id"], "condition": case["condition"], "replicate": replicate,
              "request": body, "started_at": datetime.now(timezone.utc).isoformat()}
    req = urllib.request.Request(config["base_url"].rstrip("/") + "/responses",
                                 data=json.dumps(body).encode(), method="POST",
                                 headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json",
                                          "User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=config["timeout_seconds"]) as response:
            if config.get("stream"):
                data, record["stream_event_counts"] = read_response(response, True)
            else:
                data = read_response(response, False)
        raw = extract_text(data)
        usage = data.get("usage", {})
        record.update({"response": data, "response_status": data.get("status"),
                       "raw_output": raw, "output_tokens": usage.get("output_tokens"),
                       "reasoning_tokens": usage.get("output_tokens_details", {}).get("reasoning_tokens"),
                       "input_tokens": usage.get("input_tokens")})
        record["verdict"] = evaluate(case, raw)
        if data.get("status") != "completed":
            record["verdict"] = {"pass": False, "failure_type": "incomplete_response", "partial_verdict": record["verdict"]}
    except urllib.error.HTTPError as error:
        record["api_error"] = f"HTTP {error.code}"
        record["api_error_detail"] = error.read().decode(errors="replace").replace(key, "[redacted]")[:2000]
        record["verdict"] = {"pass": False, "failure_type": "api_error"}
    except (urllib.error.URLError, TimeoutError, ValueError) as error:
        record["api_error"] = type(error).__name__
        record["api_error_detail"] = str(error).replace(key, "[redacted]")[:2000]
        record["verdict"] = {"pass": False, "failure_type": "api_error"}
    record["elapsed_seconds"] = round(time.monotonic() - start, 3)
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", default="experiments/gcml_counterexamples/runs/uncapped/suite.json")
    parser.add_argument("--api-config", help="Committed comparison config; overrides only API settings, preserving the suite")
    parser.add_argument("--out", required=True)
    parser.add_argument("--condition", action="append")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--continue-from", help="Prior stopped run with identical prompts; retain all completed answers")
    args = parser.parse_args()
    suite = json.loads((ROOT / args.suite).read_text())
    config = suite["config"]["api"]
    comparison = None
    if args.api_config:
        comparison = json.loads((ROOT / args.api_config).read_text())
        if (ROOT / comparison["suite"]).resolve() != (ROOT / args.suite).resolve():
            raise ValueError("Comparison config must refer to the selected suite")
        config = {**config, **comparison["api"]}
    key = os.environ.get("GND_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not key:
        raise ValueError("Set GND_API_KEY in the process environment")
    # Overrides support the same experiment via another explicitly selected endpoint.
    config = {**config, "base_url": os.environ.get("GND_BASE_URL", config["base_url"]),
              "model": os.environ.get("GND_MODEL", config["model"])}
    if config["style"] != "responses":
        raise ValueError("This study uses the Responses API contract")
    cases = [c for c in suite["cases"] if not args.condition or c["condition"] in args.condition]
    previous_records, prior_failed_attempts = [], []
    if args.continue_from:
        previous = json.loads((ROOT / args.continue_from).read_text())
        if previous["status"] != "stopped_api_error":
            raise ValueError("Only a stopped API run can be continued")
        previous_records, prior_failed_attempts = continuation_samples(previous, cases, config)
    retained = {(r["case_id"], r["replicate"]) for r in previous_records}
    jobs = [(c, r) for c in cases for r in range(1, c["replicates"] + 1)]
    random.Random(suite["config"]["seed"] + 30000).shuffle(jobs)
    jobs = [(case, replicate) for case, replicate in jobs if (case["id"], replicate) not in retained]
    if args.limit is not None:
        jobs = jobs[:args.limit]
    if not jobs or len(jobs) > config["max_calls"]:
        raise ValueError("Invalid call count")
    output = ROOT / args.out
    output.mkdir(parents=True, exist_ok=True)
    run_path = output / "run.json"
    if run_path.exists():
        raise FileExistsError("Choose a new run directory; existing results are never overwritten")
    payload = {"source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
               "suite_path": args.suite, "api_config": config, "planned_calls": len(jobs),
               "api_config_path": args.api_config, "comparison_config": comparison,
               "continued_from": args.continue_from, "prior_failed_attempts": prior_failed_attempts,
               "retained_completed_responses": len(previous_records),
               "started_at": datetime.now(timezone.utc).isoformat(), "status": "running", "cases": previous_records}

    def save():
        payload["summary"] = aggregate(payload["cases"], cases)
        temporary = run_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        temporary.replace(run_path)

    save()
    def accept(record):
        payload["cases"].append(record)
        save()
        print(json.dumps({"done": len(payload["cases"]), "total": len(jobs) + len(retained),
                          "case": record["case_id"], "replicate": record["replicate"],
                          "pass": record["verdict"].get("pass"),
                          "failure": record["verdict"].get("failure_type"),
                          "seconds": record["elapsed_seconds"]}), flush=True)

    # The gateway rejected urllib's default user agent on the first actual run.
    # Validate one real scheduled sample before launching concurrent calls, and
    # keep only a bounded number in flight so service failures stop new requests.
    first_case, first_replicate = jobs[0]
    first = call(first_case, first_replicate, config, key)
    accept(first)
    stopped = bool(first.get("api_error") or first.get("response_status") != "completed")
    remaining = iter(jobs[1:])
    with ThreadPoolExecutor(max_workers=config["concurrency"]) as executor:
        pending = set()
        if not stopped:
            for case, replicate in itertools.islice(remaining, config["concurrency"]):
                pending.add(executor.submit(call, case, replicate, config, key))
        while pending:
            done, pending = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                record = future.result()
                accept(record)
                if record.get("api_error") or record.get("response_status") != "completed":
                    stopped = True
            if not stopped:
                for case, replicate in itertools.islice(remaining, config["concurrency"] - len(pending)):
                    pending.add(executor.submit(call, case, replicate, config, key))
    payload["status"] = "stopped_api_error" if stopped else "completed"
    payload["finished_at"] = datetime.now(timezone.utc).isoformat()
    save()
    print(json.dumps({"output": str(run_path), "summary": payload["summary"]}), flush=True)


if __name__ == "__main__":
    main()
