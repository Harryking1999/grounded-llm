"""Run the explicitly bounded Sol pilot on frozen Qwen path cases."""
import argparse
from collections import Counter, deque
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import urllib.error
import urllib.request

from content_eval import evaluate
from prepare_path_suite import validate_suite

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "experiments/sol_dag_blocks/src"))
from tasks import TASKS

LEGACY = ROOT / "experiments/gcml_counterexamples/src"
sys.path.append(str(LEGACY))
spec = importlib.util.spec_from_file_location("responses_transport", LEGACY / "run.py")
transport = importlib.util.module_from_spec(spec)
spec.loader.exec_module(transport)


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def request_body(case, api):
    return {
        "model": api["model"], "input": TASKS[case["condition"]].prompt(case),
        "reasoning": {"effort": api["reasoning_effort"]},
        "max_output_tokens": api["max_output_tokens"],
        "temperature": api["temperature"], "top_p": api["top_p"],
        "stream": api["stream"], "tools": [],
    }


def sample(case, api, key, replicate=1):
    body = request_body(case, api)
    record = {"case_id": case["id"], "replicate": replicate, "condition": case["condition"],
              "request": body, "started_at": datetime.now(timezone.utc).isoformat()}
    request = urllib.request.Request(api["base_url"].rstrip("/") + "/responses",
        data=json.dumps(body).encode(), headers={"Authorization": "Bearer " + key,
        "Content-Type": "application/json", "User-Agent": "Mozilla/5.0"})
    started = time.monotonic()
    try:
        with urllib.request.urlopen(request, timeout=api["timeout_seconds"]) as response:
            data, events = transport.read_response(response, True)
        raw = transport.extract_text(data)
        usage = data.get("usage") or {}
        tokens = usage.get("output_tokens")
        verdict = evaluate(case, raw, TASKS)
        record.update(response=data, response_status=data.get("status"),
            stream_event_counts=events, raw_output=raw, input_tokens=usage.get("input_tokens"),
            output_tokens=tokens,
            reasoning_tokens=(usage.get("output_tokens_details") or {}).get("reasoning_tokens"),
            exceeds_requested_budget=tokens > api["max_output_tokens"] if isinstance(tokens, int) else None)
        if data.get("status") != "completed":
            verdict = {"pass": False, "failure_type": "incomplete_response", "partial_verdict": verdict}
        record["verdict"] = verdict
    except (urllib.error.URLError, TimeoutError, ConnectionError, ValueError) as error:
        detail = error.read().decode(errors="replace") if isinstance(error, urllib.error.HTTPError) else str(error)
        record.update(api_error=type(error).__name__, api_error_detail=detail.replace(key, "[redacted]")[:2000],
                      verdict={"pass": False, "failure_type": "api_error"})
        if isinstance(error, urllib.error.HTTPError):
            record["http_status"] = error.code
    record["elapsed_seconds"] = round(time.monotonic() - started, 3)
    return record


def retryable(record):
    """Only identified infrastructure failures may be sampled again."""
    if transport.final_sample(record):
        return False
    status = record.get("http_status", 0)
    if status == 429 or 500 <= status < 600:
        return True
    error = (record.get("response") or {}).get("error") or {}
    if record.get("api_error") == "HTTPError":
        try:
            error = json.loads(record.get("api_error_detail", ""))["error"]
        except (ValueError, KeyError, TypeError):
            return False
    if error.get("code") in {"jarodfund_error", "server_error", "upstream_error", "gateway_concurrency_limit"}:
        return True
    return record.get("api_error") in {"TimeoutError", "ConnectionResetError", "ConnectionAbortedError"}


def retain_previous(paths, lookup, api, slots):
    retained, archived, seen = [], [], set()
    for name in paths:
        previous = json.loads((ROOT / name).read_text(encoding="utf-8"))
        if previous["config"]["api"]["base_url"] != api["base_url"]:
            raise ValueError("Continuation changes the gateway")
        archived.extend(previous.get("prior_failed_attempts", []))
        for record in previous["cases"]:
            slot = record["case_id"], record["replicate"]
            if slot not in slots or record["request"] != request_body(lookup[slot[0]], api):
                raise ValueError("Continuation changes a trial or its request")
            if transport.final_sample(record):
                if slot in seen:
                    raise ValueError("Duplicate saved final trial")
                seen.add(slot)
                retained.append(record)
            elif retryable(record):
                archived.append(record)
            else:
                raise ValueError("Unexplained incomplete answer cannot be resampled")
    return retained, archived


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--continue-from", nargs="*", default=[])
    args = parser.parse_args()
    config = json.loads((ROOT / args.config).read_text(encoding="utf-8"))
    suite = json.loads((ROOT / config["suite"]).read_text(encoding="utf-8"))
    validate_suite(suite)
    if suite["config"] != json.loads((ROOT / config["qwen_contract"]).read_text(encoding="utf-8")):
        raise ValueError("Frozen suite does not match the Qwen contract")
    lookup = {case["id"]: case for case in suite["cases"]}
    cases = [lookup[case_id] for case_id in config["case_ids"]]
    if len(set(config["case_ids"])) != len(cases):
        raise ValueError("Duplicate pilot cases")
    if not 1 <= config["replicates"] <= min(case["replicates"] for case in cases):
        raise ValueError("Pilot replicates must fit the frozen Qwen suite")
    jobs = [(case, n) for n in range(1, config["replicates"] + 1) for case in cases]
    target = config.get("target_samples", len(jobs))
    if not 1 <= target <= len(jobs):
        raise ValueError("Invalid final-sample target")
    jobs = jobs[:target]
    retained, archived = retain_previous(args.continue_from, lookup, config["api"],
        {(case["id"], n) for case, n in jobs})
    done = {(r["case_id"], r["replicate"]) for r in retained}
    remaining = deque((case, n) for case, n in jobs if (case["id"], n) not in done)
    key = os.environ.get("GND_API_KEY")
    if not key:
        raise ValueError("Set GND_API_KEY in the process environment")
    out = (ROOT / args.out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    payload = {"source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
               "config": config, "started_at": datetime.now(timezone.utc).isoformat(),
               "status": "running", "target_samples": target, "cases": retained,
               "continued_from": args.continue_from, "prior_failed_attempts": archived,
               "retained_final_samples": len(retained), "inflight_slots": []}
    write_json(out / "suite.json", suite)
    write_json(out / "run.json", payload)
    counts = Counter((r["case_id"], r["replicate"]) for r in archived)
    maximum_errors = config["api"].get("maximum_service_failures", 0)
    error_count, attempts, stopped = 0, 0, False
    with ThreadPoolExecutor(max_workers=config["api"]["concurrency"]) as pool:
        pending = set()
        while remaining or pending:
            while remaining and not stopped and len(pending) < config["api"]["concurrency"]:
                case, replicate = remaining.popleft()
                payload["inflight_slots"].append([case["id"], replicate])
                write_json(out / "run.json", payload)
                pending.add(pool.submit(sample, case, config["api"], key, replicate))
            if not pending:
                break
            completed, pending = wait(pending, return_when=FIRST_COMPLETED)
            for future in completed:
                record = future.result()
                attempts += 1
                slot = record["case_id"], record["replicate"]
                payload["inflight_slots"].remove(list(slot))
                write_json(out / f"attempt_{attempts:03d}.json", record)
                if transport.final_sample(record):
                    payload["cases"].append(record)
                else:
                    counts[slot] += 1
                    error_count += 1
                    can_retry = (retryable(record) and counts[slot] <= config["api"]["automatic_retries"]
                                 and error_count < maximum_errors)
                    if can_retry:
                        payload["prior_failed_attempts"].append(record)
                        remaining.append((lookup[slot[0]], slot[1]))
                    else:
                        payload["cases"].append(record)
                        if not retryable(record) or error_count >= maximum_errors:
                            stopped = True
                write_json(out / "run.json", payload)
                print(json.dumps({"final_samples": sum(transport.final_sample(r) for r in payload["cases"]),
                    "target": target, "case": slot[0], "replicate": slot[1],
                    "pass": record["verdict"].get("pass"), "failure": record["verdict"].get("failure_type"),
                    "tokens": record.get("output_tokens"), "seconds": record["elapsed_seconds"]}), flush=True)
    # Replay all completed answers with the unchanged task judge before reporting.
    for record in payload["cases"]:
        if record.get("response_status") == "completed":
            assert evaluate(lookup[record["case_id"]], record["raw_output"], TASKS) == record["verdict"]
    final_count = sum(transport.final_sample(r) for r in payload["cases"])
    payload["status"] = "completed" if final_count == target else "finished_with_service_errors"
    payload["finished_at"] = datetime.now(timezone.utc).isoformat()
    write_json(out / "run.json", payload)
    print(json.dumps({"status": payload["status"], "run": str(out / "run.json")}), flush=True)


if __name__ == "__main__":
    main()
