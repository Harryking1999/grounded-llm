"""Run the explicitly bounded Sol pilot on frozen Qwen path cases."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
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


def sample(case, api, key):
    body = {
        "model": api["model"], "input": TASKS[case["condition"]].prompt(case),
        "reasoning": {"effort": api["reasoning_effort"]},
        "max_output_tokens": api["max_output_tokens"],
        "temperature": api["temperature"], "top_p": api["top_p"],
        "stream": api["stream"], "tools": [],
    }
    record = {"case_id": case["id"], "replicate": 1, "condition": case["condition"],
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
    record["elapsed_seconds"] = round(time.monotonic() - started, 3)
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    config = json.loads((ROOT / args.config).read_text(encoding="utf-8"))
    suite = json.loads((ROOT / config["suite"]).read_text(encoding="utf-8"))
    validate_suite(suite)
    if suite["config"] != json.loads((ROOT / config["qwen_contract"]).read_text(encoding="utf-8")):
        raise ValueError("Frozen suite does not match the Qwen contract")
    if config["replicates"] != 1 or config["api"]["automatic_retries"] != 0:
        raise ValueError("This pilot runs each selected case once, without retries")
    lookup = {case["id"]: case for case in suite["cases"]}
    cases = [lookup[case_id] for case_id in config["case_ids"]]
    if len(set(config["case_ids"])) != len(cases):
        raise ValueError("Duplicate pilot cases")
    key = os.environ.get("GND_API_KEY")
    if not key:
        raise ValueError("Set GND_API_KEY in the process environment")
    out = (ROOT / args.out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    payload = {"source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
               "config": config, "started_at": datetime.now(timezone.utc).isoformat(),
               "status": "running", "cases": []}
    write_json(out / "suite.json", suite)
    write_json(out / "run.json", payload)
    with ThreadPoolExecutor(max_workers=config["api"]["concurrency"]) as pool:
        futures = [pool.submit(sample, case, config["api"], key) for case in cases]
        for future in as_completed(futures):
            record = future.result()
            write_json(out / (record["case_id"] + ".json"), record)
            payload["cases"].append(record)
            write_json(out / "run.json", payload)
            print(json.dumps({"done": len(payload["cases"]), "total": len(cases),
                "case": record["case_id"], "pass": record["verdict"].get("pass"),
                "failure": record["verdict"].get("failure_type"), "tokens": record.get("output_tokens"),
                "seconds": record["elapsed_seconds"]}), flush=True)
    # Replay all completed answers with the unchanged task judge before reporting.
    for record in payload["cases"]:
        if record.get("response_status") == "completed":
            assert evaluate(lookup[record["case_id"]], record["raw_output"], TASKS) == record["verdict"]
    payload["status"] = "finished_with_service_errors" if any(r.get("api_error") for r in payload["cases"]) else "completed"
    payload["finished_at"] = datetime.now(timezone.utc).isoformat()
    write_json(out / "run.json", payload)
    print(json.dumps({"status": payload["status"], "run": str(out / "run.json")}), flush=True)


if __name__ == "__main__":
    main()
