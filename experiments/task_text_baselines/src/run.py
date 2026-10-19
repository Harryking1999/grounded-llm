"""Bounded, resumable one-response API evaluation; raw outputs remain outside Git."""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import random
import subprocess
import time
import urllib.error
import urllib.request

from .protocol import ROOT, judge
from .budget import output_limit, rescore


def write_json(path, value):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def read_response(response):
    if "text/event-stream" not in response.headers.get("Content-Type", ""):
        return json.loads(response.read().decode())
    data = None
    for line in response:
        line = line.decode("utf-8").strip()
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if payload == "[DONE]":
            break
        event = json.loads(payload)
        if isinstance(event.get("response"), dict):
            data = event["response"]
        if event.get("type") in ("response.completed", "response.failed", "response.incomplete"):
            break
    if data is None:
        raise ValueError("Stream ended without response")
    return data


def extract_text(data):
    if isinstance(data.get("output_text"), str):
        return data["output_text"]
    messages = [item for item in data.get("output", []) if item.get("type") == "message"]
    finals = [item for item in messages if item.get("channel") == "final"]
    return "".join(part.get("text", "") for item in finals or messages for part in item.get("content", []) if part.get("type") == "output_text")


def call(case, api, key, attempt):
    if attempt > 1:
        time.sleep(api.get('service_retry_delay_seconds',0))
    body = {"model": api["model"], "input": case["prompt"], "reasoning": {"effort": api["reasoning_effort"]},
            "max_output_tokens": api["max_output_tokens"], "tools": [], "stream": True, "store": False}
    record = {"case_id": case["id"], "task": case["task"], "stratum": case["stratum"], "attempt": attempt,
              "requested_max_output_tokens": api["max_output_tokens"],
              "started_at": datetime.now(timezone.utc).isoformat()}
    start = time.monotonic()
    request = urllib.request.Request(api["base_url"].rstrip("/") + "/responses", data=json.dumps(body).encode(),
                                     headers={"Authorization": "Bearer " + key, "Content-Type": "application/json", "User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(request, timeout=api["timeout_seconds"]) as response:
            data = read_response(response)
        raw = extract_text(data)
        usage = data.get("usage") or {}
        tokens = usage.get("output_tokens")
        record.update(response=data, raw_output=raw, usage=usage, response_status=data.get("status"),
                      returned_model=data.get("model"), returned_reasoning=data.get("reasoning"),
                      exceeds_requested_budget=tokens > api["max_output_tokens"] if isinstance(tokens, int) else None,
                      verdict=judge(case, raw))
        if data.get("status") != "completed":
            reason = (data.get("incomplete_details") or {}).get("reason")
            if data.get("status") == "incomplete" and reason == "max_output_tokens":
                record["verdict"] = {"solved": False, "contract_pass": False, "failure": "truncated"}
            else:
                record["service_error"] = "nonterminal_or_failed_response"
        elif data.get("model") and not str(data["model"]).startswith(api["model"]):
            record["service_error"] = "returned_model_mismatch"
        if any(item.get("type") not in ("reasoning", "message") for item in data.get("output", [])):
            record["service_error"] = "unexpected_output_item"
    except (urllib.error.URLError, TimeoutError, ConnectionError, ValueError, OSError) as error:
        detail = error.read().decode(errors="replace") if isinstance(error, urllib.error.HTTPError) else str(error)
        record.update(service_error=type(error).__name__, error_detail=detail.replace(key, "[redacted]")[:1000])
        if isinstance(error, urllib.error.HTTPError):
            record["http_status"] = error.code
    if record.get("service_error"):
        record["verdict"] = {"solved": False, "contract_pass": False, "failure": "service_error"}
    record["elapsed_seconds"] = round(time.monotonic() - start, 3)
    if api.get('scoring_output_limit') is not None:
        record = rescore(record,api['scoring_output_limit'])
    return record


def summary(records, cases):
    result = {}
    for task in dict.fromkeys(c["task"] for c in cases):
        rows = [r for r in records if r["task"] == task]
        strata = defaultdict(list)
        for row in rows:
            strata[row["stratum"]].append(row)
        result[task] = {"planned": sum(c["task"] == task for c in cases), "finished": len(rows),
                        "valid_responses": sum(not bool(r.get("service_error")) for r in rows),
                        "service_errors": sum(bool(r.get("service_error")) for r in rows),
                        "solved": sum(bool(r["verdict"].get("solved")) for r in rows),
                        "contract_pass": sum(bool(r["verdict"].get("contract_pass")) for r in rows),
                        "optimal": sum(bool(r["verdict"].get("optimal")) for r in rows),
                        "optimal_evaluated": sum(r["verdict"].get("optimal") is not None for r in rows),
                        "failures": dict(Counter(r["verdict"].get("failure") or "success" for r in rows)),
                        "output_tokens": sum((r.get("usage") or {}).get("output_tokens") or 0 for r in rows),
                        "input_tokens": sum((r.get("usage") or {}).get("input_tokens") or 0 for r in rows),
                        "strata": {name: {"finished": len(values),
                                          "valid_responses": sum(not bool(r.get("service_error")) for r in values),
                                          "service_errors": sum(bool(r.get("service_error")) for r in values),
                                          "solved": sum(bool(r["verdict"].get("solved")) for r in values)}
                                   for name, values in strata.items()}}
    return result


def pending(cases, latest, api, seed):
    rows = [c for c in cases if c["id"] not in latest or
            latest[c["id"]].get("service_error") and latest[c["id"]]["attempt"] <= api["max_service_retries"]]
    random.Random(seed).shuffle(rows)
    # Dispatch uses pop(): failed infrastructure attempts come first.
    rows.sort(key=lambda c: bool(latest.get(c["id"], {}).get("service_error")))
    return rows


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--recovery-config", help="Service-only retry policy; model and sampling parameters remain frozen")
    parser.add_argument('--budget-config',help='User-authorized output request and scoring limit')
    args = parser.parse_args()
    out = Path(args.out)
    suite = json.loads((out / "suite.json").read_text(encoding="utf-8"))
    cases, api = suite["cases"], dict(suite["config"]["api"])
    if args.budget_config:
        api['max_output_tokens'] = api['scoring_output_limit'] = output_limit(args.budget_config)
    if args.recovery_config:
        recovery = json.loads(Path(args.recovery_config).read_text(encoding="utf-8"))
        for name in ("max_service_retries", "max_total_service_retries", "service_retry_delay_seconds"):
            api[name] = recovery.get(name,0) if name == 'service_retry_delay_seconds' else recovery[name]
    key = os.environ["GND_API_KEY"]
    records_dir = out / "records"
    records_dir.mkdir(exist_ok=True)
    attempts = [json.loads(p.read_text(encoding="utf-8")) for p in records_dir.glob("*.json")]
    latest = {r["case_id"]: r for r in sorted(attempts, key=lambda r: (r["case_id"], r["attempt"]))}
    pending_cases = pending(cases, latest, api, suite["config"]["seed"])
    previous = json.loads((out/'progress.json').read_text(encoding='utf-8')) if (out/'progress.json').exists() else None
    state = {"status": "running", "started_at": datetime.now(timezone.utc).isoformat(),
             "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
             "suite_source_commit": suite.get("source_commit"),
             "recovery_config": args.recovery_config,
             "budget_config":args.budget_config,
             "request_history":(previous.get('request_history',[]) + [{
                 'source_commit':previous.get('source_commit'),'api':previous['api'],
                 'started_at':previous.get('started_at'),'ended_at':previous['updated_at']}]) if previous else [],
             "api": api, "planned": len(cases), "inflight": []}

    def save():
        scored = [rescore(r,api['scoring_output_limit']) for r in latest.values()] if api.get('scoring_output_limit') is not None else list(latest.values())
        state.update(summary=summary(scored, cases), attempts=len(attempts),
                     updated_at=datetime.now(timezone.utc).isoformat())
        write_json(out / "progress.json", state)

    save()
    retry_total = sum(r["attempt"] > 1 for r in attempts)
    fatal = False
    with ThreadPoolExecutor(max_workers=api["concurrency"]) as executor:
        active = {}
        while pending_cases or active:
            while pending_cases and len(active) < api["concurrency"] and not fatal and not (out / "stop_requested").exists():
                case = pending_cases.pop()
                attempt = latest.get(case["id"], {}).get("attempt", 0) + 1
                if attempt > 1:
                    if retry_total >= api["max_total_service_retries"]:
                        continue
                    retry_total += 1
                active[executor.submit(call, case, api, key, attempt)] = case
            state["inflight"] = [c["id"] for c in active.values()]
            save()
            if not active:
                break
            completed, _ = wait(active, timeout=10, return_when=FIRST_COMPLETED)
            for future in completed:
                case = active.pop(future)
                record = future.result()
                write_json(records_dir / f"{case['id']}_attempt{record['attempt']}.json", record)
                attempts.append(record)
                latest[case["id"]] = record
                error = record.get("service_error")
                if error:
                    fatal = fatal or record.get("http_status") in (400,401,402,403,404) or error in ("returned_model_mismatch", "unexpected_output_item")
                    if not fatal and record["attempt"] <= api["max_service_retries"] and retry_total < api["max_total_service_retries"]:
                        pending_cases.append(case)
                print(json.dumps({"finished": len(latest), "case": case["id"], "attempt": record["attempt"], "solved": record["verdict"].get("solved"), "failure": record["verdict"].get("failure"), "seconds": record["elapsed_seconds"]}), flush=True)
                save()
    state["inflight"] = []
    state["status"] = ("completed" if all(not r.get("service_error") for r in latest.values())
                       else "service_incomplete") if len(latest) == len(cases) and not pending_cases else "stopped"
    state["completed_at"] = datetime.now(timezone.utc).isoformat()
    save()


if __name__ == "__main__":
    main()
