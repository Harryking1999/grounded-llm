"""Use the existing Responses transport/continuation interface with this study's tasks."""
import importlib.util
import argparse
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from datetime import datetime, timezone
import itertools
import json
import os
from pathlib import Path
import random
import subprocess
import sys
from tasks import CONFIG, ROOT, TASKS

LEGACY = ROOT / "experiments/gcml_counterexamples/src"
sys.path.append(str(LEGACY))
spec = importlib.util.spec_from_file_location("counterexample_transport", LEGACY / "run.py")
transport = importlib.util.module_from_spec(spec)
spec.loader.exec_module(transport)


def prompt_for(case):
    return TASKS[case["condition"]].prompt(case)


def evaluate(case, raw):
    output, strict = transport.parse_output(raw)
    if not isinstance(output, dict):
        return {"pass": False, "parseable": False, "strict_json": strict, "failure_type": "parse_error"}
    verdict = TASKS[case["condition"]].judge(case, output)
    verdict["contract_pass"] = verdict.get("contract_pass", False) and strict
    return {**verdict, "parseable": True, "strict_json": strict}


transport.prompt_for = prompt_for
transport.evaluate = evaluate


def classify_service_failure(record):
    error = (record.get("response", {}).get("error") or {})
    if record.get("response_status") == "failed" and error.get("code") in (
            "gateway_concurrency_limit", "server_error", "upstream_error"):
        record["api_error"] = error["code"]
        record["verdict"] = {"pass": False, "failure_type": "api_error"}
    return record


def call(case, replicate, config, key):
    return classify_service_failure(transport.call(case, replicate, config, key))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", default="experiments/sol_dag_blocks/runs/suite.json")
    parser.add_argument("--api-config", default="experiments/sol_dag_blocks/configs/sol_medium.json")
    parser.add_argument("--out", required=True)
    parser.add_argument("--continue-from")
    parser.add_argument("--stop-file", default="experiments/sol_dag_blocks/runs/stop_requested")
    args = parser.parse_args()
    suite = json.loads((ROOT / args.suite).read_text(encoding="utf-8"))
    committed = json.loads((ROOT / args.api_config).read_text(encoding="utf-8"))
    if (ROOT / committed["suite"]).resolve() != (ROOT / args.suite).resolve():
        raise ValueError("API config does not name this suite")
    config, cases = committed["api"], suite["cases"]
    key = os.environ.get("GND_API_KEY")
    if not key:
        raise ValueError("Set GND_API_KEY in the process environment")
    retained, archived = [], []
    if args.continue_from:
        previous = json.loads((ROOT / args.continue_from).read_text(encoding="utf-8"))
        if previous["status"] not in ("stopped_api_error", "stopped_requested"):
            raise ValueError("Only stopped runs may be continued")
        for record in previous["cases"]:
            classify_service_failure(record)
        retained, archived = transport.continuation_samples(previous, cases, config)
    done_keys = {(r["case_id"], r["replicate"]) for r in retained}
    jobs = [(c, n) for c in cases for n in range(1, c["replicates"] + 1)]
    random.Random(suite["config"]["seed"] + 30000).shuffle(jobs)
    jobs = [(c, n) for c, n in jobs if (c["id"], n) not in done_keys]
    if not jobs or len(jobs) + len(retained) > config["max_calls"]:
        raise ValueError("Invalid trial count")
    out = ROOT / args.out / "run.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        raise FileExistsError("Choose a new output directory")
    payload = {"source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
               "suite_path": args.suite, "api_config": config, "api_config_path": args.api_config,
               "planned_calls": len(jobs), "continued_from": args.continue_from,
               "prior_failed_attempts": archived, "retained_final_samples": len(retained),
               "started_at": datetime.now(timezone.utc).isoformat(), "status": "running", "cases": retained}

    def save():
        payload["summary"] = transport.aggregate(payload["cases"], cases)
        temporary = out.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        temporary.replace(out)

    def accept(record):
        payload["cases"].append(record)
        save()
        print(json.dumps({"done": len(payload["cases"]), "total": config["max_calls"], "case": record["case_id"],
                          "replicate": record["replicate"], "pass": record["verdict"].get("pass"),
                          "failure": record["verdict"].get("failure_type"), "seconds": record["elapsed_seconds"]}), flush=True)

    stopped, remaining = None, iter(jobs)
    save()
    # The previous phase already validated this identical model/endpoint contract.
    if not retained:
        case, replicate = next(remaining)
        record = call(case, replicate, config, key)
        accept(record)
        if not transport.final_sample(record):
            stopped = "stopped_api_error"
    with ThreadPoolExecutor(max_workers=config["concurrency"]) as executor:
        pending = set()
        while True:
            if not stopped and (ROOT / args.stop_file).exists():
                stopped = "stopped_requested"
            if not stopped:
                for case, replicate in itertools.islice(remaining, config["concurrency"] - len(pending)):
                    pending.add(executor.submit(call, case, replicate, config, key))
            if not pending:
                break
            completed, pending = wait(pending, return_when=FIRST_COMPLETED)
            for future in completed:
                record = future.result()
                accept(record)
                if not transport.final_sample(record):
                    stopped = "stopped_api_error"
    payload["status"] = stopped or "completed"
    payload["finished_at"] = datetime.now(timezone.utc).isoformat()
    save()
    print(json.dumps({"output": str(out), "status": payload["status"], "final_samples": sum(transport.final_sample(r) for r in payload["cases"])}), flush=True)


if __name__ == "__main__":
    main()
