"""Run the authorized difficulty ladder, stopping each task at its first model error."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess

from .difficulty import levels, prepare_level
from .protocol import ROOT
from .report import build
from .run import call, summary, write_json


def latest_records(folder):
    rows = [json.loads(path.read_text(encoding="utf-8")) for path in (folder/"records").glob("*.json")]
    return {row["case_id"]:row for row in sorted(rows,key=lambda r:(r["case_id"],r["attempt"]))}


def baseline_failure(folder,task):
    original_task = {"cube3":"cube2","puzzle15":"puzzle8"}.get(task,task)
    for row in latest_records(folder).values():
        if row["task"] == original_task and not row.get("service_error") and not row["verdict"].get("contract_pass"):
            return {"case_id":row["case_id"],"origin":"original_baseline","verdict":row["verdict"]}
    return None


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--config",required=True)
    parser.add_argument("--out",required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    api = json.loads((ROOT/config["api_config"]).read_text())["api"]
    recovery = json.loads((ROOT/config["service_recovery"]).read_text())
    api.update({key:recovery[key] for key in ("max_service_retries","max_total_service_retries")})
    api["concurrency"] = config["concurrency"]
    out = Path(args.out)
    (out/"records").mkdir(parents=True,exist_ok=True)
    source_commit = subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()
    suite = json.loads((out/"suite.json").read_text()) if (out/"suite.json").exists() else {
        "config":dict(config,api=api),"source_commit":source_commit,"cases":[]}
    state = json.loads((out/"progress.json").read_text()) if (out/"progress.json").exists() else {
        "status":"running","started_at":datetime.now(timezone.utc).isoformat(),"source_commit":source_commit,
        "api":api,"tasks":{task:{"status":"active","level":0} for task in config["tasks"]}}
    latest = latest_records(out)
    key = os.environ["GND_API_KEY"]

    def save():
        state.update(summary=summary(list(latest.values()),suite["cases"]),updated_at=datetime.now(timezone.utc).isoformat())
        write_json(out/"suite.json",suite)
        write_json(out/"progress.json",state)

    state["status"] = "running"
    save()
    fatal = False
    while not fatal and any(row["status"] == "active" for row in state["tasks"].values()):
        for task,options in config["tasks"].items():
            item = state["tasks"][task]
            if item["status"] != "active":
                continue
            if (out/"stop_requested").exists():
                state["status"] = "stopped"
                save()
                return
            failure = baseline_failure(ROOT/config["baseline_run"],task)
            if failure:
                item.update(status="model_error_found",first_error=failure)
                save()
                continue
            level = item["level"]
            if level >= len(levels(task,options)):
                item["status"] = "range_exhausted"
                save()
                continue
            cases = [case for case in suite["cases"] if case["task"] == task and case["level"] == level]
            if not cases:
                cases = prepare_level(task,level,options,config)
                suite["cases"].extend(cases)
                save()
            selected = None
            for case in cases:
                record = latest.get(case["id"])
                if record is not None and not record.get("service_error"):
                    if not record["verdict"].get("contract_pass"):
                        item.update(status="model_error_found",first_error={"case_id":case["id"],"origin":"difficulty_ladder","level_value":case["level_value"],"verdict":record["verdict"]})
                        break
                    continue
                if record and record["attempt"] > api["max_service_retries"]:
                    item.update(status="service_blocked",case_id=case["id"])
                    break
                selected = case
                break
            if item["status"] != "active":
                save()
                continue
            if selected is None:
                item["level"] += 1
                save()
                continue
            attempt = latest.get(selected["id"],{}).get("attempt",0)+1
            if attempt > 1 and sum(max(0,row["attempt"]-1) for row in latest.values()) >= api["max_total_service_retries"]:
                state["status"] = "service_budget_exhausted"
                save()
                return
            state["inflight"] = [selected["id"]]
            save()
            record = call(selected,api,key,attempt)
            write_json(out/"records"/f"{selected['id']}_attempt{attempt}.json",record)
            latest[selected["id"]] = record
            state["inflight"] = []
            fatal = record.get("http_status") in (400,401,403,404) or record.get("service_error") in ("returned_model_mismatch","unexpected_output_item")
            print(json.dumps({"case":selected["id"],"level_value":selected["level_value"],"attempt":attempt,"verdict":record["verdict"]}),flush=True)
            save()
            if fatal:
                break
    state["status"] = "api_fatal" if fatal else "completed_breakpoints_or_ranges"
    save()
    result,text = build(out)
    (out/"summary").mkdir(exist_ok=True)
    write_json(out/"summary/summary.json",result)
    (out/"summary/report.md").write_text(text,encoding="utf-8")


if __name__ == "__main__":
    main()
