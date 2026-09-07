"""Detached sampling job followed by offline replay; produces one acceptance notice."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
from tasks import ROOT, STUDY


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--continue-from", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    notice_path = STUDY / "runs/finalization.json"
    notice = {"status": "sampling", "run_path": str(out / "run.json"),
              "started_at": datetime.now(timezone.utc).isoformat()}

    def save():
        temp = notice_path.with_suffix(".json.tmp")
        temp.write_text(json.dumps(notice, indent=2) + "\n", encoding="utf-8")
        temp.replace(notice_path)

    save()
    try:
        with (out / "console.log").open("w", encoding="utf-8") as log:
            child = subprocess.Popen([sys.executable, str(STUDY / "src/run.py"), "--out", args.out,
                                      "--continue-from", args.continue_from], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            notice["sampler_pid"] = child.pid
            save()
            code = child.wait()
        if code:
            raise RuntimeError(f"Sampler exited with code {code}; inspect console.log")
        notice["status"] = "replaying"
        save()
        subprocess.run([sys.executable, str(STUDY / "src/analyze.py"), "--run", str(out / "run.json"),
                        "--out", str(out / "analysis")], cwd=ROOT, check=True, capture_output=True, text=True)
        summary = json.loads((out / "analysis/summary.json").read_text(encoding="utf-8"))
        full = all(s["final_samples"] == s["planned_samples"] and s["pass8_eligible_cases"] == 16
                   for s in summary["conditions"].values())
        notice.update({"status": "ready_for_acceptance" if full else "needs_attention",
                       "analysis_path": str(out / "analysis"), "run_status": summary["run_status"],
                       "final_samples": sum(s["final_samples"] for s in summary["conditions"].values()),
                       "planned_samples": sum(s["planned_samples"] for s in summary["conditions"].values())})
    except Exception as error:
        notice.update({"status": "needs_attention", "error": str(error)})
    notice["finished_at"] = datetime.now(timezone.utc).isoformat()
    save()


if __name__ == "__main__":
    main()
