"""Use the existing Responses transport/continuation interface with this study's tasks."""
import importlib.util
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


if __name__ == "__main__":
    if "--suite" not in sys.argv:
        sys.argv.extend(["--suite", "experiments/sol_dag_blocks/runs/suite.json"])
    transport.main()
