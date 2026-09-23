"""Matched stepwise path256 comparison with per-decision output budget."""

import argparse
from collections import Counter
import json
from pathlib import Path
import subprocess
import time

from experiments.qwen_path_blocks.src.prepare_path_suite import validate_suite
from experiments.sol_dag_blocks.src.tasks import TASKS

from .interface import graph_prompt
from .planner import SGLangCaller, parse_action_id
from .q_map import GraphQMap
from .transitions import environment_from_suite


def run_trial(case, replicate, env, q_map, caller, config, endpoint):
    current = case["start"]
    path = [current]
    trace = []
    failure = None
    case_index = int(case["id"].rsplit("_", 1)[-1])
    base_seed = config["seed"] + case_index * config["replicates"] + replicate - 1
    for step_index in range(config["step_limit"]):
        if current == case["goal"]:
            break
        legal = [a for a in env.legal_actions(current) if int(env.actions[a, 1]) not in path]
        if not legal:
            failure = "no_legal_unvisited_action"
            break
        prompt = graph_prompt(env, current, case["goal"], q_map, case=case, path=path)
        seed = base_seed + step_index * config["pair_count"] * config["replicates"]
        started = time.monotonic()
        response = caller(prompt, seed=seed, endpoint=endpoint)
        record = {"step": step_index, "current": current, "seed": seed, "prompt": prompt,
                  "response": response, "elapsed_seconds": time.monotonic() - started}
        trace.append(record)
        if response["finish_reason"] != "stop":
            failure = "budget_truncated" if response["finish_reason"] == "length" else "model_incomplete"
            break
        try:
            action = parse_action_id(response["text"])
        except (ValueError, json.JSONDecodeError) as error:
            record["error"] = repr(error)
            failure = "invalid_format"
            break
        if action not in legal:
            record["action_id"] = action
            failure = "illegal_or_repeated_action"
            break
        following = env.execute(current, action)
        record.update({"action_id": action, "actual_next": following})
        path.append(following)
        current = following
    if not failure and current != case["goal"]:
        failure = "step_limit"
    moves = [{"from": left, "to": right} for left, right in zip(path, path[1:])]
    verdict = TASKS["path_undirected_256"].judge(case, {"path": moves, "final_node": current})
    if verdict["executed_moves"] != len(moves):
        raise AssertionError("Environment and original judge disagree")
    return {"case_id": case["id"], "replicate": replicate, "start": case["start"],
            "goal": case["goal"], "path": path, "reached": verdict["execution_pass"],
            "shortest": verdict["pass"], "shortest_moves": verdict["shortest_moves"],
            "moves": len(moves), "failure": failure,
            "output_tokens": sum(t["response"].get("completion_tokens") or 0 for t in trace),
            "input_tokens": sum(t["response"].get("prompt_tokens") or 0 for t in trace),
            "elapsed_seconds": sum(t["elapsed_seconds"] for t in trace), "trace": trace}


def summarize(records):
    reached = [r for r in records if r["reached"]]
    failures = Counter(r["failure"] for r in records if r["failure"])
    by_case = {}
    for record in records:
        bucket = by_case.setdefault(record["case_id"], {"trials": 0, "shortest": 0, "reached": 0})
        bucket["trials"] += 1
        bucket["shortest"] += int(record["shortest"])
        bucket["reached"] += int(record["reached"])
    return {"trials": len(records), "shortest": sum(r["shortest"] for r in records),
            "reached": len(reached), "shortest_pass_at_8": sum(b["shortest"] > 0 for b in by_case.values()),
            "failures": failures, "mean_extra_moves_when_reached":
            sum(r["moves"] - r["shortest_moves"] for r in reached) / len(reached) if reached else None,
            "total_output_tokens": sum(r["output_tokens"] for r in records),
            "total_input_tokens": sum(r["input_tokens"] for r in records),
            "total_model_seconds": sum(r["elapsed_seconds"] for r in records), "by_case": by_case}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--map", type=Path, required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--endpoints", nargs="+", required=True)
    parser.add_argument("--condition", choices=["plain", "reasoning", "distance"], required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--first-trial-only", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    suite = json.loads(args.suite.read_text(encoding="utf-8"))
    validate_suite(suite)
    if (len(suite["cases"]) != config["pair_count"] or
            any(c["replicates"] != config["replicates"] for c in suite["cases"])):
        raise ValueError("Suite does not match fixed pair/replicate contract")
    env = environment_from_suite(suite)
    q_map = GraphQMap.load(args.map, len(env.adjacency), len(env.actions)) if args.condition == "distance" else None
    condition = config["conditions"][args.condition]
    caller = SGLangCaller(args.model_path, args.endpoints[0],
                           enable_thinking=condition["enable_thinking"],
                           max_new_tokens=config["sampling"]["max_new_tokens"],
                           timeout_seconds=config["timeout_seconds"], sampling=config["sampling"],
                           context_length=config["serving"]["context_length"])
    args.out.mkdir(parents=True, exist_ok=True)
    slots = [(case, replicate) for case in suite["cases"]
             for replicate in range(1, config["replicates"] + 1)]
    if args.first_trial_only:
        slots = slots[:1]
    identity = {"source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                "config": config, "suite": str(args.suite.resolve()), "condition": args.condition,
                "model_path": args.model_path, "map": str(args.map.resolve()) if q_map is not None else None}
    identity_path = args.out / "run_config.json"
    if identity_path.exists():
        if json.loads(identity_path.read_text(encoding="utf-8")) != identity:
            raise ValueError("Existing run has a different contract")
    else:
        identity_path.write_text(json.dumps(identity, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    record_dir = args.out / "samples"
    record_dir.mkdir(exist_ok=True)
    for index, (case, replicate) in enumerate(slots):
        path = record_dir / f"{case['id']}__{replicate}.json"
        if path.exists():
            continue
        record = run_trial(case, replicate, env, q_map, caller, config, args.endpoints[index % len(args.endpoints)])
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temporary.replace(path)
        print(json.dumps({"saved": path.name, "shortest": record["shortest"],
                          "reached": record["reached"], "failure": record["failure"],
                          "steps": len(record["trace"]), "output_tokens": record["output_tokens"]}), flush=True)
    records = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(record_dir.glob("*.json"))]
    (args.out / "summary.json").write_text(json.dumps(summarize(records), ensure_ascii=False, indent=2) + "\n",
                                           encoding="utf-8")
    print(json.dumps({"complete": len(records) == config["pair_count"] * config["replicates"],
                      "saved_trials": len(records)}), flush=True)


if __name__ == "__main__":
    main()
