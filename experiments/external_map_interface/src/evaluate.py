"""Stepwise path evaluation with actual environment feedback after each action."""

import argparse
import json
from pathlib import Path
import subprocess

from experiments.cml_map_scaling.src.core import shortest_distances

from .interface import graph_prompt
from .planner import SGLangCaller, parse_action_id
from .q_map import GraphQMap
from .transitions import GraphEnvironment


def run_episode(env: GraphEnvironment, q_map: GraphQMap | None, caller,
                start: int, goal: int, step_limit: int) -> dict:
    current = start
    trace = []
    for _ in range(step_limit):
        if current == goal:
            break
        prompt = graph_prompt(env, current, goal, q_map)
        response = caller(prompt)
        try:
            action_id = parse_action_id(response)
            next_node = env.execute(current, action_id)
        except (ValueError, json.JSONDecodeError) as error:
            return {"start": start, "goal": goal, "reached": False,
                    "failure": "invalid_action", "error": str(error), "trace": trace}
        trace.append({"current": current, "action_id": action_id,
                      "actual_next": next_node, "prompt": prompt, "response": response})
        current = next_node
    return {"start": start, "goal": goal, "reached": current == goal,
            "failure": None if current == goal else "step_limit", "trace": trace}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--map", type=Path, required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--condition", choices=["plain", "reasoning", "distance"], required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    env = GraphEnvironment.load(args.inputs)
    q_map = GraphQMap.load(args.map, len(env.adjacency), len(env.actions)) if args.condition == "distance" else None
    caller = SGLangCaller(args.model_path, args.endpoint,
                           enable_thinking=args.condition != "plain",
                           max_new_tokens=config["max_new_tokens"])
    truth = shortest_distances(env.adjacency)  # Evaluation only; never sent to caller.
    records = []
    for start, goal in config["pairs"]:
        record = run_episode(env, q_map, caller, start, goal, config["step_limit"])
        record["shortest_moves"] = int(truth[start, goal])
        record["moves"] = len(record["trace"])
        records.append(record)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    source_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    args.out.write_text(json.dumps({"source_commit": source_commit,
                                   "config": config, "condition": args.condition,
                                   "inputs": str(args.inputs.resolve()),
                                   "map": str(args.map.resolve()) if q_map is not None else None,
                                   "model_path": args.model_path,
                                   "records": records},
                                   ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
