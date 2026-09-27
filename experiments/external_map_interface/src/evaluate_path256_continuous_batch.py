"""Run the frozen five-graph suite using the validated continuous-path protocol."""

import argparse
import json
import os
from pathlib import Path

from .evaluate_path256_continuous import (
    PROTOCOL, SGLangContinuousCaller, run_trial,
)
from .q_map import GraphQMap
from .transitions import environment_from_suite


def trial_seed(config, graph_index, case_index, replicate_index):
    return (config["seed"] + graph_index * len(config["case_indices"]) * config["replicates"]
            + case_index * config["replicates"] + replicate_index)


def shard_cases(config, shard_index, shard_count):
    if shard_count < 1 or not 0 <= shard_index < shard_count:
        raise ValueError("Invalid shard")
    return [index for index in config["case_indices"] if index % shard_count == shard_index]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--map", type=Path, required=True)
    parser.add_argument("--graph-index", type=int, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--condition", required=True)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--replicate-start", type=int, default=1)
    parser.add_argument("--replicate-end", type=int)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if config.get("protocol") != PROTOCOL:
        parser.error("Unsupported protocol")
    if not 0 <= args.graph_index < config["graph_count"]:
        parser.error("Graph index outside configured range")
    condition = config["conditions"][args.condition]
    if args.model_path.name != f"Qwen3-4B-{condition['model'].capitalize()}-2507":
        parser.error("Condition and model path disagree")
    suite = json.loads(args.suite.read_text(encoding="utf-8"))
    if len(suite["cases"]) != len(config["case_indices"]):
        parser.error("Frozen suite has an unexpected number of cases")
    indices = shard_cases(config, args.shard_index, args.shard_count)
    final_replicate = args.replicate_end or config["replicates"]
    if not 1 <= args.replicate_start <= final_replicate <= config["replicates"]:
        parser.error("Invalid replicate range")
    env = environment_from_suite(suite)
    q_map = (GraphQMap.load(args.map, len(env.adjacency), len(env.actions))
             if condition["map_updates"] else None)
    caller = SGLangContinuousCaller(
        str(args.model_path), args.endpoint,
        enable_thinking=condition["model"] == "thinking",
        sampling=config["sampling"], context_length=config["context_length"],
        per_request_seed=config["sglang_per_request_seed"],
    )
    args.out.mkdir(parents=True, exist_ok=True)
    for replicate in range(args.replicate_start, final_replicate + 1):
        for index in indices:
            case = suite["cases"][index]
            destination = args.out / f"{case['id']}_rep{replicate:02d}.json"
            if destination.exists():
                previous = json.loads(destination.read_text(encoding="utf-8"))
                if (previous.get("case_id") != case["id"] or
                        previous.get("condition") != args.condition or
                        previous.get("replicate") != replicate or
                        previous.get("graph_index") != args.graph_index):
                    raise ValueError(f"Existing record has different identity: {destination}")
                continue
            seed = trial_seed(config, args.graph_index, index, replicate - 1)
            record = run_trial(case, args.condition, condition, caller, env, q_map,
                               config["max_generated_tokens_per_question"], seed)
            record.update({"graph_index": args.graph_index, "case_index": index,
                           "replicate": replicate, "run_seed": seed,
                           "run_inputs": {"config": str(args.config),
                                          "suite": str(args.suite), "map": str(args.map),
                                          "model_path": str(args.model_path),
                                          "backend": "sglang"}})
            temporary = destination.with_suffix(".json.tmp")
            with temporary.open("w", encoding="utf-8") as stream:
                json.dump(record, stream, ensure_ascii=False, indent=2)
                stream.write("\n")
            os.replace(temporary, destination)
            print(json.dumps({"graph": args.graph_index, "condition": args.condition,
                              "case": index, "replicate": replicate,
                              "reached": record["reached"], "failure": record["failure"],
                              "moves": len(record["final_path"]) - 1 if record["final_path"] else None,
                              "generated_tokens": record["generated_tokens"]}), flush=True)


if __name__ == "__main__":
    main()
