"""Replay frozen five-graph records and summarize valid arrival and cost."""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

from .evaluate_path256_continuous import detect_boundary, map_update, path_error, trial_prompt
from .evaluate_path256_continuous_batch import trial_seed
from .q_map import GraphQMap
from .transitions import environment_from_suite


def summarize(config, source, root, allow_partial=False):
    conditions = config["conditions"]
    expected = (config["graph_count"] * len(config["case_indices"])
                * config["replicates"] * len(conditions))
    files = sorted(root.glob("graph_*/**/path_undirected_256_*_rep*.json"))
    seen = set()
    groups = defaultdict(list)
    for graph in range(config["graph_count"]):
        graph_dir = f"graph_{graph:02d}"
        suite = json.loads((source / graph_dir / "suite.json").read_text(encoding="utf-8"))
        env = environment_from_suite(suite)
        q_map = GraphQMap.load(source / graph_dir / "map" / "map.npz",
                               len(env.adjacency), len(env.actions))
        for file in files:
            if file.parent.parent.name != graph_dir:
                continue
            record = json.loads(file.read_text(encoding="utf-8"))
            condition_name = file.parent.name
            if condition_name not in conditions:
                raise ValueError(f"Unknown condition: {file}")
            condition = conditions[condition_name]
            case_index = record["case_index"]
            replicate = record["replicate"]
            identity = (graph, condition_name, case_index, replicate)
            if identity in seen:
                raise ValueError(f"Duplicate trial: {identity}")
            seen.add(identity)
            if (record["graph_index"] != graph or record["condition"] != condition_name
                    or case_index not in config["case_indices"]
                    or not 1 <= replicate <= config["replicates"]):
                raise ValueError(f"Incorrect trial identity: {file}")
            case = suite["cases"][case_index]
            if (record["case_id"] != case["id"] or
                    record["prompt"] != trial_prompt(case, condition) or
                    record["requested_seed"] != trial_seed(config, graph, case_index, replicate - 1) or
                    not record["seed_applied"]):
                raise ValueError(f"Prompt, case, or seed mismatch: {file}")
            if (record["generated_tokens"] != sum(s["completion_tokens"] for s in record["segments"])
                    or record["generated_tokens"] > config["max_generated_tokens_per_question"]
                    or not record["token_accounting_complete"]):
                raise ValueError(f"Token accounting mismatch: {file}")
            if condition["map_updates"]:
                confirmed = [case["start"]]
                for index, action in enumerate(record["actions"]):
                    update, choices = map_update(case, confirmed, env, q_map)
                    segment = record["segments"][action["segment"]]
                    boundary = detect_boundary(segment["text"], confirmed)
                    if (record["updates"][index] != update or boundary != segment["boundary"]
                            or segment["accepted_text"] != segment["text"][:boundary["end"]]
                            or segment["discarded_text"] != segment["text"][boundary["end"]:]
                            or boundary.get("node") != action["to"]
                            or action["from"] != confirmed[-1]
                            or action["to"] not in [n for n, _ in choices]
                            or action["to"] in confirmed
                            or env.execute(confirmed[-1], action["action_id"]) != action["to"]):
                        raise ValueError(f"Intervention replay mismatch: {file}")
                    confirmed.append(action["to"])
                if record["confirmed"] != confirmed:
                    raise ValueError(f"Confirmed path mismatch: {file}")
            elif record["updates"] or record["actions"] or len(record["segments"]) != 1:
                raise ValueError(f"Unexpected intervention in one-shot trial: {file}")
            if record["reached"]:
                if (path_error(case, record["final_path"]) is not None or
                        record["shortest"] != (len(record["final_path"]) - 1 == case["reference"]["length"])):
                    raise ValueError(f"Arrival judge mismatch: {file}")
                if condition["map_updates"] and record["final_path"] != record["confirmed"]:
                    raise ValueError(f"Controller path mismatch: {file}")
            groups[(graph, condition_name)].append(record)
    if len(seen) != len(files):
        raise ValueError("Unassigned record files")
    if len(seen) != expected and not allow_partial:
        raise ValueError(f"Incomplete batch: {len(seen)}/{expected}")

    def group_summary(records):
        failures = Counter(record["failure"] for record in records if record["failure"])
        arrivals = [record for record in records if record["reached"]]
        return {
            "records": len(records), "arrived": len(arrivals),
            "shortest": sum(record["shortest"] for record in records),
            "failures": dict(sorted(failures.items())),
            "mean_moves_if_arrived": (sum(len(record["final_path"]) - 1 for record in arrivals)
                                      / len(arrivals) if arrivals else None),
            "mean_generated_tokens": (sum(record["generated_tokens"] for record in records)
                                      / len(records) if records else None),
            "mean_inserted_tokens": (sum(record["inserted_tokens"] for record in records)
                                     / len(records) if records else None),
        }

    by_graph = {f"graph_{graph:02d}": {name: group_summary(groups[(graph, name)])
                                       for name in conditions}
                for graph in range(config["graph_count"])}
    overall = {name: group_summary([record for graph in range(config["graph_count"])
                                    for record in groups[(graph, name)]]) for name in conditions}
    return {"study": config["study"], "protocol": config["protocol"],
            "expected_records": expected, "validated_records": len(seen),
            "complete": len(seen) == expected, "by_graph": by_graph, "overall": overall}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    result = summarize(config, args.source, args.records, args.allow_partial)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"validated_records": result["validated_records"],
                      "expected_records": result["expected_records"],
                      "complete": result["complete"]}))


if __name__ == "__main__":
    main()
