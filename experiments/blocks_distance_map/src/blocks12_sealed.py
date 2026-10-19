"""Freeze independent roots without models; explicitly open one final comparison."""
import argparse
import json
from pathlib import Path
import random

import torch

import grounded_llm.tasks.blocks.rules as blocks
from .blocks12_eval import construct
from .full_graph_eval import evaluate_rows, matched, sections
from grounded_llm.tasks.blocks.oracle import DistanceOracle, successors
from .tree_supervision_eval import QScorer


def validate_cases(roots, rows, placements):
    """Replay legal certificates, without loading Q or searching for distances."""
    for record in roots:
        source = int(record["initial"])
        if len(record["construction"]) != placements:
            raise ValueError("Wrong construction size")
        state = source
        for action in record["construction"]:
            state = blocks.apply(state, action)
        if state != 0:
            raise ValueError("Construction does not certify clearing")
        state = source
        for action_id in record["evaluation_path"]:
            state = blocks.apply(state, blocks.PLACEMENTS[action_id][1])
    for row in rows:
        record = roots[row["board"]]
        if row["source"] != record["initial"]:
            raise ValueError("Case source changed")
        state = int(row["source"])
        if row["certificate"] == "construction":
            actions = record["construction"]
        else:
            actions = [blocks.PLACEMENTS[i][1]
                       for i in record["evaluation_path"][:row["certificate_steps"]]]
        for action in actions:
            state = blocks.apply(state, action)
        if state != int(row["goal"]):
            raise ValueError("Goal does not match its legal-path certificate")


def excluded_masks(filenames):
    """Read training roots or monitoring cases, including frozen clear1000."""
    if isinstance(filenames, (str, Path)):
        filenames = [filenames]
    if not filenames:
        raise ValueError("At least one exclusion file is required")
    masks = set()
    for filename in filenames:
        records = json.loads(Path(filename).read_text())
        if isinstance(records, dict):
            records = records["roots"] if "roots" in records else records["cases"]
        masks.update(int(r["initial"] if "initial" in r else r["source"]) for r in records)
    return masks


def prepare(config, out, source_commit):
    train_masks = excluded_masks(config["excluded_training_roots"])
    monitor_masks = excluded_masks(config["excluded_monitoring_cases"])
    excluded = train_masks | monitor_masks
    # A single predetermined seed/prefix; filtering cannot choose an easier seed.
    candidates = construct(config["boards"] + len(excluded),
                           config["construction_placements"], config["root_seed"])
    roots = [r for r in candidates if int(r["initial"]) not in excluded][:config["boards"]]
    if len(roots) != config["boards"] or len({r["initial"] for r in roots}) != len(roots):
        raise ValueError("Insufficient distinct independent roots")
    rng = random.Random(config["path_seed"])
    rows = []
    for board, record in enumerate(roots):
        state, path, states = int(record["initial"]), [], []
        while True:
            choices = successors(state)
            if not choices:
                break
            action, state = rng.choice(choices)
            path.append(action)
            if state:
                states.append((state, len(path)))
        record["evaluation_path"] = path
        rows.append({"board": board, "source": record["initial"], "goal": "0",
                     "certificate": "construction",
                     "certificate_steps": len(record["construction"])})
        selected = set()
        for name in config["intermediate_path_positions"]:
            if not states:
                raise ValueError("No intermediate states")
            position = len(states) - 1 if name == "last" else int(name) - 1
            if position < 0:
                raise ValueError("Path positions are positive, one-based integers or 'last'")
            if position >= len(states):
                continue
            goal, steps = states[position]
            if goal in selected:
                continue
            selected.add(goal)
            rows.append({"board": board, "source": record["initial"], "goal": str(goal),
                         "certificate": "evaluation_path", "certificate_steps": steps,
                         "path_position": name})
    validate_cases(roots, rows, config["construction_placements"])
    masks = {int(r["initial"]) for r in roots}
    summary = {"source_commit": source_commit, "status": "prepared_sealed_models_not_evaluated",
               "boards": len(roots), "tasks": len(rows),
               "clear_tasks": sum(r["goal"] == "0" for r in rows),
               "intermediate_tasks": sum(r["goal"] != "0" for r in rows),
               "intermediate_position_counts": {str(name): sum(r.get("path_position") == name for r in rows)
                                                for name in config["intermediate_path_positions"]},
               "excluded_training_roots": len(train_masks), "excluded_monitoring_roots": len(monitor_masks),
               "training_root_overlap": len(masks & train_masks),
               "monitoring_root_overlap": len(masks & monitor_masks),
               "reachable_certificates_validated": len(rows), "models_evaluated": False,
               "scope": "Distinct initial roots; descendant masks can overlap across roots. Exact shortest distances are computed only when the final evaluation is explicitly opened."}
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    (out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    (out / "source_commit.txt").write_text(source_commit + "\n")
    (out / "cases.json").write_text(json.dumps({"roots": roots, "rows": rows}) + "\n")
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)
    return summary


def final_evaluate(data, checkpoint, out, source_commit, parent_checkpoint, device="cpu"):
    """Caller must first choose the final candidate using monitoring evidence."""
    data, out = Path(data), Path(out)
    config = json.loads((data / "config.json").read_text())
    cases = json.loads((data / "cases.json").read_text())
    validate_cases(cases["roots"], cases["rows"], config["construction_placements"])
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(config["cpu_threads"])
    oracle = DistanceOracle(cache_limit=config["oracle_cache_limit"], seconds=config["oracle_seconds"])
    result = {"analysis_commit": source_commit, "prepared_source_commit":
              (data / "source_commit.txt").read_text().strip(),
              "data": str(data), "opening": "one_explicit_final_checkpoint_no_reselection",
              "policy": config["policy"], "models": {}, "cycles": 0, "illegal_actions": 0,
              "scope": "Initial roots excluded from supplied training and monitoring files; descendant states may overlap. Trace ends at reach or first irreversible error."}
    results = {}
    for name, path in (("parent", parent_checkpoint),
                       ("final_candidate", checkpoint)):
        scorer = QScorer(path, device)
        rows, bins, trajectory = evaluate_rows(cases["rows"], scorer, oracle)
        result["models"][name] = {"identity": scorer.identity, "sections": sections(rows),
                                 "distance_bins": bins, "trajectory_forks": trajectory, "rows": rows}
        results[name] = rows
        del scorer
    result["matched_parent"] = matched(results["parent"], results["final_candidate"])
    # This entry point never writes the monitoring selection or any best weight.
    (out / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"matched_parent": result["matched_parent"], "models": {
        name: row["sections"] for name, row in result["models"].items()}}), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    preparation = commands.add_parser("prepare")
    preparation.add_argument("--config", type=Path, required=True)
    preparation.add_argument("--out", type=Path, required=True)
    preparation.add_argument("--source-commit", required=True)
    preparation.add_argument("--excluded-training-roots", nargs="+", required=True,
                             help="All relevant training root files")
    preparation.add_argument("--excluded-monitoring-cases", nargs="+", required=True,
                             help="All reused monitoring root/case files, including clear1000")
    evaluation = commands.add_parser("final-evaluate")
    evaluation.add_argument("--data", type=Path, required=True)
    evaluation.add_argument("--checkpoint", type=Path, required=True)
    evaluation.add_argument("--parent-checkpoint", type=Path, required=True)
    evaluation.add_argument("--out", type=Path, required=True)
    evaluation.add_argument("--source-commit", required=True)
    evaluation.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.command == "prepare":
        config = json.loads(args.config.read_text())
        config["excluded_training_roots"] = args.excluded_training_roots
        config["excluded_monitoring_cases"] = args.excluded_monitoring_cases
        prepare(config, args.out, args.source_commit)
    else:
        final_evaluate(args.data, args.checkpoint, args.out, args.source_commit,
                       parent_checkpoint=args.parent_checkpoint, device=args.device)


if __name__ == "__main__":
    main()
