"""Recompute the reachability metric on evaluation shards that predate it.

Every shard on disk was written before `1c516e4`, so its turns record
`action_environment_shortest` but not `action_keeps_goal_reachable`. The state a
turn saw is still recoverable offline -- a reference shard replays the gold
demonstration it was scored against, a rollout shard replays the path its own
answers produced -- so the metric can be filled in without another generation pass.

A shard is rewritten only after the reconstruction reproduces, turn by turn, the
two values the shard already holds: `remaining_shortest` and
`action_environment_shortest`. One disagreement leaves that shard alone, which is
also what makes the replay trustworthy rather than merely plausible.

The same tool re-derives summaries when the *aggregation* changes rather than the
metric, which needs no replay and is expected to disagree with the recorded
numbers:

Usage:
    python -m experiments.flamingo_map_reader.src.backfill_reachability --run <run root> [--apply]
    python -m experiments.flamingo_map_reader.src.backfill_reachability --run <run root> --resummarize [--apply]
"""

from argparse import ArgumentParser
from collections import defaultdict
import json
import math
import os
from pathlib import Path

import numpy as np

from .trajectory_dataset import load_record
from .trajectory_eval import TaskEnvironment
from .trajectory_metrics import summarize_turns


TURN_FIELDS = ("action_keeps_goal_reachable", "reachable_candidates", "candidate_slots")
DIMENSIONS = ("turn", "remaining_shortest", "candidates", "group")


class ReplayError(Exception):
    """The reconstruction cannot be trusted, so the shard is left untouched."""


def same(recorded, rebuilt):
    if recorded is None or rebuilt is None:
        return recorded is rebuilt
    if isinstance(recorded, float) or isinstance(rebuilt, float):
        return math.isclose(float(recorded), float(rebuilt), rel_tol=1e-12, abs_tol=1e-12)
    return recorded == rebuilt


def reference_states(trajectories, config, record, case):
    """A reference shard was scored against the gold demonstration, re-numbered the same way."""
    demo = load_record(trajectories, record, config, case["variant"])
    if len(demo.turns) != len(case["turns"]):
        raise ReplayError(f"{case['trajectory_id']} v{case['variant']}: "
                          f"{len(demo.turns)} gold turns against {len(case['turns'])} recorded")
    for index, turn in enumerate(demo.turns):
        yield index, turn.step, list(turn.executed_path)


def rollout_states(environment, record, case):
    """Replay closed_loop's own rng: its shuffles decide which local id meant which move."""
    path, goal = [int(record["start"])], int(record["goal"])
    rng = np.random.default_rng(record["sample_seed"] + 104729 * (case["variant"] + 1))
    recorded = [int(node) for node in case["actual_path"]]
    for index in range(len(case["turns"])):
        step = environment.step(path, goal, rng)
        if path != recorded[:index + 1]:
            raise ReplayError(f"{case['trajectory_id']} v{case['variant']}: replayed path {path} "
                              f"diverges from {recorded[:index + 1]} at turn {index}")
        yield index, step, list(path)
        if len(recorded) < index + 2:
            return
        row = case["turns"][index]
        action, destination = environment.execute(step, row["chosen_id"])
        if action != row.get("chosen_action"):
            raise ReplayError(f"{case['trajectory_id']} v{case['variant']}: replayed action {action} "
                              f"against recorded {row.get('chosen_action')} at turn {index}")
        path.append(destination)


def enrich(case, environment, states):
    """Add the reachability fields to one case's scored turns, checking the replay as it goes."""
    checked = 0
    for index, step, path in states:
        row = case["turns"][index]
        # Closed loop only attaches this on turns whose action it went on to execute;
        # reference attaches it everywhere, terminal turns included.
        if "action_environment_shortest" not in row:
            continue
        remaining = environment.remaining(step.current, step.goal, path)
        if remaining != row["remaining_shortest"]:
            raise ReplayError(f"{case['trajectory_id']} v{case['variant']} turn {index}: "
                              f"remaining {remaining} against recorded {row['remaining_shortest']}")
        chosen = row.get("chosen_id")
        if environment.chosen_is_shortest(step, chosen, path, remaining) != row["action_environment_shortest"]:
            raise ReplayError(f"{case['trajectory_id']} v{case['variant']} turn {index}: "
                              "recomputed action_environment_shortest disagrees with the recorded one")
        row["action_keeps_goal_reachable"] = environment.chosen_keeps_reachable(step, chosen, path, remaining)
        row["reachable_candidates"] = environment.reachable_candidates(step, path, remaining)
        row["candidate_slots"] = 0 if step.done else len(step.candidate_actions)
        checked += 1
    return checked


def restratify(rows):
    groups = {}
    for dimension in DIMENSIONS:
        by_key = defaultdict(list)
        for row in rows:
            by_key[str(row[dimension])].append(row)
        groups[dimension] = {key: summarize_turns(value) for key, value in by_key.items()}
    return groups


def rebuild(cases, summary, verify=True):
    """Re-derive the shard's aggregates, refusing to proceed unless the old ones come back.

    `verify` compares every recorded field against the rebuilt one, which is what
    makes the replay behind a backfill trustworthy. It has to be turned off for
    --resummarize, where the recorded numbers are the *old* aggregation rule's and
    are expected to differ: there the rows are taken as already correct and only
    the summary is re-derived.
    """
    rows = [dict(row, group=case["group"]) for case in cases for row in case["turns"]]
    overall = summarize_turns(rows)
    strata = restratify(rows)
    if verify:
        for field, recorded in summary["overall"].items():
            if not same(recorded, overall.get(field)):
                raise ReplayError(f"rebuilt overall[{field}]={overall.get(field)!r} against recorded {recorded!r}")
        for dimension, recorded in summary["strata"].items():
            for key, value in recorded.items():
                rebuilt = strata.get(dimension, {}).get(key)
                if rebuilt is None:
                    raise ReplayError(f"stratum {dimension}={key} disappeared")
                for field, count in value.items():
                    if not same(count, rebuilt.get(field)):
                        raise ReplayError(f"rebuilt strata[{dimension}][{key}][{field}]="
                                          f"{rebuilt.get(field)!r} against recorded {count!r}")
    summary["overall"], summary["strata"] = overall, strata
    summary["backfill"] = dict(metric="action_keeps_goal_reachable", commit="1c516e4", fields=list(TURN_FIELDS))
    return summary


def write_shard(shard, cases, summary, rewrite_cases=True):
    """Replace a shard's files, so the per-turn record and its summary agree."""
    for name, text in (([("cases.jsonl", "".join(json.dumps(case) + "\n" for case in cases))] if rewrite_cases else [])
                       + [("summary.json", json.dumps(summary, indent=2) + "\n")]):
        temporary = shard / (name + ".backfill")
        temporary.write_text(text)
        os.replace(temporary, shard / name)


def shard_cases(shard):
    lines = (shard / "cases.jsonl").read_text().splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def main():
    parser = ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", required=True, type=Path, help="run root holding <task>/data and evaluation/")
    parser.add_argument("--apply", action="store_true", help="rewrite the shards; without it, only report")
    parser.add_argument("--shard", action="append", type=Path, help="limit to these shard directories")
    parser.add_argument("--resummarize", action="store_true",
                        help="re-derive each summary from the rows already on disk instead of replaying the "
                             "episode, and skip the check against the recorded numbers; for when the "
                             "aggregation rule changes rather than the metric it aggregates")
    args = parser.parse_args()
    shards = args.shard or sorted(path.parent for path in args.run.glob("evaluation/*/*/*/*/summary.json"))
    context, totals = {}, dict(checked=0, skipped=0, turns=0)
    for shard in sorted(shards):
        if args.resummarize:
            summary = json.loads((shard / "summary.json").read_text())
            rebuild(shard_cases(shard), summary, verify=False)
            if args.apply:
                write_shard(shard, None, summary, rewrite_cases=False)
            totals["checked"] += 1
            overall = summary["overall"]
            print(f"{'wrote' if args.apply else 'ok  '} {shard} solvable={overall['solvable_decisions']} "
                  f"reachable={overall['action_keeps_goal_reachable_rate']} "
                  f"floor={overall['reachable_candidate_rate']}", flush=True)
            continue
        task = shard.relative_to(args.run / "evaluation").parts[0]
        if task not in context:
            manifest = json.loads((args.run / task / "data" / "manifest.json").read_text())
            context[task] = (manifest, manifest["config"], TaskEnvironment(manifest["config"], manifest),
                             args.run / task / "data" / "trajectories")
        manifest, config, environment, trajectories = context[task]
        records = {record["trajectory_id"]: record for record in manifest["records"]}
        summary = json.loads((shard / "summary.json").read_text())
        cases = shard_cases(shard)
        try:
            turns = sum(enrich(case, environment, (
                reference_states(trajectories, config, records[case["trajectory_id"]], case)
                if summary["contract"]["mode"] == "reference" else rollout_states(environment, records[case["trajectory_id"]], case))
                ) for case in cases)
            rebuild(cases, summary)
        except ReplayError as error:
            print(f"SKIP {shard}: {error}", flush=True)
            totals["skipped"] += 1
            continue
        if args.apply:
            write_shard(shard, cases, summary)
        totals["checked"] += 1
        totals["turns"] += turns
        overall = summary["overall"]
        print(f"{'wrote' if args.apply else 'ok  '} {shard} turns={turns} "
              f"reachable={overall['action_keeps_goal_reachable_rate']} floor={overall['reachable_candidate_rate']}",
              flush=True)
    print(json.dumps(totals), flush=True)
    print("dry run: nothing was written" if not args.apply else "applied", flush=True)


if __name__ == "__main__":
    main()
