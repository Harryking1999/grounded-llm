"""Compact, checkable evidence for the long-trajectory run.

Reads every finished battery under the run root, reuses ``trajectory_eval.aggregate``
so the counts are the same quantities the queue writes into a group summary, and adds
the reachability audit the report quotes.

Scoring convention (``DESIGN.md`` section 7): ``action_keeps_goal_reachable`` and
``action_environment_shortest`` are read on the **solvable non-terminal decision
turns** -- turns that are not terminal and whose state was still solvable before the
model answered (``remaining_shortest >= 0``; identical to the ``>= 1`` in ``metric_notes``
because no non-terminal turn has distance 0). The turn is excluded explicitly; a
reference shard carries the field on its stopping turn too, where remaining is 0 and
the action is always scored false. A turn whose reachability fields were never
recorded stays missing rather than counting as a failure. Each rate is printed beside
the floor of a uniformly random legal action on the same turns.

Usage:
    python collect_evidence.py --run-root RUN [--update EVIDENCE.json] [--table]

``--update`` rewrites only the ``snapshot`` and ``evaluation_batteries`` sections of an
existing evidence file and leaves the rest (``data``, ``action_coverage``,
``documentation_scope``, ``metric_notes``, ``greedy_reference_groups``) untouched. Counts
are recomputed from each shard's ``cases.jsonl`` through ``trajectory_eval.aggregate``, the
same function the queue uses to write the shard ``summary.json``.
"""

import argparse
import json
import math
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

CODE = "/zhanghanyue/experiment/flamingo_map_reader/code/long_aec7166"
DEPS = "/zhanghanyue/experiment/flamingo_map_reader/deps"
sys.path[:0] = [CODE, DEPS]
from experiments.flamingo_map_reader.src.trajectory_eval import aggregate  # noqa: E402

SHARD = 64
# Physical tasks per split, from the manifests. The train battery is a fixed
# 256-task diagnostic (trajectory_eval --limit 256), so it is 4 shards regardless
# of how many training tasks exist.
SPLIT_TASKS = {("path", "train"): 256, ("path", "validation"): 532, ("path", "test"): 1032,
               ("blocks", "train"): 256, ("blocks", "validation"): 1100, ("blocks", "test"): 2200}


def expected_tasks(task, split):
    return SPLIT_TASKS[(task, split)]


def split_of(group):
    return group.split("_", 1)[0]


def shards_of(directory):
    return sorted(path.parent for path in directory.glob("*/summary.json"))


def load(directory):
    cases = []
    for shard in shards_of(directory):
        text = (shard / "cases.jsonl").read_text()
        cases.extend(json.loads(line) for line in text.splitlines() if line.strip())
    return cases


def audit(cases):
    """Keep-reachable / environment-shortest on solvable non-terminal decision turns.

    A shard evaluated before the reachability fields existed keeps them missing, so the
    scored subset is the solvable turns that actually carry ``action_keeps_goal_reachable``.
    Counting a missing field as a failure would publish a deflated rate; the gap is
    reported as ``missing_reachability_turns`` instead.
    """
    rows = [row for case in cases for row in case["turns"]]
    decisions = [row for row in rows if not row["done"]]
    solvable = [row for row in decisions if row.get("remaining_shortest", -1) >= 0]
    scored = [row for row in solvable if "action_keeps_goal_reachable" in row]
    slots = sum(row.get("candidate_slots", 0) for row in scored)
    reachable = sum(row.get("reachable_candidates", 0) for row in scored)
    summary = dict(
        decision_turns=len(decisions),
        already_unreachable_turns=len(decisions) - len(solvable),
        current_reachable_turns=len(solvable),
        reachability_scored_turns=len(scored),
        missing_reachability_turns=len(solvable) - len(scored),
    )
    if not scored:
        return summary
    per_turn = [row.get("reachable_candidates", 0) / row["candidate_slots"]
                for row in scored if row.get("candidate_slots")]
    summary.update(
        keeps_goal_reachable=sum(bool(row["action_keeps_goal_reachable"]) for row in scored),
        action_environment_shortest=sum(bool(row.get("action_environment_shortest")) for row in scored),
        candidate_slots=slots,
        reachable_candidates=reachable,
        zero_candidate_turns=sum(row.get("candidate_slots", 0) == 0 for row in scored),
        uniform_expected_rate_current_reachable=sum(per_turn) / len(per_turn) if per_turn else None,
        uniform_expected_rate_all_decisions=(sum(per_turn) / len(decisions)) if decisions and per_turn else None,
        reachable_candidate_rate=reachable / slots if slots else None,
    )
    return summary


def rollout_of(cases):
    reached = [case for case in cases if case.get("reached_goal")]
    return dict(
        attempts=sum(case.get("mode") == "rollout" for case in cases),
        reached=len(reached),
        success=sum(bool(case.get("success")) for case in cases),
        shortest_success=sum(bool(case.get("shortest_success")) for case in cases),
        # reached_shortest drops the correct-stop requirement: arrival plus the
        # environment's shortest number of moves (DESIGN.md section 7).
        reached_shortest=sum(case.get("moves", 0) == case.get("shortest_moves") for case in reached),
        successful_moves=sum(case.get("moves", 0) for case in cases if case.get("success")),
        successful_shortest_moves=sum(case.get("shortest_moves", 0) for case in cases if case.get("success")),
    )


def battery(root, task, name, manifest):
    directory = root / "evaluation" / task / name
    cases = load(directory)
    if not cases:
        return None
    split = split_of(directory.name)
    by_group = defaultdict(list)
    for case in cases:
        by_group[case["group"]].append(case)
    entry = dict(aggregate(cases, manifest))
    entry["counts"] = entry.pop("overall")
    entry.pop("strata", None)
    groups = {}
    for group, members in sorted(by_group.items()):
        summary = dict(aggregate(members, manifest))
        summary["counts"] = summary.pop("overall")
        summary.pop("strata", None)
        summary["physical_tasks"] = len({case["trajectory_id"] for case in members})
        summary["case_variants"] = len(members)
        summary["reachability_audit"] = audit(members)
        summary["rollout"] = rollout_of(members)
        groups[group] = summary
    entry["groups"] = groups
    entry["reachability_audit"] = audit(cases)
    entry["rollout"] = rollout_of(cases)
    shards = shards_of(directory)
    entry["sources"] = [shard.relative_to(root).as_posix() + "/summary.json" for shard in shards]
    entry["intervals"] = sorted([int(part) for part in shard.name.split("_")] for shard in shards)
    entry["unique_case_keys"] = len({(case["trajectory_id"], case.get("variant", 0)) for case in cases})
    entry["variants"] = len({case.get("variant", 0) for case in cases})
    entry["expected_physical_tasks"] = expected_tasks(task, split)
    need = math.ceil(entry["expected_physical_tasks"] / SHARD)
    entry["completion"] = "complete" if len(entry["sources"]) >= need else "partial"
    return entry


def percent(numerator, denominator):
    return "n/a" if not denominator else "%.1f%%" % (100.0 * numerator / denominator)


def show(task, name, entry):
    audit_ = entry["reachability_audit"]
    rollout = entry["rollout"]
    print("\n== %s/%s  [%s]" % (task, name, entry["completion"]))
    if rollout["attempts"]:
        print("   rollout: reached %s (%d/%d), reached+shortest %s (%d/%d)"
              % (percent(rollout["reached"], rollout["attempts"]), rollout["reached"], rollout["attempts"],
                 percent(rollout["reached_shortest"], rollout["attempts"]), rollout["reached_shortest"], rollout["attempts"]))
    if audit_.get("reachability_scored_turns"):
        scored = audit_["reachability_scored_turns"]
        print("   reference: keep-reachable %s (%d/%d)  floor %s"
              % (percent(audit_["keeps_goal_reachable"], scored), audit_["keeps_goal_reachable"], scored,
                 percent(audit_["uniform_expected_rate_current_reachable"], 1)))
        print("              env-shortest  %s (%d/%d)"
              % (percent(audit_["action_environment_shortest"], scored),
                 audit_["action_environment_shortest"], scored))
        print("              scored %d of %d solvable of %d decision turns (%d already unreachable, %d field missing)"
              % (scored, audit_["current_reachable_turns"], audit_["decision_turns"],
                 audit_["already_unreachable_turns"], audit_["missing_reachability_turns"]))
    for group, summary in sorted(entry["groups"].items()):
        roll = summary["rollout"]
        aud = summary["reachability_audit"]
        bits = []
        if roll["attempts"]:
            bits.append("reached %s" % percent(roll["reached"], roll["attempts"]))
            bits.append("reached+shortest %s" % percent(roll["reached_shortest"], roll["attempts"]))
        if aud["current_reachable_turns"]:
            bits.append("keep-reachable %s" % percent(aud["keeps_goal_reachable"], aud["current_reachable_turns"]))
            bits.append("floor %s" % percent(aud["uniform_expected_rate_current_reachable"], 1))
        print("      %-28s n=%-5d %s" % (group, roll["attempts"] or aud["current_reachable_turns"], "  ".join(bits)))


def checkpoint_epochs(root):
    """Map each saved checkpoint to its step and epoch, so the report's epoch names resolve.

    Intermediates are saved every third of an epoch (``checkpoint.every_epoch_fraction``), so
    three times the median gap between consecutive saved steps is one epoch. The epoch is
    written as a fraction -- 1.0, 2.0, 3.0 are the epoch boundaries the report quotes, and
    0.333/0.667 are the intermediate saves inside them.
    """
    out = {}
    for task in ("path", "blocks"):
        directory = root / task / "training" / "models"
        if not directory.is_dir():
            continue
        steps = sorted(int(path.name.split("-")[1]) for path in directory.iterdir()
                       if path.name.startswith("checkpoint-"))
        gaps = [b - a for a, b in zip(steps, steps[1:]) if b - a > 1]
        if not gaps:
            continue
        per_epoch = 3 * sorted(gaps)[len(gaps) // 2]
        names = [path.name for path in directory.iterdir()
                 if path.name.startswith("checkpoint-") or path.name == "final"]
        steps = {name: steps[-1] if name == "final" else int(name.split("-")[1]) for name in names}
        out[task] = {name: dict(step=step, epoch=round(step / per_epoch, 3))
                     for name, step in sorted(steps.items())}
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--update", type=Path, help="existing evidence file to refresh in place")
    parser.add_argument("--table", action="store_true", help="print the per-battery table")
    args = parser.parse_args()
    root = args.run_root

    batteries = {}
    manifests = {}
    for task in ("path", "blocks"):
        manifest = json.loads((root / task / "data" / "manifest.json").read_text())
        manifests[task] = manifest
        for directory in sorted((root / "evaluation" / task).glob("*/*")):
            if not directory.is_dir():
                continue
            entry = battery(root, task, directory.relative_to(root / "evaluation" / task).as_posix(), manifest)
            if entry is None:
                continue
            batteries["%s/%s" % (task, directory.relative_to(root / "evaluation" / task).as_posix())] = entry
            if args.table:
                show(task, directory.relative_to(root / "evaluation" / task).as_posix(), entry)

    # The two containers that share this run root; a completed shard is one path, so shard
    # paths -- not hosts -- are what the deduplication below counts.
    snapshot = dict(snapshot_utc=datetime.now(timezone.utc).isoformat(), run_root=str(root),
                    hosts=["172.16.78.10:41467", "172.16.78.10:35016"],
                    checkpoint_epochs=checkpoint_epochs(root),
                    deduplication="Both hosts share one run root; each completed shard path counted once.")
    if args.update:
        evidence = json.loads(args.update.read_text())
        evidence["snapshot"] = snapshot
        evidence["evaluation_batteries"] = batteries
        # sort_keys stays off: the file's existing key order is part of its readability, and
        # reordering it would bury the counts under a whole-file reformat in every diff.
        args.update.write_text(json.dumps(evidence, indent=2) + "\n")
        print("updated %s with %d batteries" % (args.update, len(batteries)))
    else:
        print(json.dumps(dict(snapshot=snapshot, evaluation_batteries=batteries), indent=2))


if __name__ == "__main__":
    main()
