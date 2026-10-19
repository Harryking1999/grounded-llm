"""Freeze 40 reproducible cases per task, prompts and offline reference witnesses."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import random
import subprocess

import numpy as np

import grounded_llm.tasks.blocks.rules as blocks
from experiments.qwen_path_blocks.src.prepare_path_suite import regular_connected_graph, shortest
from . import cube2, hanoi, puzzle8
from grounded_llm.tasks.text import ROOT, prompt, judge


def choose_by_depth(table, depths, count, rng):
    buckets = defaultdict(list)
    for state, depth in table.items():
        if depth in depths:
            buckets[depth].append(state)
    return [(state, depth) for depth in depths for state in rng.sample(buckets[depth], count)]


def make_blocks(options, count, rng):
    cases, seen = [], set()
    custom = options.get('shape_rows')
    if custom:
        from experiments.sol_dag_blocks.src.tasks import BlocksTask
        rules = BlocksTask(options)
    placements = rules.placements if custom else blocks.PLACEMENTS
    shape_count = len(rules.shapes) if custom else len(blocks.SHAPES)
    while len(cases) < count:
        state, witness = 0, []
        for _ in range(options["placements"]):
            shape = rng.randrange(shape_count)
            candidates = [(mask, action) for mask, action in placements if action["shape_id"] == shape and not mask & state]
            if not candidates:
                break
            mask, action = rng.choice(candidates)
            state |= mask
            witness.append(action)
        if len(witness) != options["placements"] or state in seen:
            continue
        seen.add(state)
        if custom:
            minimum, reference = None,witness
        else:
            exact = blocks.solver()
            minimum, indices = exact(state)
            assert minimum is not None
            reference = [blocks.PLACEMENTS[i][1] for i in indices]
        cases.append({"task": "blocks", "initial": state, "goal": 0,
                      "stratum": f"placements_{options['placements']}", "generation_placements": len(witness),
                      **({'shape_options':options} if custom else {}),
                      "reference": {"length": minimum, "actions": reference,
                                    **({'upper_bound':len(witness)} if custom else {})}})
        detail = f"reference upper bound={len(witness)}" if custom else f"exact minimum={minimum}"
        print(f"prepared blocks {len(cases)}/{count}, {detail}", flush=True)
    return cases


def make_path(options, rng):
    graph = regular_connected_graph(options["node_count"], options["regular_degree"], random.Random(options["graph_seed"]))
    buckets = defaultdict(list)
    for source in range(options["node_count"]):
        for target in range(source + 1, options["node_count"]):
            route = shortest(graph, source, target)
            if len(route) in options["distances"]:
                buckets[len(route)].append((source, target, route))
    neighbors = {str(n): sorted(values) for n, values in graph.items()}
    result = []
    for depth in options["distances"]:
        for source, target, route in rng.sample(buckets[depth], options["per_distance"]):
            result.append({"task": "path", "initial": source, "goal": target,
                           "stratum": f"distance_{depth}", "neighbors": neighbors,
                           "reference": {"length": depth, "actions": route}})
    return result


def make_hanoi(options, rng):
    goal = (ord(options["goal_peg"]) - 65,) * options["disks"]
    table = hanoi.distances(goal)
    result = []
    for low, high in options["distance_ranges"]:
        pool = [state for state, depth in table.items() if low <= depth <= high]
        for state in rng.sample(pool, options["per_range"]):
            result.append({"task": "hanoi", "initial": list(state), "goal": list(goal),
                           "stratum": f"distance_{low}_{high}",
                           "reference": {"length": table[state], "actions": hanoi.solve(state, table)}})
    return result


def make_cube(options, out, rng):
    env = cube2.environment()
    env.build_oracle(out / "cube2_qtm_depth.npy")
    result = []
    for depth in options["distances"]:
        pool = np.flatnonzero(env.depth == depth).tolist()
        for state in rng.sample(pool, options["per_distance"]):
            result.append({"task": "cube2", "initial": state, "goal": options["goal"],
                           "stratum": f"distance_{depth}",
                           "reference": {"length": depth, "actions": env.solve(state, options["goal"])}})
    return result


def make_puzzle(options, rng):
    goal = tuple(options["goal"])
    table = puzzle8.distances(goal)
    return [{"task": "puzzle8", "initial": list(state), "goal": list(goal),
             "stratum": f"distance_{depth}",
             "reference": {"length": depth, "actions": puzzle8.solve(state, table)}}
            for state, depth in choose_by_depth(table, options["distances"], options["per_distance"], rng)]


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "suite.json").exists():
        raise FileExistsError("Suite already frozen; choose a new output directory")
    rng, cases = random.Random(config["seed"]), []
    for task, factory in (("blocks", make_blocks), ("path", make_path), ("hanoi", make_hanoi), ("cube2", make_cube), ("puzzle8", make_puzzle)):
        options = config["tasks"][task]
        group = factory(options, config["cases_per_task"], rng) if task == "blocks" else factory(options, out, rng) if task == "cube2" else factory(options, rng)
        assert len(group) == config["cases_per_task"]
        for number, case in enumerate(group):
            case["id"] = f"{task}_{number:02d}"
            case["prompt"] = prompt(case)
            verdict = judge(case, json.dumps({"actions": case["reference"]["actions"], "final_status": "solved"}))
            assert verdict["contract_pass"] and verdict["optimal"], (case["id"], verdict)
        assert len({json.dumps([c["initial"], c["goal"]]) for c in group}) == len(group)
        cases.extend(group)
        print(json.dumps({"task": task, "cases": len(group), "strata": dict(Counter(c["stratum"] for c in group))}), flush=True)
    suite = {"config": config, "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(), "cases": cases}
    (out / "suite.json").write_text(json.dumps(suite, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Frozen {len(cases)} cases with replay-verified reference solutions: {out / 'suite.json'}", flush=True)


if __name__ == "__main__":
    main()
