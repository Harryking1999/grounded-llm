"""Prepare broad, reproducible pass@8 suites and retain exact offline diagnostics."""
import argparse
from collections import deque
import copy
import json
from pathlib import Path
import random
import sys

import blocks
import path as path_task

ROOT = Path(__file__).resolve().parents[3]


def load_official(config, rng):
    # h5py is needed only to extract the original data, not to judge/run the suite.
    try:
        import h5py
    except ImportError:
        sys.path.insert(0, str(ROOT / "literature_review/gcml_deps"))
        import h5py
    old = json.loads((ROOT / config["previous_blocks_run"]).read_text(encoding="utf-8-sig"))
    seen = {blocks.normalized_key(blocks.from_grid(case["input_grid"])) for case in old["cases"]}
    first, end = config["blocks"]["official_rows"]
    indices = list(range(first, end))
    rng.shuffle(indices)
    pool = []
    with h5py.File(ROOT / config["official_dataset"], "r") as f:
        group = f["tiling"]
        for row in indices:
            flat = group["pre_obs"][row, 0].astype(int)
            mask = sum(1 << i for i, x in enumerate(flat) if x)
            key = blocks.normalized_key(mask)
            if key in seen:
                continue
            seen.add(key)
            ids = group["action"][row].astype(int).tolist()
            reference = [blocks.PLACEMENTS[i][1] for i in ids]
            current = mask
            for action in reference:
                current = blocks.apply(current, action)
            if current:
                raise ValueError(f"Official witness does not clear row {row}")
            pool.append({"id": f"blocks8_row{row}", "condition": "blocks8",
                         "grid": blocks.to_grid(mask), "budget": 8,
                         "source_hdf5_row": row, "construction_objects": 8,
                         "construction_reference": reference})
            if len(pool) == config["blocks"]["official_pool_size"]:
                break
    if len(pool) != config["blocks"]["official_pool_size"]:
        raise ValueError("Insufficient independent official silhouettes")
    return pool, seen


def neighboring_cells(mask):
    result = 0
    for i in blocks.cells(mask):
        row, col = divmod(i, blocks.SIZE)
        for r, c in ((row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)):
            if 0 <= r < blocks.SIZE and 0 <= c < blocks.SIZE:
                result |= 1 << (r * blocks.SIZE + c)
    return result & ~mask


def generate_blocks(config, rng, seen):
    options = config["blocks"]
    count, objects = options["generated_pool_size"], options["generated_objects"]
    pool = []
    attempts = 0
    while len(pool) < count:
        attempts += 1
        if attempts > count * 100:
            raise ValueError("Unable to generate enough independent connected silhouettes")
        mask, additions = 0, []
        for _ in range(objects):
            boundary = neighboring_cells(mask)
            candidates = [i for i, (tile, _) in enumerate(blocks.PLACEMENTS)
                          if tile & mask == 0 and (not mask or tile & boundary)]
            if not candidates:
                break
            index = rng.choice(candidates)
            mask |= blocks.PLACEMENTS[index][0]
            additions.append(blocks.PLACEMENTS[index][1])
        key = blocks.normalized_key(mask)
        if len(additions) != objects or key in seen:
            continue
        seen.add(key)
        current = mask
        for action in additions:
            current = blocks.apply(current, action)
        if current:
            raise AssertionError("Generated witness failed")
        pool.append({"id": f"blocks{objects}_generated{len(pool):03d}",
                     "condition": f"blocks{objects}", "grid": blocks.to_grid(mask),
                     "budget": objects, "construction_objects": objects,
                     "generator": "uniform placement among nonoverlapping adjacent placements",
                     "construction_reference": additions})
    return pool


def graph_distances(neighbors, start):
    distances, queue = {start: 0}, deque([start])
    while queue:
        node = queue.popleft()
        for other in neighbors[str(node)]:
            if other not in distances:
                distances[other] = distances[node] + 1
                queue.append(other)
    return distances


def prepare_paths(config, rng):
    graph = json.loads((ROOT / config["graph_config"]).read_text())
    neighbors = graph["neighbors"]
    pairs = sorted({(p["start"], p["goal"]) for p in graph["pairs"]})
    candidates = [(a, b) for a in range(32) for b in range(32) if a != b and (a, b) not in pairs]
    rng.shuffle(candidates)
    pairs += candidates[:config["path"]["shortest_pair_count"] - len(pairs)]
    cases = []
    for start, goal in pairs:
        case = {"id": f"path_shortest_{start}_{goal}", "condition": "path_shortest",
                "objective": "shortest", "start": start, "goal": goal,
                "neighbors": neighbors, "bits": 0, "initial_mask": 0,
                "gates": [], "switches": [], "step_limit": None,
                "replicates": config["path"]["shortest_replicates"],
                "previously_tested_pair": any(p["start"] == start and p["goal"] == goal for p in graph["pairs"])}
        case["reference"] = path_task.shortest(case)
        cases.append(case)
    edges = [(a, b) for a in range(32) for b in neighbors[str(a)] if a < b]
    used = set()
    for bits in config["path"]["gate_bits"]:
        found = 0
        for attempt in range(config["path"]["max_layout_attempts"]):
            start, goal = rng.sample(range(32), 2)
            distances = graph_distances(neighbors, start)
            distance = distances[goal]
            if distance < 2 * bits + 1:
                continue
            cuts = [2] if bits == 1 else [2, 4]
            gates = [{"edge": [a, b], "bit": bit} for bit, cut in enumerate(cuts)
                     for a, b in edges if (distances[a] < cut) != (distances[b] < cut)]
            switches = []
            for bit, cut in enumerate(cuts):
                lower = 0 if bit == 0 else cuts[bit - 1]
                nodes = [node for node, d in distances.items() if lower <= d < cut and node != start]
                switches.append({"node": rng.choice(nodes), "bit": bit})
            identity = (start, goal, tuple(s["node"] for s in switches))
            if identity in used:
                continue
            paired = []
            for mask in config["path"]["initial_masks"]:
                case = {"id": f"path_gates{bits}_layout{found}_mask{mask}",
                        "layout_id": f"gates{bits}_layout{found}",
                        "condition": f"path_gates{bits}", "objective": "shortest",
                        "start": start, "goal": goal, "neighbors": neighbors,
                        "bits": bits, "initial_mask": mask, "gates": gates, "switches": switches,
                        "step_limit": None, "replicates": config["path"]["gate_replicates"],
                        "base_graph_distance": distance}
                case["reference"] = path_task.shortest(case)
                paired.append(case)
            if any(c["reference"] is None or c["reference"]["length"] > config["path"]["gate_reference_length_limit"] for c in paired):
                continue
            if paired[0]["reference"]["length"] < distance + config["path"]["minimum_closed_gate_detour"]:
                continue
            if paired[0]["reference"]["path"] == paired[1]["reference"]["path"]:
                continue
            # At least one closed-mask optimal witness revisits a physical node;
            # its complete state changes, so a node-only visited set is wrong.
            nodes = [s["node"] for s in paired[0]["reference"]["states"]]
            if len(nodes) == len(set(nodes)):
                continue
            used.add(identity)
            cases.extend(paired)
            found += 1
            if found == config["path"]["gate_layouts_per_bit_count"]:
                break
        if found != config["path"]["gate_layouts_per_bit_count"]:
            raise ValueError(f"Insufficient gate layouts for {bits} bits")
    return cases


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="experiments/gcml_counterexamples/configs/pilot.json")
    parser.add_argument("--out", default="experiments/gcml_counterexamples/runs/pilot/suite.json")
    args = parser.parse_args()
    config = json.loads((ROOT / args.config).read_text())
    rng = random.Random(config["seed"])
    official, seen = load_official(config, rng)
    generated = generate_blocks(config, rng, seen)
    # Select before computing diagnostic scores. Pool order is randomized by
    # extraction/generation, and samples are drawn using the committed seed.
    selected = rng.sample(official, config["blocks"]["random8_count"])
    selected += rng.sample(generated, config["blocks"]["random12_count"])
    selected_ids = {c["id"] for c in selected}
    for index, case in enumerate(official + generated):
        case["diagnostics"] = blocks.diagnose(
            blocks.from_grid(case["grid"]), case["budget"], random.Random(config["seed"] + 10000 + index),
            config["blocks"]["proxy_rollouts"], config["blocks"]["latent_prefix_cap"])
        case["selected_for_api"] = case["id"] in selected_ids
        case["replicates"] = config["blocks"]["replicates"]
        if (index + 1) % 16 == 0:
            print(f"Diagnosed {index + 1}/{len(official) + len(generated)} boards", flush=True)
    paths = prepare_paths(config, random.Random(config["seed"] + 20000))
    cases = selected + paths
    calls = sum(c["replicates"] for c in cases)
    if calls > config["api"]["max_calls"]:
        raise ValueError("Suite exceeds committed call budget")
    payload = {"config": config, "config_path": args.config,
               "case_count": len(cases), "planned_calls": calls,
               "selection_note": "Block scores were computed only after random selection. Gate layouts are structural challenge cases, paired across initial masks.",
               "cases": cases, "offline_blocks_pool": official + generated}
    output = ROOT / args.out
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "cases": len(cases), "calls": calls,
                      "official_pool": len(official), "generated_pool": len(generated)}), flush=True)


if __name__ == "__main__":
    main()
