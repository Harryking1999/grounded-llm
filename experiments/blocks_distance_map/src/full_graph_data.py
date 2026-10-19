"""Complete removal DAGs as products of connected-component DAGs.

Moves use connected tiles and only remove cells. Components cannot interact.
Covering source XOR goal is equivalent to removing that mask from the original
component, so one complete BFS also gives exact distances for every state pair.
"""
from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse
import json
import math
import os
from pathlib import Path

import numpy as np

from experiments.gcml_counterexamples.src import blocks
from .blocks12_eval import construct


def components(mask):
    result = []
    while mask:
        pending = [(mask & -mask).bit_length() - 1]
        part = 0
        while pending:
            cell = pending.pop()
            if part & (1 << cell):
                continue
            part |= 1 << cell
            row, col = divmod(cell, blocks.SIZE)
            for r, c in ((row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)):
                bit = 1 << (r * blocks.SIZE + c) if 0 <= r < blocks.SIZE and 0 <= c < blocks.SIZE else 0
                if mask & bit and not part & bit:
                    pending.append(r * blocks.SIZE + c)
        mask ^= part
        result.append(list(blocks.cells(part)))
    return result


def local_tiles(cells):
    part = sum(1 << cell for cell in cells)
    return sorted(sum(((tile >> cell) & 1) << i for i, cell in enumerate(cells))
                  for tile, _ in blocks.PLACEMENTS if tile & part == tile)


def build_graph(width, tiles, out, maximum_states):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    full = (1 << width) - 1
    states, depths = [full], [0]
    index = {full: 0}
    children, offsets = [], [0]
    for i, state in enumerate(states):
        for tile in tiles:
            if state & tile != tile:
                continue
            child = state ^ tile
            child_id = index.get(child)
            if child_id is None:
                if len(states) >= maximum_states:
                    raise RuntimeError(f"Component exceeds {maximum_states} states; graph is incomplete, never truncate or label unknown as unreachable: {out}")
                child_id = len(states)
                index[child] = child_id
                states.append(child)
                depths.append(depths[i] + 1)
            children.append(child_id)
        offsets.append(len(children))
    masks = np.asarray(states, dtype=np.uint64)
    removed = np.uint64(full) ^ masks
    order = np.argsort(removed)
    np.save(out / "states.npy", masks)
    np.save(out / "removed.npy", removed[order])
    np.save(out / "distances.npy", np.asarray(depths, dtype=np.int8)[order])
    np.save(out / "offsets.npy", np.asarray(offsets, dtype=np.int64))
    np.save(out / "children.npy", np.asarray(children, dtype=np.int32))
    summary = {"width": width, "states": len(states), "edges": len(children),
               "tiles": tiles, "complete": True}
    (out / "summary.json").write_text(json.dumps(summary) + "\n")
    return summary


def _build_job(job):
    from .symbolic_graph import build_symbolic
    gid, width, tiles, out, maximum = job
    result = build_symbolic(width, tiles, out, maximum)
    print(json.dumps({"completed_component": gid, **{k: result[k] for k in ("states", "edges")}}), flush=True)
    return gid, result


class Component:
    def __init__(self, path):
        path = Path(path)
        self.summary = json.loads((path / "summary.json").read_text())
        if not self.summary["complete"]:
            raise ValueError("Incomplete component graph")
        for name in ("states", "removed", "distances", "offsets", "children"):
            setattr(self, name, np.load(path / f"{name}.npy", mmap_mode="r"))
        self.size = len(self.states)
        self.initial_id, self.zero_id = 0, int(np.argmin(self.states))

    def close(self):
        for name in ("states", "removed", "distances", "offsets", "children"):
            getattr(self, name)._mmap.close()

    def distance(self, source, goal):
        source, goal = np.broadcast_arrays(np.asarray(source, dtype=np.uint64),
                                          np.asarray(goal, dtype=np.uint64))
        delta = source ^ goal
        index = np.searchsorted(self.removed, delta)
        safe = np.minimum(index, len(self.removed) - 1)
        valid = ((source & goal) == goal) & (index < len(self.removed)) & (self.removed[safe] == delta)
        return np.where(valid, self.distances[safe], -1).astype(np.int16)

    def degrees(self, ids):
        return self.offsets[ids + 1] - self.offsets[ids]

    def take(self, ids, offsets):
        return self.children[self.offsets[ids] + offsets]

    def star(self, ids):
        degree = self.degrees(ids)
        rows = np.repeat(np.arange(len(ids)), degree)
        offsets = np.arange(len(rows)) - np.repeat(np.cumsum(degree) - degree, degree)
        return rows, self.take(ids[rows], offsets)


def load_component(path):
    from .symbolic_graph import SymbolicComponent
    summary = json.loads((Path(path) / "summary.json").read_text())
    return (SymbolicComponent if summary.get("representation") == "symbolic_complete_BFS"
            else Component)(path)


class Root:
    def __init__(self, record, graphs):
        self.record = record
        self.parts = [(graphs[part["graph"]], part["cells"]) for part in record["components"]]
        self.size = math.prod(graph.size for graph, _ in self.parts)

    def decode(self, ids):
        values, result = np.asarray(ids, dtype=np.uint64).copy(), []
        for graph, _ in self.parts:
            result.append((values % graph.size).astype(np.int64))
            values //= graph.size
        return np.stack(result, axis=-1)

    def distance_parts(self, source, goal):
        return np.stack([graph.distance(graph.states[source[..., i]], graph.states[goal[..., i]])
                         for i, (graph, _) in enumerate(self.parts)], axis=-1)

    def distance(self, source, goal):
        values = self.distance_parts(self.decode(source), self.decode(goal))
        return np.where((values < 0).any(-1), -1, values.sum(-1)).astype(np.int16)

    def packed(self, local_ids):
        lo = np.zeros(local_ids.shape[:-1], dtype=np.uint64)
        hi = lo.copy()
        for i, (graph, cells) in enumerate(self.parts):
            masks = graph.states[local_ids[..., i]]
            for bit, cell in enumerate(cells):
                if cell < 64:
                    lo |= ((masks >> np.uint64(bit)) & np.uint64(1)) << np.uint64(cell)
                else:
                    hi |= ((masks >> np.uint64(bit)) & np.uint64(1)) << np.uint64(cell - 64)
        return np.stack((lo, hi), axis=-1)

    def advance(self, local_ids, rng, steps):
        """Exact random legal transitions, including all branch choices."""
        current = local_ids.copy()
        limits = np.broadcast_to(steps, current.shape[:-1])
        for move in range(int(np.max(limits, initial=0))):
            degrees = np.stack([g.degrees(current[:, i])
                                for i, (g, _) in enumerate(self.parts)], -1)
            total = degrees.sum(-1)
            active = (total > 0) & (move < limits)
            if not active.any():
                break
            ids = np.flatnonzero(active)
            choice = (rng.random(len(ids)) * total[ids]).astype(np.int64)
            ends = np.cumsum(degrees[ids], axis=-1)
            selected = (choice[:, None] >= ends).sum(-1)
            starts = np.column_stack((np.zeros(len(ids), dtype=np.int64), ends[:, :-1]))
            offset = choice - starts[np.arange(len(ids)), selected]
            for i, (graph, _) in enumerate(self.parts):
                keep = np.flatnonzero(selected == i)
                rows = ids[keep]
                current[rows, i] = graph.take(current[rows, i], offset[keep])
        return current


def prepare(config, out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    (out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    excluded = {r["initial"] for r in construct(config["evaluation_boards"], 12, config["evaluation_seed"])}
    candidates = construct(config["train_boards"] + config["evaluation_boards"], 12, config["seed"])
    roots, specs, catalog = [], [], {}
    reuse = Path(config["reuse_graph_data"]) if config.get("reuse_graph_data") else None
    reused = {}
    previous_roots = []
    if reuse:
        previous = json.loads((reuse / "summary.json").read_text())
        if not previous["complete"]:
            raise ValueError("Cannot reuse incomplete exact graphs")
        previous_roots = json.loads((reuse / "roots.json").read_text())
        for width, tiles in json.loads((reuse / "graph_specs.json").read_text()):
            catalog[(width, tuple(tiles))] = len(specs)
            specs.append((width, tiles))
        for i in range(len(specs)):
            summary_path = reuse / "graphs" / str(i) / "summary.json"
            if summary_path.exists() or not config.get("rebuild_missing_components", False):
                reused[i] = json.loads(summary_path.read_text())
    for record in candidates:
        if record["initial"] in excluded:
            continue
        parts = []
        for cells in components(int(record["initial"])):
            tiles = local_tiles(cells)
            signature = (len(cells), tuple(tiles))
            if signature not in catalog:
                catalog[signature] = len(specs)
                specs.append((len(cells), tiles))
            parts.append({"graph": catalog[signature], "cells": cells})
        roots.append({**record, "components": parts})
        if len(roots) == config["train_boards"]:
            break
    if len(roots) != config["train_boards"]:
        raise ValueError("Insufficient distinct training roots")
    if previous_roots and roots[:len(previous_roots)] != previous_roots:
        raise ValueError("Expansion must preserve the exact previous root/index prefix")
    if config.get("forbidden_roots"):
        frozen = json.loads(Path(config["forbidden_roots"]).read_text())
        heldout = {int(r["source"]) for r in frozen["cases"]}
        if any(int(r["initial"]) in heldout for r in roots):
            raise ValueError("Fixed clearing benchmark roots must not enter training")
    action_counts = [sum(int(r["initial"]) & tile == tile for r in roots)
                     for tile, _ in blocks.PLACEMENTS]
    if not all(action_counts):
        raise ValueError("Training roots do not cover every original placement action")
    (out / "roots.json").write_text(json.dumps(roots) + "\n")
    (out / "graph_specs.json").write_text(json.dumps(specs) + "\n")
    (out / "graphs").mkdir()
    for i, summary in reused.items():
        if not summary["complete"]:
            raise ValueError("Incomplete reused component")
        os.symlink((reuse / "graphs" / str(i)).resolve(), out / "graphs" / str(i), target_is_directory=True)
    jobs = [(i, width, tiles, out / "graphs" / str(i), config["maximum_diagram_nodes"])
            for i, (width, tiles) in enumerate(specs) if i not in reused]
    summaries = dict(reused)
    with ProcessPoolExecutor(max_workers=config["workers"]) as executor:
        futures = [executor.submit(_build_job, job) for job in jobs]
        for future in as_completed(futures):
            i, summary = future.result()
            summaries[i] = summary
            if len(summaries) % 100 == 0 or len(summaries) == len(specs):
                print(json.dumps({"complete_components": len(summaries), "total_components": len(specs),
                                  "reused_components": len(reused)}), flush=True)
    counts, pairs, edges = [], [], []
    for root in roots:
        parts = [summaries[p["graph"]] for p in root["components"]]
        size = math.prod(p["states"] for p in parts)
        counts.append(size)
        pairs.append(size * size)
        edges.append(sum(p["edges"] * (size // p["states"]) for p in parts))
    summary = {"complete": True, "train_roots": len(roots), "component_graphs": len(specs),
               "distinct_initial_masks": len({r["initial"] for r in roots}),
               "distinct_root_component_structures": len({tuple(sorted(p["graph"] for p in r["components"])) for r in roots}),
               "reused_roots": len(previous_roots), "reused_component_graphs": len(reused),
               "component_states": sum(s["states"] for s in summaries.values()),
               "root_state_counts": counts, "root_ordered_pair_counts": pairs,
               "root_directed_edge_counts": edges,
               "total_root_local_states": sum(counts), "total_root_local_pairs": sum(pairs),
               "total_root_local_edges": sum(edges),
               "actions_available": len(action_counts),
               "action_training_root_counts": action_counts,
               "representation": "symbolic_complete_BFS_component_products",
               "scope": "Complete Cartesian-product removal DAG for every selected twelve-placement root; exact arbitrary-pair lookup. Counts are root-local and may overlap across roots. Available relations are not actual training exposure.",
               "evaluation_root_overlap": 0}
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if not k.startswith("root_")}), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    prepare(json.loads(args.config.read_text()), args.out)


if __name__ == "__main__":
    main()
