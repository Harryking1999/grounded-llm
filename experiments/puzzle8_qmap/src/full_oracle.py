"""Construct every exact Puzzle8 source-goal distance, outside Git."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import time
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
from .environment import Puzzle8


def initialize(graph, path):
    global GRAPH, TABLE
    GRAPH = graph
    TABLE = np.load(path, mmap_mode="r+")


def fill(bounds):
    begin, end = bounds
    values = shortest_path(GRAPH, directed=False, unweighted=True, indices=np.arange(begin, end))
    if values.min() < 0 or values.max() > 31 or not np.isfinite(values).all():
        raise ValueError("Unexpected disconnected component or distance range")
    if np.any(values[np.arange(end - begin), np.arange(begin, end)] != 0):
        raise ValueError("Nonzero diagonal in exact graph distances")
    TABLE[begin:end] = values.astype(np.int8)
    return end - begin


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if config["task"] != "puzzle8" or config["coverage"] != "all_ordered_source_goal_pairs":
        raise ValueError("Expected the complete Puzzle8 pair-distance contract")
    args.out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    env = Puzzle8()
    rows, actions = np.where(env.next >= 0)
    graph = csr_matrix((np.ones(len(rows), dtype=np.int8), (rows, env.next[rows, actions])),
                       shape=(env.size, env.size))
    path = args.out / "distances.npy"
    table = np.lib.format.open_memmap(path, mode="w+", dtype=np.int8, shape=(env.size, env.size))
    del table
    chunk = config["goal_rows_per_chunk"]
    bounds = [(b, min(b + chunk, env.size)) for b in range(0, env.size, chunk)]
    completed, last_report = 0, 0.
    with ProcessPoolExecutor(config["workers"], initializer=initialize, initargs=(graph, path)) as pool:
        for count in pool.map(fill, bounds):
            completed += count
            elapsed = time.monotonic() - started
            if elapsed - last_report >= 60 or completed == env.size:
                last_report = elapsed
                progress = {"complete_goal_rows": completed, "total_goal_rows": env.size,
                            "seconds": round(elapsed, 2), "ordered_pairs": completed * env.size}
                (args.out / "progress.json").write_text(json.dumps(progress, indent=2) + "\n")
                print(json.dumps(progress), flush=True)
    np.load(path, mmap_mode="r+").flush()
    summary = {"task": "puzzle8", "states": env.size, "shape": [env.size, env.size],
               "dtype": "int8", "complete": True, "ordered_pairs": env.size ** 2,
               "label_bytes": env.size ** 2, "source_commit": args.source_commit,
               "goal_state": list(env.goal_state), "move": "blank_direction", "move_cost": 1,
               "seconds": round(time.monotonic() - started, 2)}
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
