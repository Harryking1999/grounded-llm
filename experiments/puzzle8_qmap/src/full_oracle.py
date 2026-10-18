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


def tensor_bfs(adjacency, goals, size, diameter):
    """Batched exact BFS; tensor arithmetic only changes oracle throughput."""
    import torch
    columns = torch.arange(len(goals), device=goals.device)
    distance = torch.full((size, len(goals)), -1, dtype=torch.int8, device=goals.device)
    frontier = torch.zeros((size, len(goals)), device=goals.device)
    frontier[goals, columns] = 1.
    distance[goals, columns] = 0
    for depth in range(1, diameter + 1):
        following = (torch.sparse.mm(adjacency, frontier) > 0) & (distance < 0)
        distance[following] = depth
        frontier = following.float()
    if bool((distance < 0).any()):
        raise ValueError("GPU BFS did not cover the connected component")
    return distance.T.contiguous().cpu().numpy()


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--resume-gpu", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if config["task"] != "puzzle8" or config["coverage"] != "all_ordered_source_goal_pairs":
        raise ValueError("Expected the complete Puzzle8 pair-distance contract")
    args.out.mkdir(parents=True, exist_ok=args.resume_gpu)
    started = time.monotonic()
    env = Puzzle8()
    rows, actions = np.where(env.next >= 0)
    graph = csr_matrix((np.ones(len(rows), dtype=np.int8), (rows, env.next[rows, actions])),
                       shape=(env.size, env.size))
    path = args.out / "distances.npy"
    if args.resume_gpu:
        import torch
        torch.set_num_threads(1)
        table = np.load(path, mmap_mode="r+")
        if table.shape != (env.size, env.size) or table.dtype != np.int8:
            raise ValueError("Resume table does not match this exact graph")
        previous = json.loads((args.out / "progress.json").read_text())
        begin = completed = previous["complete_goal_rows"]
        device = torch.device("cuda")
        adjacency = torch.sparse_csr_tensor(torch.as_tensor(graph.indptr, device=device),
                    torch.as_tensor(graph.indices, device=device),
                    torch.as_tensor(graph.data, dtype=torch.float32, device=device), size=graph.shape)
        for row in range(begin, env.size, config["gpu_goal_batch"]):
            end = min(row + config["gpu_goal_batch"], env.size)
            table[row:end] = tensor_bfs(adjacency, torch.arange(row, end, device=device), env.size, env.diameter)
            completed = end
            elapsed = time.monotonic() - started
            progress = {"complete_goal_rows": completed, "total_goal_rows": env.size,
                        "seconds": round(previous["seconds"] + elapsed, 2),
                        "ordered_pairs": completed * env.size, "completion_device": "cuda"}
            (args.out / "progress.json").write_text(json.dumps(progress, indent=2) + "\n")
            if completed == env.size or (row - begin) % (16 * config["gpu_goal_batch"]) == 0:
                print(json.dumps(progress), flush=True)
        table.flush()
        summary = {"task": "puzzle8", "states": env.size, "shape": [env.size, env.size], "dtype": "int8",
                   "complete": True, "ordered_pairs": env.size ** 2, "label_bytes": env.size ** 2,
                   "source_commit": args.source_commit, "goal_state": list(env.goal_state),
                   "move": "blank_direction", "move_cost": 1, "cpu_prefix_rows": begin,
                   "gpu_completion_rows": env.size - begin, "gpu_completion_seconds": round(time.monotonic() - started, 2)}
        (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps(summary), flush=True)
        return
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
