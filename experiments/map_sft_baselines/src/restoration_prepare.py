"""Long, broad, certified map-greedy demonstrations for fixed-goal restoration."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import heapq
import json
import multiprocessing
from pathlib import Path
import shutil
import numpy as np
import torch
from transformers import AutoTokenizer
from grounded_llm.core.files import atomic_json
from grounded_llm.core.sft.transcript import encode_trajectory
from grounded_llm.tasks.restoration_sft import environment, load_map, make_step, demonstration

WORKER = None


def route(start, successors, depth, policy=None):
    states, actions = [int(start)], []
    while states[-1] != 0:
        state = states[-1]
        children = successors[state]
        if policy is None:
            choices = np.flatnonzero((children >= 0) & (depth[np.maximum(children, 0)] == depth[state] - 1))
            action = int(choices[0])
        else:
            action = int(policy[state])
        child = int(children[action])
        if child < 0 or depth[child] != depth[state] - 1:
            raise ValueError("Trajectory is not a shortest path")
        actions.append(action)
        states.append(child)
    return states, actions


def select_routes(eligible, depth, successors, policy, count, seed):
    """Exact length priority; lazy set-cover tie breaking within a depth."""
    rng = np.random.default_rng(seed)
    selected, covered = [], set()
    for length in sorted(np.unique(depth[eligible]), reverse=True):
        candidates = rng.permutation(np.flatnonzero(eligible & (depth == length)))
        heap = [(-int(length), tie, int(state)) for tie, state in enumerate(candidates)]
        heapq.heapify(heap)
        while heap and len(selected) < count:
            old_gain, tie, start = heapq.heappop(heap)
            states, actions = route(start, successors, depth, policy)
            gain = len(set(states[:-1]) - covered)
            entry = (-gain, tie, start)
            if heap and entry > heap[0]:
                heapq.heappush(heap, entry)
                continue
            selected.append((states, actions))
            covered.update(states[:-1])
        if len(selected) == count:
            break
    if len(selected) != count:
        raise ValueError(f"Only {len(selected)} qualified distinct starts, need {count}")
    return selected, covered


def prepare_worker(item):
    index, record, states = item
    cfg, env, vectors, distances, tokenizer, out = WORKER
    steps = [make_step(env, None, state, vectors=vectors, distances=distances) for state in states]
    demo = demonstration(cfg["task"], env, record, steps)
    encoded = encode_trajectory(demo, tokenizer, cfg["maximum_sequence_tokens"])
    name = f"{index:05d}.pt"
    torch.save({"map": {"encoded": encoded, "steps": steps}}, out / "trajectories" / name)
    return dict(record, prepared_file=name, tokens=len(encoded.input_ids))


def prepare(args):
    global WORKER
    cfg = json.loads(args.config.read_text())
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    if (out / "ready.json").exists():
        raise ValueError("Accepted dataset already exists")
    torch.set_num_threads(2)
    env = environment(cfg["task"], out / "oracle.npy")
    if cfg["task"] == "cube2":
        env.build_oracle(out / "oracle.npy")
    depth = env.distance(np.arange(env.size), 0)
    successors = env.successors(np.arange(env.size))
    rng = np.random.default_rng(cfg["seed"])
    # Lock the cohort before using Q. It includes Q failures without filtering.
    heldout = rng.choice(np.arange(1, env.size), cfg["evaluation"]["tasks"], replace=False)
    heldout_set = set(map(int, heldout))
    evaluation = []
    for index, start in enumerate(heldout):
        states, actions = route(start, successors, depth)
        evaluation.append(dict(trajectory_id=f"{cfg['task']}_eval_{index:04d}",
            start=int(start), goal=0, reference_actions=[env.actions[a] for a in actions],
            length=len(actions)))
    atomic_json(out / "evaluation.json", evaluation)
    atomic_json(out / "probe_indices.json", sorted(map(int, rng.choice(
        len(evaluation), cfg["evaluation"]["probe_tasks"], replace=False))))
    reader = load_map(cfg["task"], env, args.q_checkpoint, args.device)
    vectors = np.lib.format.open_memmap(out / "vectors.npy", mode="w+", dtype="float32",
                                       shape=(env.size, cfg["map"]["state_dim"]))
    distances = np.empty(env.size, dtype=np.float32)
    goal_q = reader.encode([0])
    for begin in range(0, env.size, 8192):
        end = min(begin + 8192, env.size)
        q = reader.encode(np.arange(begin, end))
        vectors[begin:end] = q.cpu().numpy()
        distances[begin:end] = reader.distance(q, goal_q).cpu().numpy()
        if begin % 262144 == 0:
            atomic_json(out / "progress.json", dict(stage="encode_q", done=end, total=env.size))
    vectors.flush()
    np.save(out / "q_distances.npy", distances)
    del reader, goal_q, q
    torch.cuda.empty_cache()
    scores = distances[np.maximum(successors, 0)].copy()
    scores[successors < 0] = np.inf
    policy = scores.argmin(axis=1)
    greedy_next = successors[np.arange(env.size), policy]
    valid = np.zeros(env.size, dtype=bool)
    valid[0] = True
    for length in range(1, int(depth.max()) + 1):
        ids = np.flatnonzero(depth == length)
        child = greedy_next[ids]
        valid[ids] = (depth[child] == length - 1) & valid[child]
    # No held-out initial state occurs anywhere in a training trajectory.
    eligible = np.zeros(env.size, dtype=bool)
    eligible[0] = True
    for length in range(1, int(depth.max()) + 1):
        ids = np.flatnonzero((depth == length) & valid)
        eligible[ids] = eligible[greedy_next[ids]]
        eligible[list(heldout_set)] = False
    eligible &= depth >= cfg["data"]["minimum_shortest_moves"]
    ordinary = cfg["data"]["samples_per_epoch"]
    atomic_json(out / "progress.json", dict(stage="select", qualified=int(eligible.sum()),
        needed=ordinary, map_shortest_states=int(valid.sum())))
    selected, covered = select_routes(eligible, depth, successors, policy, ordinary, cfg["seed"])
    items = []
    for index, (states, actions) in enumerate(selected):
        record = dict(trajectory_id=f"{cfg['task']}_train_{index:05d}", start=states[0], goal=0,
                      reference_actions=[env.actions[a] for a in actions], length=len(actions))
        items.append((index, record, states))
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    (out / "trajectories").mkdir(exist_ok=True)
    WORKER = (cfg, env, vectors, distances, tokenizer, out)
    rows = []
    context = multiprocessing.get_context("fork")
    with ProcessPoolExecutor(args.workers, mp_context=context) as executor:
        for row in executor.map(prepare_worker, items, chunksize=16):
            rows.append(row)
            if len(rows) % 256 == 0:
                atomic_json(out / "progress.json", dict(stage="tokenize", done=len(rows), total=len(items)))
    if covered & heldout_set:
        raise ValueError("Held-out initial state entered training")
    atomic_json(out / "dataset.json", dict(task=cfg["task"], response_order="ranking_first", rows=rows))
    shutil.copy2(args.q_checkpoint, out / "qmap.pt")
    audit = dict(samples=len(rows), ordinary_unique_starts=ordinary,
        initial_terminal_samples=0, terminal_turns=len(rows),
        nonterminal_state_goal_coverage=len(covered), reachable_nonterminal_states=env.size - 1,
        eligible_starts=int(eligible.sum()), shortest_map_greedy_verified=True,
        heldout_initial_overlap=0, evaluation_tasks=len(evaluation),
        length_histogram=dict(Counter(len(actions) for _, actions in selected)),
        maximum_tokens=max(row["tokens"] for row in rows), scope="fixed_goal_restoration",
        goal_id=0, q_selection=cfg["q_selection"])
    atomic_json(out / "acceptance.json", audit)
    atomic_json(out / "ready.json", dict(samples=len(rows), smoke=False))
    atomic_json(out / "progress.json", dict(stage="completed", **audit))


def main():
    parser = argparse.ArgumentParser()
    for name in ("config", "out", "q-checkpoint", "model-path"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    prepare(parser.parse_args())


if __name__ == "__main__":
    main()
