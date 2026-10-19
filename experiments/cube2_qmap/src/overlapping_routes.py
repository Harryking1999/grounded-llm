"""Relational multi-goal maps: physical overlap, retention and transfer."""
import argparse
import copy
import json
from pathlib import Path
import time

import numpy as np
import torch

from grounded_llm.tasks.cube2.relation_data import pair_keys
from grounded_llm.tasks.cube2.model import MODEL_TYPE, QMap
from grounded_llm.tasks.cube2.data import pose_groups
from grounded_llm.tasks.cube2.environment import Cube2
from .route_complete import actual_update, losses, metrics, run_model
from .structure_study import load_contract


def pool_rows(env, base, config, forbidden, kind):
    options = config["data"]
    rng = np.random.default_rng(options["seed"] + (kind == "disjoint"))
    rows = [base["poses_fit"][0].copy()]
    parents = np.unique(base["tasks"][:, 0])
    goals = np.unique(base["tasks"][:, 1])
    for _ in range(1, options["collections"]):
        occupied = np.unique(np.concatenate(rows))
        known_goals = np.unique(np.concatenate([row[goals] for row in rows]))
        if kind == "disjoint":
            exclude = forbidden.copy()
            exclude[occupied] = True
            row = pose_groups(env, base["states"][None], exclude, rng, options["maximum_pose_attempts"])[0]
        else:
            for attempt in range(options["maximum_pose_attempts"]):
                old = rows[int(rng.integers(len(rows)))]
                target = int(old[int(rng.choice(parents))])
                source = int(base["states"][int(rng.choice(parents))])
                # k * source = target. Align an actual parent, which also
                # aligns all six of its right-action neighbors.
                transform = env.compose(target, env.relative(source, 0))
                row = env.compose(transform, base["states"])
                shared = np.isin(row, occupied)
                new_goals = (~np.isin(row[goals], known_goals)).sum()
                if (not forbidden[row].any() and len(np.intersect1d(row, old)) >= options["minimum_shared_states"] and
                        (~shared).sum() >= options["minimum_new_states"] and new_goals >= options["minimum_new_goals"]):
                    break
            else:
                raise RuntimeError("Could not construct a valid physically overlapping pool")
        rows.append(row)
    return np.stack(rows)


def make_schedule(collections, updates, mode, seed):
    if mode == "sequential":
        return np.repeat(np.arange(collections), updates)
    rng = np.random.default_rng(seed + 101)
    return np.concatenate([rng.permutation(collections) for _ in range(updates)])


def novel_relations(env, base, rows, config):
    rng = np.random.default_rng(config["data"]["seed"] + 301)
    states = np.unique(rows)
    train_keys = np.unique(np.concatenate([pair_keys(row[base["pairs"]], env.size) for row in rows]))
    pairs = np.column_stack(np.triu_indices(len(states), 1))
    physical_pairs = states[pairs]
    keep = ~np.isin(pair_keys(physical_pairs, env.size), train_keys)
    pairs, physical_pairs = pairs[keep], physical_pairs[keep]
    distances = env.distance(physical_pairs[:, 0], physical_pairs[:, 1])
    selected = []
    for level in config["evaluation"]["cross_distance_levels"]:
        ids = np.flatnonzero(distances == level)
        if not len(ids):
            continue
        selected.extend(rng.choice(ids, min(len(ids), config["evaluation"]["cross_pairs_per_level"]), replace=False))
    selected = np.asarray(selected, dtype=np.int64)
    parent_ids = np.unique(rows[:, base["tasks"][:, 0]])
    goal_ids = np.unique(rows[:, base["tasks"][:, 1]])
    a, b = np.meshgrid(parent_ids, goal_ids, indexing="ij")
    tasks = np.column_stack((a.ravel(), b.ravel()))
    children = env.successors(tasks[:, 0])
    # Every input is seen, while none of the parent/child-to-goal labels was
    # calibrated inside any collection. No hidden novel input condition.
    candidates = np.concatenate((tasks[:, :1], children), axis=1)
    all_keys = np.minimum(candidates, tasks[:, 1, None]) * env.size + np.maximum(candidates, tasks[:, 1, None])
    keep = (~np.isin(all_keys, train_keys).any(-1)) & (tasks[:, 0] != tasks[:, 1])
    tasks, children = tasks[keep], children[keep]
    parent_truth = env.distance(tasks[:, 0], tasks[:, 1])
    chosen = []
    for level in config["evaluation"]["cross_distance_levels"]:
        ids = np.flatnonzero(parent_truth == level)
        if not len(ids):
            continue
        chosen.extend(rng.choice(ids, min(len(ids), config["evaluation"]["cross_decisions_per_level"]), replace=False))
    chosen = np.asarray(chosen, dtype=np.int64)
    if not np.isin(children, states).all():
        raise ValueError("Novel relation evaluation introduced an unseen candidate input")
    if not len(selected) or not len(chosen):
        raise ValueError("No eligible seen-state novel relations")
    return {"states": states, "pairs": pairs[selected], "pair_truth": distances[selected],
            "tasks": np.searchsorted(states, tasks[chosen]),
            "children": np.searchsorted(states, children[chosen]),
            "parent_truth": parent_truth[chosen],
            "truth": env.distance(children[chosen], tasks[chosen, 1, None])}


def build_study(env, base, protected, config):
    forbidden = np.zeros(env.size, dtype=bool)
    forbidden[protected] = True
    forbidden[np.r_[base["poses_validation"].ravel(), base["poses_test"].ravel()]] = True
    pools = {kind: pool_rows(env, base, config, forbidden, kind) for kind in ("overlap", "disjoint")}
    all_train = np.unique(np.r_[pools["overlap"].ravel(), pools["disjoint"].ravel()])
    forbidden[all_train] = True
    rng = np.random.default_rng(config["data"]["seed"] + 501)
    heldout = {}
    for split in ("validation", "test"):
        result = []
        for _ in range(config["data"][f"{split}_poses"]):
            row = pose_groups(env, base["states"][None], forbidden, rng, config["data"]["maximum_pose_attempts"])[0]
            forbidden[row] = True
            result.append(row)
        heldout[split] = np.stack(result)
    novel = {kind: novel_relations(env, base, rows, config) for kind, rows in pools.items()}
    # Coverage of eligible novel decisions can differ between physical
    # cohorts. Compare macro scores only on distance levels present in both.
    for target, keys in (("pair_truth", ("pairs", "pair_truth")),
                         ("parent_truth", ("tasks", "children", "parent_truth", "truth"))):
        levels = np.intersect1d(novel["overlap"][target], novel["disjoint"][target])
        if not len(levels):
            raise ValueError("No common distance strata for novel-relation comparison")
        counts = {level: min(int((data[target] == level).sum()) for data in novel.values()) for level in levels}
        for data in novel.values():
            idx = np.concatenate([np.flatnonzero(data[target] == level)[:counts[level]] for level in levels])
            for key in keys:
                data[key] = data[key][idx]
    return pools, heldout, novel


@torch.no_grad()
def novel_metrics(model, features, data):
    q = model.encode(features)
    a, b = data["pairs"].T
    prediction = (q[a] - q[b]).abs().amax(-1).cpu().numpy()
    errors = np.abs(prediction - data["pair_truth"])
    scores = (q[data["children"]] - q[data["tasks"][:, 1, None]]).abs().amax(-1).cpu().numpy()
    good = data["truth"][np.arange(len(scores)), scores.argmin(-1)] == data["parent_truth"] - 1
    return {"distance_count": len(errors), "mae": float(errors.mean()),
            "macro_mae": float(np.mean([errors[data["pair_truth"] == level].mean() for level in np.unique(data["pair_truth"])])),
            "decision_count": len(good), "action_rate": float(good.mean()),
            "macro_action_rate": float(np.mean([good[data["parent_truth"] == level].mean() for level in np.unique(data["parent_truth"])])),
            "decisions_by_distance": {str(level): {"count": int((data["parent_truth"] == level).sum()),
                "action_rate": float(good[data["parent_truth"] == level].mean())} for level in np.unique(data["parent_truth"])},
            "by_distance": {str(level): {"count": int((data["pair_truth"] == level).sum()),
                "mae": float(errors[data["pair_truth"] == level].mean())} for level in np.unique(data["pair_truth"])}}


def run_condition(env, base, rows, heldout, novel, config, condition, initial, seed, out, device, source_commit):
    out.mkdir()
    model = QMap(env.feature_dim, **config["model"], cap=env.diameter).to(device)
    model.load_state_dict(initial)
    features = [torch.as_tensor(env.features(row), device=device) for row in rows]
    novel_features = torch.as_tensor(env.features(novel["states"]), device=device)
    heldout_features = {s: [torch.as_tensor(env.features(row), device=device) for row in rr] for s, rr in heldout.items()}
    opts = config["training"]
    optimizer = torch.optim.Adam(model.parameters(), lr=opts["learning_rate"])
    schedule = make_schedule(len(rows), opts["updates_per_collection"], condition["schedule"], seed)
    counts = np.zeros(len(rows), dtype=int)
    started, best, best_step = time.monotonic(), float("inf"), 0

    def save(path, step):
        torch.save({"model_type": MODEL_TYPE, "task": env.name, "actions": list(env.actions), "cap": env.diameter,
                    "model_config": config["model"], "source_commit": source_commit, "seed": seed,
                    "step": step, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "condition": condition, "collection_updates": counts.tolist()}, path)

    @torch.no_grad()
    def inspect(step, log, actual=None, current=None):
        nonlocal best, best_step
        fit = [metrics(model.encode(f), base, config, True) for f in features]
        objective = float(np.mean([m["objective"] for m in fit]))
        if objective < best:
            best, best_step = objective, step
            save(out / "best.pt", step)
        row = {"step": step, "seconds": time.monotonic() - started, "current_collection": current,
               "collection_updates": counts.tolist(), "balanced_training_objective": objective,
               "fit": fit, "old_actual_update": actual}
        log.write(json.dumps(row) + "\n")
        log.flush()
        print(json.dumps({"seed": seed, "condition": condition["name"], "step": step,
                          "seconds": row["seconds"], "old_mae": fit[0]["mae"],
                          "old_routes": fit[0]["initial_routes"]["solved"],
                          "all_routes": sum(m["initial_routes"]["solved"] for m in fit),
                          "objective": objective}), flush=True)

    @torch.no_grad()
    def evaluate():
        fit = [metrics(model.encode(f), base, config, True) for f in features]
        return {"fit": fit, "novel_seen_state_relations": novel_metrics(model, novel_features, novel),
                "heldout": {split: [metrics(model.encode(f), base, config, True) for f in ff]
                            for split, ff in heldout_features.items()}}

    initial_result = evaluate()
    with (out / "progress.jsonl").open("w") as log:
        inspect(0, log)
        for step, pool in enumerate(schedule, 1):
            monitor = (step == 1 or step % opts["eval_every"] == 0 or
                       (step - 1) % opts["updates_per_collection"] == 0 or step == len(schedule))
            before = model.encode(features[0]).detach() if monitor else None
            q = model.encode(features[pool])
            huber, rank, _, _ = losses(q, base, config)
            loss = huber + opts["rank_weight"] * rank
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), opts["clip_norm"])
            optimizer.step()
            counts[pool] += 1
            if monitor:
                with torch.no_grad():
                    after = model.encode(features[0])
                    update = actual_update(before, after, base, config)
                inspect(step, log, update, int(pool))
    save(out / "last.pt", step)
    last = evaluate()
    model.load_state_dict(torch.load(out / "best.pt", map_location=device, weights_only=False)["model"])
    selected = evaluate()
    result = {"source_commit": source_commit, "seed": seed, "condition": condition, "best_step": best_step,
              "seconds": time.monotonic() - started, "initial": initial_result, "last": last, "best": selected,
              "collection_updates": counts.tolist(), "primary": "last; fixed and exposure-matched budget"}
    (out / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    config = load_contract(args.contract)
    if args.seed not in config["seeds"]:
        raise ValueError("Seed not in contract")
    torch.set_num_threads(2)
    env = Cube2(args.inputs / "oracle_depth.npy", metric="QTM_URF_6")
    base = dict(np.load(args.inputs / "base_data.npz", allow_pickle=False))
    pools, heldout, novel = build_study(env, base, np.load(args.inputs / "protected_states.npy"), config)
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "contract.json").write_text(json.dumps(config, indent=2) + "\n")
    for kind in pools:
        np.savez_compressed(args.out / f"data_{kind}.npz", rows=pools[kind], **novel[kind],
                            **{f"poses_{s}": rows for s, rows in heldout.items()})
    warm_config = json.loads((args.contract.parent / config["warmup_contract"]).read_text())
    torch.manual_seed(args.seed)
    model = QMap(env.feature_dim, **config["model"], cap=env.diameter).to(args.device)
    features = torch.as_tensor(env.features(base["poses_fit"][0]), device=args.device)
    warm = run_model(model, "encoder", features, env, base, warm_config, args.seed,
                     args.out / "warmup", args.source_commit)
    if not warm["finite_route_criterion_passed"]:
        raise RuntimeError("Old map warmup failed; retention comparison has no learned starting map")
    initial = copy.deepcopy(model.state_dict())
    summary = {"source_commit": args.source_commit, "seed": args.seed, "warmup": warm,
               "data": {kind: {"unique_states": len(np.unique(rows)),
                        "overlap_matrix": [[len(np.intersect1d(a, b)) for b in rows] for a in rows],
                        "unique_goals": len(np.unique(rows[:, np.unique(base["tasks"][:, 1])]))}
                        for kind, rows in pools.items()}, "conditions": {}}
    for condition in config["conditions"]:
        kind = condition["data_kind"]
        summary["conditions"][condition["name"]] = run_condition(env, base, pools[kind], heldout, novel[kind],
            config, condition, initial, args.seed, args.out / condition["name"], args.device, args.source_commit)
        (args.out / "study_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (args.out / "complete.json").write_text(json.dumps({"source_commit": args.source_commit}) + "\n")


if __name__ == "__main__":
    main()
