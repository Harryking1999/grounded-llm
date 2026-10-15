"""Complete shortest-route supervision and matched free-Q/encoder diagnostics."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from experiments.puzzle_qmap.src.model import MODEL_TYPE, QMap, soft_rank_loss
from .data import pose_groups
from .environment import Cube2


def optimal_closure(env, initial):
    pending, decisions = list(initial), set()
    while pending:
        state, goal = map(int, pending.pop())
        if state == goal or (state, goal) in decisions:
            continue
        decisions.add((state, goal))
        children = env.successors(state)
        good = children[env.distance(children, goal) == env.distance(state, goal) - 1]
        if not len(good):
            raise ValueError("Nonterminal state without a shortest successor")
        pending.extend((int(child), goal) for child in good)
    return np.array(sorted(decisions), dtype=np.int64)


def build_routes(env, config, protected):
    options = config["data"]
    rng = np.random.default_rng(options["seed"])
    layer = np.flatnonzero(env.depth == options["root_depth"])
    for attempt in range(options["maximum_attempts"]):
        root = int(rng.choice(layer))
        path = [root]
        while path[-1]:
            children = env.successors(path[-1])
            path.append(int(children[np.flatnonzero(env.depth[children] == env.depth[path[-1]] - 1)[0]]))
        goals = [0, path[options["subgoal_path_index"]]]
        initial = np.array([(s, g) for s in [root, *goals] for g in goals if s != g])
        tasks = optimal_closure(env, initial)
        parents = np.unique(tasks[:, 0])
        states = np.unique(np.r_[goals, parents, env.successors(parents).ravel()])
        branching = (env.distance(env.successors(tasks[:, 0]), tasks[:, 1, None]) ==
                     env.distance(tasks[:, 0], tasks[:, 1])[:, None] - 1).sum(-1)
        if len(states) <= min(options["maximum_states"], config["model"]["dimension"]) and (branching > 1).sum() >= options["minimum_branching_decisions"]:
            break
    else:
        raise RuntimeError("No contracted finite route template found")
    labels = env.distance(states[:, None], states[None, :]).astype(np.float32)
    task_idx = np.searchsorted(states, tasks)
    children = np.searchsorted(states, env.successors(tasks[:, 0]))
    data = {"states": states, "tasks": task_idx, "children": children,
            "initial": np.searchsorted(states, initial), "distances": labels,
            "truth": labels[children, task_idx[:, 1, None]],
            "parent_truth": labels[task_idx[:, 0], task_idx[:, 1]],
            "pairs": np.column_stack(np.triu_indices(len(states), 1))}
    # Prove closure for every optimal action, not just one oracle path.
    supported = set(map(tuple, task_idx.tolist()))
    for row, child_row, truth in zip(task_idx, children, data["truth"]):
        parent, goal = row
        for child in child_row[truth == labels[parent, goal] - 1]:
            if child != goal and (int(child), int(goal)) not in supported:
                raise ValueError("Optimal action exits decision supervision")
    forbidden = np.zeros(env.size, dtype=bool)
    forbidden[protected] = True
    for split, count in (("fit", 1), ("validation", options["validation_poses"]), ("test", options["test_poses"])):
        rows = []
        for _ in range(count):
            row = pose_groups(env, states[None], forbidden, rng, 256)[0]
            forbidden[row] = True
            rows.append(row)
        data[f"poses_{split}"] = np.stack(rows)
    witness = np.abs(labels[:, None] - labels[None]).max(-1)
    np.testing.assert_array_equal(witness, labels)
    data["construction_attempts"] = np.array(attempt + 1)
    return data


class FreeCoordinates(nn.Module):
    """Finite-set diagnostic: no encoder and no oracle coordinate targets."""
    def __init__(self, initial, cap):
        super().__init__()
        self.cap = cap
        self.logits = nn.Parameter(torch.logit(initial.detach() / cap))

    def forward(self):
        return self.cap * self.logits.sigmoid()


def vectors(model, features):
    return model() if isinstance(model, FreeCoordinates) else model.encode(features)


def losses(q, data, config):
    a, b = data["pairs"].T
    pair = (q[a] - q[b]).abs().amax(-1)
    target = torch.as_tensor(data["distances"][a, b], device=q.device)
    scores = (q[data["children"]] - q[data["tasks"][:, 1, None]]).abs().amax(-1)
    truth = torch.as_tensor(data["truth"], device=q.device)
    huber = F.smooth_l1_loss(pair, target)
    rank = soft_rank_loss(scores, truth, torch.ones_like(truth, dtype=torch.bool), config["training"]["temperature"])
    return huber, rank, pair, scores


def covered_rollout(q, data, initial):
    """All actions remain available; unsupported decisions count as failures."""
    children = {tuple(task): row for task, row in zip(data["tasks"], data["children"])}
    result = {"count": len(initial), "solved": 0, "cycle": 0, "unsupported": 0, "budget": 0,
              "first_wrong_steps": []}
    for start, goal in initial:
        state, goal = int(start), int(goal)
        seen, first_wrong = set(), None
        for step in range(29):
            if state == goal:
                result["solved"] += 1
                break
            if state in seen:
                result["cycle"] += 1
                break
            seen.add(state)
            row = children.get((state, goal))
            if row is None:
                result["unsupported"] += 1
                break
            if step == 28:
                result["budget"] += 1
                break
            scores = np.abs(q[row] - q[goal]).max(-1)
            child = int(row[scores.argmin()])
            if first_wrong is None and data["distances"][child, goal] != data["distances"][state, goal] - 1:
                first_wrong = step
            state = child
        result["first_wrong_steps"].append(first_wrong)
    result["success_rate"] = result["solved"] / result["count"]
    return result


@torch.no_grad()
def metrics(q, data, config, with_rollout=False):
    huber, rank, pair, scores = losses(q, data, config)
    a, b = data["pairs"].T
    error = (pair - torch.as_tensor(data["distances"][a, b], device=q.device)).abs()
    truth = torch.as_tensor(data["truth"], device=q.device)
    chosen = scores.argmin(-1)
    good = truth[torch.arange(len(truth), device=q.device), chosen] == torch.as_tensor(data["parent_truth"], device=q.device) - 1
    result = {"huber": float(huber), "rank_ce": float(rank),
              "objective": float(huber + config["training"]["rank_weight"] * rank),
              "mae": float(error.mean()), "median_error": float(error.median()),
              "p95_error": float(torch.quantile(error, .95)), "max_error": float(error.max()),
              "optimal_decisions": int(good.sum()), "decision_count": len(good),
              "action_rate": float(good.float().mean())}
    if with_rollout:
        coordinates = q.cpu().numpy()
        result["initial_routes"] = covered_rollout(coordinates, data, data["initial"])
        result["all_decision_routes"] = covered_rollout(coordinates, data, data["tasks"])
    return result


def actual_update(before_q, after_q, data, config):
    """Observe an actual Adam update, including its accumulated moments."""
    with torch.no_grad():
        bh, br, bp, bs = losses(before_q, data, config)
        ah, ar, ap, ass = losses(after_q, data, config)
        a, b = data["pairs"].T
        target = torch.as_tensor(data["distances"][a, b], device=before_q.device)
        threshold = config["criterion"]["maximum_mae"]
        previously_fit = (bp - target).abs() <= threshold
        truth = torch.as_tensor(data["truth"], device=before_q.device)
        parent = torch.as_tensor(data["parent_truth"], device=before_q.device) - 1
        rows = torch.arange(len(truth), device=before_q.device)
        before_good = truth[rows, bs.argmin(-1)] == parent
        after_good = truth[rows, ass.argmin(-1)] == parent
        return {"huber_before": float(bh), "huber_after": float(ah),
                "rank_before": float(br), "rank_after": float(ar),
                "q_change_rms": float((after_q - before_q).square().mean().sqrt()),
                "previously_fit_pairs": int(previously_fit.sum()),
                "pair_regressions": int((previously_fit & ((ap - target).abs() > threshold)).sum()),
                "previously_optimal_decisions": int(before_good.sum()),
                "decision_regressions": int((before_good & ~after_good).sum()),
                "decision_corrections": int((~before_good & after_good).sum())}


def run_model(model, kind, features, env, data, config, seed, out, source_commit):
    out.mkdir()
    opts = config["training"]
    optimizer = torch.optim.Adam(model.parameters(), lr=opts["learning_rate"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, opts["steps"], opts["minimum_learning_rate"])
    best, best_step = float("inf"), 0
    started = time.monotonic()

    def save(path, step):
        torch.save({"model_type": MODEL_TYPE if kind == "encoder" else "free_q_diagnostic",
                    "task": env.name, "actions": list(env.actions), "cap": env.diameter,
                    "model_config": config["model"], "source_commit": source_commit, "seed": seed,
                    "step": step, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict()}, path)

    def inspect(step, log, update=None):
        nonlocal best, best_step
        result = metrics(vectors(model, features), data, config, True)
        if result["objective"] < best:
            best, best_step = result["objective"], step
            save(out / "best.pt", step)
        row = {"step": step, "seconds": time.monotonic() - started, "fit": result, "actual_update": update}
        log.write(json.dumps(row) + "\n")
        log.flush()
        print(json.dumps({"seed": seed, "kind": kind, **row}), flush=True)

    with (out / "progress.jsonl").open("w") as log:
        inspect(0, log)
        for step in range(1, opts["steps"] + 1):
            q = vectors(model, features)
            huber, rank, _, _ = losses(q, data, config)
            loss = huber + opts["rank_weight"] * rank
            monitor = step == 1 or step % opts["eval_every"] == 0 or step == opts["steps"]
            before = q.detach().clone() if monitor else None
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), opts["clip_norm"])
            optimizer.step()
            scheduler.step()
            if monitor:
                after = vectors(model, features).detach()
                inspect(step, log, actual_update(before, after, data, config))
    save(out / "last.pt", step)
    last = metrics(vectors(model, features), data, config, True)
    model.load_state_dict(torch.load(out / "best.pt", map_location=features.device, weights_only=False)["model"])
    final = metrics(vectors(model, features), data, config, True)
    heldout = {}
    if kind == "encoder":
        for split in ("validation", "test"):
            heldout[split] = [metrics(model.encode(torch.as_tensor(env.features(pose), device=features.device)), data, config, True)
                              for pose in data[f"poses_{split}"]]
    criterion = config["criterion"]
    passed = (final["mae"] <= criterion["maximum_mae"] and final["p95_error"] <= criterion["maximum_p95_error"] and
              (not criterion["all_decisions_optimal"] or final["optimal_decisions"] == final["decision_count"]) and
              (not criterion["all_initial_routes_solved"] or final["initial_routes"]["success_rate"] == 1))
    result = {"seed": seed, "kind": kind, "source_commit": source_commit, "best_step": best_step,
              "seconds": time.monotonic() - started, "best": final, "last": last,
              "heldout": heldout, "finite_route_criterion_passed": passed}
    (out / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--prepared-run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    config = json.loads(args.contract.read_text())
    if args.seed not in config["seeds"]:
        raise ValueError("Seed not included in contract")
    torch.set_num_threads(2)
    env = Cube2(args.prepared_run / "oracle_depth.npy", metric="QTM_URF_6")
    protected = np.load(args.prepared_run / "data.npz", allow_pickle=False)["protected_states"]
    data = build_routes(env, config, protected)
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "contract.json").write_text(json.dumps(config, indent=2) + "\n")
    np.savez_compressed(args.out / "data.npz", **data)
    features = torch.as_tensor(env.features(data["poses_fit"][0]), device=args.device)
    torch.manual_seed(args.seed)
    encoder = QMap(env.feature_dim, **config["model"], cap=env.diameter).to(args.device)
    with torch.no_grad():
        initial = encoder.encode(features)
    free = FreeCoordinates(initial, env.diameter).to(args.device)
    gap = float((free().detach() - initial).abs().max())
    if gap > 1e-6:
        raise ValueError("Free coordinates do not match initial encoder Q")
    summary = {"source_commit": args.source_commit, "seed": args.seed,
               "states": len(data["states"]), "decisions": len(data["tasks"]),
               "initial_routes": len(data["initial"]), "pairs": len(data["pairs"]),
               "initial_q_max_difference": gap, "models": {}}
    for kind, model in (("free", free), ("encoder", encoder)):
        summary["models"][kind] = run_model(model, kind, features, env, data, config, args.seed,
                                            args.out / kind, args.source_commit)
        (args.out / "study_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (args.out / "complete.json").write_text(json.dumps({"source_commit": args.source_commit}) + "\n")


if __name__ == "__main__":
    main()
