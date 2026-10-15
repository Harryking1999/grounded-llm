"""Small multi-goal geometry checks with isolated supervision interventions."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from experiments.puzzle_qmap.src.evaluate import encode, rollout, task_indices
from experiments.puzzle_qmap.src.model import MODEL_TYPE, MapReader, QMap, soft_rank_loss
from .data import pose_groups
from .environment import Cube2


def build_template(env, config, protected):
    rng = np.random.default_rng(config["seed"])
    paths = []
    for depth in config["data"]["path_depths"]:
        state = int(rng.choice(np.flatnonzero(env.depth == depth)))
        path = [state]
        while state:
            children = env.successors(state)
            state = int(children[np.flatnonzero(env.depth[children] == env.depth[state] - 1)[0]])
            path.append(state)
        paths.append(path)
    # Include the known square/far-goal witness as well as ordinary paths.
    cycle = np.array([1, 1574641, 1224721, 629857])
    parents = np.unique(np.r_[cycle, np.concatenate([p[:-1] for p in paths])])
    states = np.unique(np.r_[0, parents, env.successors(parents).ravel()])
    if len(states) > config["data"]["maximum_template_states"]:
        raise ValueError("Template exceeds the predeclared anchor-feasibility dimension")
    goals = np.unique([0, paths[0][0], paths[1][0], paths[0][len(paths[0]) // 2],
                       paths[1][len(paths[1]) // 2], int(cycle[1])])
    if len(goals) != config["data"]["goals"]:
        raise ValueError("Expected distinct predeclared goals")
    p, g = np.meshgrid(parents, goals, indexing="ij")
    tasks = np.column_stack((p.ravel(), g.ravel()))
    tasks = tasks[tasks[:, 0] != tasks[:, 1]]
    task_idx = np.searchsorted(states, tasks)
    child_idx = np.searchsorted(states, env.successors(tasks[:, 0]))
    labels = env.distance(states[:, None], states[None, :])
    truth = labels[child_idx, task_idx[:, 1, None]]
    parent_truth = labels[task_idx[:, 0], task_idx[:, 1]]
    edges = np.unique(np.sort(np.column_stack((np.repeat(np.searchsorted(states, parents), 6),
                                               np.searchsorted(states, env.successors(parents)).ravel())), axis=1), axis=0)
    pairs = np.column_stack(np.triu_indices(len(states), 1))
    forbidden = np.zeros(env.size, dtype=bool)
    forbidden[protected] = True
    poses = {}
    for split, count in (("fit", 1), ("validation", config["data"]["validation_poses"]),
                         ("test", config["data"]["test_poses"])):
        rows = []
        for _ in range(count):
            row = pose_groups(env, states[None], forbidden, rng, 256)[0]
            forbidden[row] = True
            rows.append(row)
        poses[split] = np.stack(rows)
    return {"states": states, "tasks": task_idx, "children": child_idx,
            "truth": truth, "parent_truth": parent_truth, "distances": labels,
            "pairs": pairs, "edges": edges, "poses": poses,
            "forbidden": forbidden, "paths": paths}


def relation_values(model, q, data, pair_indices):
    a, b = data["pairs"][pair_indices].T
    pair = model.distance(q[:, a], q[:, b])
    tasks, children = data["tasks"], data["children"]
    scores = model.distance(q[:, children], q[:, tasks[:, 1], None])
    parent = model.distance(q[:, tasks[:, 0]], q[:, tasks[:, 1]])
    a, b = data["edges"].T
    edges = model.distance(q[:, a], q[:, b])
    return pair, scores, parent, edges


def objective(model, q, data, config, condition, pair_indices):
    if condition.get("coordinate_teacher", False):
        if condition["fresh"] or model.metric != "linf":
            raise ValueError("Finite anchor-coordinate probe requires fixed-pose Linf")
        # All anchors belong to the fixed training set, with known exact
        # distances. Only diagnoses discovery of a feasible finite embedding;
        # random fixed anchors were already insufficient on the whole cube.
        target = F.pad(torch.as_tensor(data["distances"], device=q.device).float(),
                       (0, q.shape[-1] - len(data["states"])))
        return F.smooth_l1_loss(q, target[None].expand_as(q))
    pair, scores, parent, edges = relation_values(model, q, data, pair_indices)
    pairs = data["pairs"][pair_indices]
    target = torch.as_tensor(data["distances"][pairs[:, 0], pairs[:, 1]], device=q.device).float()
    pair_loss = F.smooth_l1_loss(pair, target[None].expand_as(pair))
    truth = torch.as_tensor(data["truth"], device=q.device)
    rank = soft_rank_loss(scores.reshape(-1, 6), truth.repeat(len(q), 1),
                          torch.ones_like(truth, dtype=torch.bool).repeat(len(q), 1),
                          config["training"]["temperature"])
    edge_loss = F.relu(edges - 1).square().mean()
    good = truth == torch.as_tensor(data["parent_truth"], device=q.device)[:, None] - 1
    progress = parent[:, :, None] - scores
    path_loss = F.smooth_l1_loss(progress[:, good], torch.ones_like(progress[:, good]))
    symmetry = (pair[0] - pair[1]).square().mean() if len(q) == 2 else pair.new_zeros(())
    loss = (pair_loss + config["training"]["rank_weight"] * rank +
            condition["edge"] * edge_loss + condition["path"] * path_loss +
            condition["symmetry"] * symmetry)
    return loss


@torch.no_grad()
def measure(model, env, data, poses, config, device, with_rollout=False):
    model.eval()
    q = model.encode(torch.as_tensor(env.features(poses), device=device)).reshape(len(poses), len(data["states"]), -1)
    pair, scores, parent, edges = relation_values(model, q, data, np.arange(len(data["pairs"])))
    a, b = data["pairs"].T
    true_pairs = torch.as_tensor(data["distances"][a, b], device=device)
    truth = torch.as_tensor(data["truth"], device=device)
    parent_truth = torch.as_tensor(data["parent_truth"], device=device)
    selected = scores.argmin(-1)
    chosen_truth = truth[torch.arange(len(truth), device=device)[None], selected]
    correct = chosen_truth == parent_truth[None] - 1
    good = truth == parent_truth[:, None] - 1
    error = (pair - true_pairs[None]).abs()
    rank = soft_rank_loss(scores.reshape(-1, 6), truth.repeat(len(poses), 1),
                          torch.ones_like(truth, dtype=torch.bool).repeat(len(poses), 1),
                          config["training"]["temperature"])
    huber = F.smooth_l1_loss(pair, true_pairs[None].expand_as(pair).float())
    middle = (parent_truth >= 5) & (parent_truth <= 10)
    difference = (q[:, a] - q[:, b]).abs()
    if model.metric == "max_l1":
        difference = difference.reshape(*difference.shape[:-1], -1, model.group_size).sum(-1)
    winners = torch.bincount(difference.argmax(-1).ravel(), minlength=difference.shape[-1])
    result = {"objective": float(huber + config["training"]["rank_weight"] * rank),
              "huber": float(huber), "rank_ce": float(rank), "mae": float(error.mean()),
              "max_error": float(error.max()), "shortest_action_rate": float(correct.float().mean()),
              "middle_shortest_action_rate": float(correct[:, middle].float().mean()) if middle.any() else None,
              "edge_mae": float((edges - 1).abs().mean()),
              "edge_over_one_rate": float((edges > 1.01).float().mean()),
              "edge_max": float(edges.max()),
              "shortcut_more_than_one_rate": float((pair < true_pairs[None] - 1).float().mean()),
              "optimal_edge_nonprogress_rate": float(((parent[:, :, None] - scores)[:, good] <= 0).float().mean()),
              "common_left_gap_mean": float((pair - pair[:1]).abs().mean()) if len(poses) > 1 else None,
              "winner_coordinates_or_groups": int((winners > 0).sum()),
              "winner_top1_fraction": float(winners.max() / winners.sum())}
    if with_rollout:
        tasks = poses[:, data["tasks"]].reshape(-1, 2)
        # Fixed subset independent of model outputs; all intermediate states
        # are generated by the environment and may leave the finite template.
        idx = np.random.default_rng(config["seed"] + 81).choice(len(tasks), min(128, len(tasks)), replace=False)
        result["rollout"] = rollout(model, env, tasks[idx], device, 2)["q"]
    return result


def run_condition(env, data, config, condition, out, device, source_commit):
    out.mkdir()
    torch.manual_seed(config["seed"])
    rng = np.random.default_rng(config["seed"] + 1)
    model_config = {**config["model"], "metric": condition["metric"]}
    model = QMap(env.feature_dim, **model_config, cap=env.diameter).to(device)
    opts = config["training"]
    optimizer = torch.optim.Adam(model.parameters(), lr=opts["learning_rate"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, opts["steps"], opts["minimum_learning_rate"])
    complete = np.arange(len(data["pairs"]))
    fixed_features = torch.as_tensor(env.features(data["poses"]["fit"]), device=device)
    best, best_step = float("inf"), 0
    started = time.monotonic()

    def save(path, step):
        torch.save({"model_type": MODEL_TYPE, "task": env.name, "actions": list(env.actions),
                    "model_config": model_config, "cap": env.diameter, "model": model.state_dict(),
                    "source_commit": source_commit, "step": step}, path)

    def inspect(step, log):
        nonlocal best, best_step
        fit = measure(model, env, data, data["poses"]["fit"], config, device)
        val = measure(model, env, data, data["poses"]["validation"], config, device)
        score = val["objective"] if condition["fresh"] else fit["objective"]
        if score < best:
            best, best_step = score, step
            save(out / "best.pt", step)
        record = {"step": step, "seconds": round(time.monotonic() - started, 2), "fit": fit, "validation": val}
        log.write(json.dumps(record) + "\n")
        log.flush()
        print(json.dumps({"condition": condition["name"], **record}), flush=True)

    with (out / "progress.jsonl").open("w") as log:
        inspect(0, log)
        for step in range(1, opts["steps"] + 1):
            model.train()
            if condition["fresh"]:
                poses = pose_groups(env, np.tile(data["states"], (2, 1)), data["forbidden"], rng, 256)
                features = torch.as_tensor(env.features(poses), device=device)
                count = 2
            else:
                features, count = fixed_features, 1
            q = model.encode(features).reshape(count, len(data["states"]), -1)
            # Advance this RNG in every condition, preserving the pose stream
            # across single-factor fresh-pose interventions.
            sparse = rng.choice(len(complete), min(opts["sparse_pairs"], len(complete)), replace=False)
            indices = sparse if condition["pairs"] == "sparse" else complete
            loss = objective(model, q, data, config, condition, indices)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            scheduler.step()
            if step % opts["eval_every"] == 0 or step == opts["steps"]:
                inspect(step, log)
    save(out / "last.pt", step)
    last = {split: measure(model, env, data, poses, config, device, True)
            for split, poses in data["poses"].items()}
    model.load_state_dict(torch.load(out / "best.pt", map_location=device, weights_only=False)["model"])
    final = {split: measure(model, env, data, poses, config, device, True)
             for split, poses in data["poses"].items()}
    criterion = config["fit_criterion"]
    result = {"source_commit": source_commit, "condition": condition, "best_step": best_step,
              "seconds": time.monotonic() - started, "last": last, "best": final,
              "finite_fit_passed": (final["fit"]["max_error"] < criterion["maximum_absolute_distance_error"] and
                                    (not criterion["all_actions_shortest"] or final["fit"]["shortest_action_rate"] == 1.))}
    (out / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


@torch.no_grad()
def audit_frozen(run, device):
    """Validation-only audit, never used to change the norm-trial selection."""
    config = json.loads((run / "config.json").read_text())
    env = Cube2(run / "oracle_depth.npy", metric="QTM_URF_6")
    data = dict(np.load(run / "data.npz", allow_pickle=False))
    idx = task_indices(env, data, "validation", config["evaluation"]["decision_count"], config)
    tasks = data["tasks_validation"][idx]
    model = MapReader(env, run / "best.pt", device).model
    children = env.successors(tasks[:, 0])
    a = encode(model, env, tasks[:, 0], device)
    b = encode(model, env, children, device)
    g = encode(model, env, tasks[:, 1], device)
    edge = model.distance(a[:, None], b)
    before = model.distance(a, g)
    after = model.distance(b, g[:, None])
    truth = torch.as_tensor(env.distance(children, tasks[:, 1, None]), device=device)
    depth = torch.as_tensor(env.distance(tasks[:, 0], tasks[:, 1]), device=device)
    good = truth == depth[:, None] - 1
    progress = before[:, None] - after
    # Along a sampled exact shortest trajectory, detect the first failure to
    # rank a descending child. This is oracle trajectory scoring, not solving.
    current, goals = tasks[:, 0].copy(), tasks[:, 1]
    first_error = np.full(len(tasks), -1)
    for step in range(env.diameter):
        live = np.flatnonzero(current != goals)
        if not len(live):
            break
        candidates = env.successors(current[live])
        labels = env.distance(candidates, goals[live, None])
        parent_depth = env.distance(current[live], goals[live])
        scores = model.distance(encode(model, env, candidates, device),
                                encode(model, env, goals[live], device)[:, None]).cpu().numpy()
        wrong = labels[np.arange(len(live)), scores.argmin(-1)] != parent_depth - 1
        record = live[wrong & (first_error[live] < 0)]
        first_error[record] = step
        exact_action = (labels == parent_depth[:, None] - 1).argmax(-1)
        current[live] = candidates[np.arange(len(live)), exact_action]
    return {"source_commit": torch.load(run / "best.pt", map_location="cpu", weights_only=False)["source_commit"],
            "validation_only": True, "decision_count": len(tasks),
            "edge_over_one_rate": float((edge > 1.01).float().mean()),
            "edge_mae": float((edge - 1).abs().mean()), "edge_max": float(edge.max()),
            "candidate_underestimate_more_than_one_rate": float((after < truth - 1).float().mean()),
            "optimal_edge_nonprogress_rate": float((progress[good] <= 0).float().mean()),
            "absolute_error_gap_lt_one_fraction": float(((after - truth).abs() < 1).float().mean()),
            "all_shortest_trajectory_decisions_correct_rate": float((first_error < 0).mean()),
            "first_error_step_counts": {str(k): int(v) for k, v in zip(*np.unique(first_error, return_counts=True))}}


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--prepared-run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--audit-study", type=Path, help="Optional completed fixed-metric study to audit on validation")
    args = parser.parse_args()
    torch.set_num_threads(2)
    config = json.loads(args.contract.read_text())
    if "base_contract" in config:
        base = json.loads((args.contract.parent / config["base_contract"]).read_text())
        config = {**base, **config}
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "contract.json").write_text(json.dumps(config, indent=2) + "\n")
    env = Cube2(args.prepared_run / "oracle_depth.npy", metric="QTM_URF_6")
    protected = np.load(args.prepared_run / "data.npz", allow_pickle=False)["protected_states"]
    data = build_template(env, config, protected)
    np.savez_compressed(args.out / "data.npz", **{k: v for k, v in data.items() if k not in ("poses", "paths")},
                        **{f"poses_{k}": v for k, v in data["poses"].items()})
    # With <= dimension states, the full distance-to-each-state construction
    # proves exact Linf feasibility. This is not a learned or inference result.
    exact = data["distances"].astype(float)
    witness = np.abs(exact[:, None] - exact[None, :]).max(-1)
    np.testing.assert_array_equal(witness, exact)
    summary = {"source_commit": args.source_commit, "template_states": len(data["states"]),
               "decision_groups": len(data["tasks"]), "full_distance_pairs": len(data["pairs"]),
               "linf_constructive_feasibility_max_error": float(np.abs(witness - exact).max()),
               "conditions": {}}
    for condition in config["conditions"]:
        summary["conditions"][condition["name"]] = run_condition(
            env, data, config, condition, args.out / condition["name"], args.device, args.source_commit)
        (args.out / "study_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    if args.audit_study is not None:
        completed = json.loads((args.audit_study / "complete.json").read_text())
        summary["frozen_validation_audit"] = {name: audit_frozen(args.audit_study / name, args.device)
                                               for name in completed["conditions"]}
        (args.out / "study_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (args.out / "complete.json").write_text(json.dumps({"source_commit": args.source_commit,
        "conditions": list(summary["conditions"])}) + "\n")


if __name__ == "__main__":
    main()
