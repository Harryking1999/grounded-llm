"""Bounded Q-only repair: independent relation/exit losses and current-policy mining."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from torch.nn import functional as F
from grounded_llm.qmap_sampling import CoverageSampler, task_indices
from grounded_llm.qmap_metrics import pair_metrics, ranking_loss
from .environment import Puzzle8
from .model import QMap, MODEL_TYPE, soft_rank_loss
from .policy_audit import audit
from .evaluate import evaluate


def exit_margins(scores, parent, truth, depth, valid, margin):
    """Train a shortest exit to beat every wrong exit and the current height."""
    good = valid & (truth == depth[:, None] - 1)
    bad = valid & ~good
    if not good.any(-1).all():
        raise ValueError("Every nonterminal training star needs a shortest exit")
    best_good = scores.masked_fill(~good, torch.inf).amin(-1)
    best_bad = scores.masked_fill(~bad, torch.inf).amin(-1)
    competing = bad.any(-1)
    rank = F.relu(margin + best_good[competing] - best_bad[competing]).mean() if competing.any() else scores.sum() * 0
    progress = F.relu(margin + best_good - parent).mean()
    return rank, progress


def exit_path_parents(env, parents, depth, steps):
    """Add the true continuation after a mined trap; training supervision only."""
    parts = [parents]
    current = parents.copy()
    for _ in range(steps):
        current = current[current != 0]
        if not len(current):
            break
        children = env.next[current]
        good = (children >= 0) & (depth[np.maximum(children, 0)] == depth[current, None] - 1)
        current = children[np.arange(len(current)), good.argmax(-1)]
        parts.append(current[current != 0])
    return np.unique(np.concatenate(parts))


def run(config, parent_run, out, source_commit, device):
    out.mkdir(parents=True, exist_ok=False)
    (out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    saved = torch.load(parent_run / config["parent_checkpoint"], map_location=device, weights_only=False)
    if saved["model_type"] != MODEL_TYPE or saved["task"] != "puzzle8":
        raise ValueError("Repair needs the recorded Puzzle8 Q checkpoint")
    torch.manual_seed(config["seed"])
    options = config["training"]
    torch.set_num_threads(options["cpu_threads"])
    rng = np.random.default_rng(config["seed"])
    env = Puzzle8()
    oracle = np.load(parent_run / "oracle.npz")
    env.tables = oracle["distances"]
    env.goal_columns = {int(g): i for i, g in enumerate(oracle["goals"])}
    data = dict(np.load(parent_run / "data.npz"))
    # Frozen parent cohorts and labels are reused verbatim. Only goal-zero
    # training decisions are mined; test trajectories never enter repair.
    model = QMap(**saved["config"]["model"]).to(device)
    model.load_state_dict(saved["model"])
    optimizer = torch.optim.Adam(model.parameters(), lr=options["learning_rate"])
    optimizer.load_state_dict(saved["optimizer"])
    for group in optimizer.param_groups:
        group["lr"] = options["learning_rate"]
    features = torch.as_tensor(env.features(np.arange(env.size)), device=device)
    def q(ids):
        return model.encode(features[torch.as_tensor(ids, device=device, dtype=torch.long)])
    relation_pairs = data["pairs_train"][data["pairs_train"][:, 1] != 0]
    relation_tasks = np.flatnonzero(data["tasks_train"][:, 1] != 0)
    ps = CoverageSampler(relation_pairs[:, 2], rng, len(relation_pairs), .5)
    rs = CoverageSampler(data["parent_distances_train"][relation_tasks], rng, len(relation_tasks), .5)
    depth = env.distance(np.arange(env.size), 0)
    broad = CoverageSampler(depth[1:], rng, env.size - 1, .5)
    idx = task_indices(env, data, "validation", config["evaluation"]["decision_count"], saved["config"])
    vt = data["tasks_validation"][idx]
    cohort = data["tasks_restoration"][:, 0]
    initial, arrays = audit(model, env, device, cohort=cohort)
    reference = arrays["within"].copy()
    replay = np.flatnonzero(arrays["correct"] & (np.arange(env.size) != 0))
    hard = exit_path_parents(env, arrays["hard_parents"], depth, options["exit_steps"])
    parent_info = {"run": str(parent_run), "checkpoint": config["parent_checkpoint"],
                   "source_commit": saved["source_commit"], "step": saved["step"]}
    def save(step, path):
        torch.save({"model_type": MODEL_TYPE, "task": env.name, "actions": list(env.actions),
                    "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "config": saved["config"], "repair_config": config, "source_commit": source_commit,
                    "step": saved["step"] + step, "continuation_step": step, "parent": parent_info}, path)
    def validate():
        pm = pair_metrics(model, env, data["pairs_validation"], device)
        rank = ranking_loss(model, env, vt, data["child_distances_validation"][idx], device, options["rank_temperature"])
        return {"mae": pm["mae"], "objective": pm["huber"] + rank}
    initial_relation = validate()
    best_score, best_step = initial["within_budget"], 0
    best_report, best_relation = initial, initial_relation
    save(0, out / "best.pt")
    (out / "initial_audit.json").write_text(json.dumps(initial, indent=2) + "\n")
    print(json.dumps({"initial": initial, "relation": initial_relation}), flush=True)
    started = time.monotonic()
    with (out / "progress.jsonl").open("w") as log:
        for step in range(1, options["steps"] + 1):
            model.train()
            parents = broad.draw(options["exit_batch_size"]) + 1
            # Preserve already correct stars while correcting actual current traps.
            hard_draw = rng.choice(hard if len(hard) else np.arange(1, env.size), options["hard_batch_size"])
            replay_draw = rng.choice(replay if len(replay) else np.arange(1, env.size), options["replay_batch_size"])
            parents = np.r_[parents, hard_draw, replay_draw]
            children = env.next[parents]
            valid = torch.as_tensor(children >= 0, device=device)
            scores = model.distance(q(np.maximum(children, 0)), q([0])[:, None])
            heights = model.distance(q(parents), q([0]))
            truth = torch.as_tensor(depth[np.maximum(children, 0)], device=device)
            parent_depth = torch.as_tensor(depth[parents], device=device)
            exit_huber = F.smooth_l1_loss(heights, parent_depth.float())
            exit_rank = soft_rank_loss(scores, truth, valid, options["rank_temperature"])
            margin, progress = exit_margins(scores, heights, truth, parent_depth, valid, options["margin"])
            rows = relation_pairs[ps.draw(options["relation_pair_batch_size"])]
            relation_huber = F.smooth_l1_loss(model.distance(q(rows[:, 0]), q(rows[:, 1])),
                                              torch.as_tensor(rows[:, 2], device=device).float())
            ri = relation_tasks[rs.draw(options["relation_rank_batch_size"])]
            rc = data["children_train"][ri]
            rscore = model.distance(q(np.maximum(rc, 0)), q(data["tasks_train"][ri, 1])[:, None])
            relation_rank = soft_rank_loss(rscore, torch.as_tensor(data["child_distances_train"][ri], device=device),
                                            torch.as_tensor(rc >= 0, device=device), options["rank_temperature"])
            exit_loss = exit_huber + exit_rank
            relation_loss = relation_huber + relation_rank
            loss = (options["exit_weight"] * exit_loss + options["relation_weight"] * relation_loss +
                    options["margin_weight"] * margin + options["progress_weight"] * progress)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step()
            if step == 1 or step % options["eval_every"] == 0 or step == options["steps"]:
                report, arrays = audit(model, env, device, reference=reference, cohort=cohort)
                relation = validate()
                # Selection uses all-state training-goal execution and validation
                # relations, never the heldout test goal results.
                score = report["within_budget"] - options["lost_solution_penalty"] * report["previously_solved_lost"]
                eligible = relation["mae"] <= initial_relation["mae"] + config["selection"]["maximum_validation_mae_increase"]
                if eligible and score > best_score:
                    best_score, best_step, best_report, best_relation = score, step, report, relation
                    save(step, out / "best.pt")
                hard = exit_path_parents(env, arrays["hard_parents"], depth, options["exit_steps"])
                record = {"continuation_step": step, "seconds": round(time.monotonic() - started, 2),
                          "exit_loss": float(exit_loss.detach()), "relation_loss": float(relation_loss.detach()),
                          "margin_loss": float(margin.detach()), "progress_loss": float(progress.detach()),
                          "policy": report, "relation_validation": relation, "selection_eligible": eligible,
                          "best_step": best_step, "broad_exit_passes": broad.passes,
                          "relation_pair_passes": ps.passes, "relation_decision_passes": rs.passes}
                log.write(json.dumps(record) + "\n"); log.flush()
                print(json.dumps(record), flush=True)
    save(step, out / "last.pt")
    final_checkpoint = torch.load(out / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(final_checkpoint["model"])
    final = evaluate(model, env, data, saved["config"], device)
    summary = {"source_commit": source_commit, "parent": parent_info, "repair_config": config,
               "initial": initial, "initial_relation_validation": initial_relation,
               "best_step": best_step, "best_policy": best_report, "best_relation_validation": best_relation,
               "last_step": step, "last_policy": report, "last_relation_validation": relation,
               "training_seconds": round(time.monotonic() - started, 2), "final": final,
               "scope": "all restoration states are training states; test intermediate goals remain heldout"}
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"complete": str(out), "best_step": best_step, "policy": best_report}), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--parent-run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    run(json.loads(args.config.read_text(encoding="utf-8")), args.parent_run, args.out, args.source_commit, args.device)


if __name__ == "__main__":
    main()
