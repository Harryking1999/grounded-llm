"""All physical states and edges; relational holdout, multi-goal stars, replay."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from experiments.puzzle_qmap.src.data import pair_keys
from experiments.puzzle_qmap.src.evaluate import decisions, pair_metrics, rollout
from experiments.puzzle_qmap.src.model import MODEL_TYPE, QMap, soft_rank_loss
from .environment import Cube2
from .route_complete import losses, metrics


class RelationGuard:
    """Unordered relation exclusions apply to both loss families."""
    def __init__(self, size, pairs):
        self.size = size
        self.keys = np.unique(pair_keys(pairs, size))

    def contains(self, pairs):
        shape = np.asarray(pairs).shape[:-1]
        keys = pair_keys(np.asarray(pairs).reshape(-1, 2), self.size)
        pos = np.searchsorted(self.keys, keys)
        safe = np.minimum(pos, max(0, len(self.keys) - 1))
        found = np.zeros(len(keys), dtype=bool) if not len(self.keys) else (pos < len(self.keys)) & (self.keys[safe] == keys)
        return found.reshape(shape)

    def task_blocked(self, env, tasks):
        star = np.column_stack((tasks[:, 0], env.successors(tasks[:, 0])))
        pairs = np.stack((star, np.broadcast_to(tasks[:, 1, None], star.shape)), axis=-1)
        return self.contains(pairs).any(-1)


def task_pairs(env, tasks):
    stars = np.column_stack((tasks[:, 0], env.successors(tasks[:, 0])))
    return np.stack((stars, np.broadcast_to(tasks[:, 1, None], stars.shape)), axis=-1).reshape(-1, 2)


def fixed_batch(env, tasks):
    physical = task_pairs(env, tasks)
    states, inverse = np.unique(physical, return_inverse=True)
    pairs = inverse.reshape(-1, 2)
    width = len(env.actions) + 1
    grouped = pairs.reshape(-1, width, 2)
    return {"states": states, "pairs": pairs,
            "labels": env.distance(physical[:, 0], physical[:, 1]).astype(np.float32),
            "children": grouped[:, 1:, 0], "goals": grouped[:, 0, 1],
            "truth": env.distance(env.successors(tasks[:, 0]), tasks[:, 1, None])}


def sample_tasks(env, levels, count, rng, layers, forbidden):
    result = []
    for level in levels:
        rows = []
        while len(rows) < count:
            parents = rng.integers(env.size, size=max(32, count - len(rows)))
            relatives = rng.choice(layers[level], size=len(parents))
            goals = env.compose(parents, relatives)
            candidates = np.column_stack((parents, goals))
            candidates = candidates[~forbidden.task_blocked(env, candidates)]
            rows.extend(candidates.tolist())
        result.extend(rows[:count])
    return np.asarray(result, dtype=np.int64)


def prepare(env, config, base, old):
    """Reuse old cohorts and isolate new relational tasks before sampling."""
    rng = np.random.default_rng(config["data"]["seed"])
    layers = [np.flatnonzero(env.depth == d) for d in range(env.diameter + 1)]
    replay_pairs = np.concatenate([row[base["pairs"]] for row in old["rows"]])
    replay_guard = RelationGuard(env.size, replay_pairs)
    cohorts = {"historical_cross": old["states"][old["tasks"]],
               "historical_pose": old["poses_test"][:, base["initial"]].reshape(-1, 2)}
    # The historical novel distance pairs are independently excluded as well.
    excluded = [old["states"][old["pairs"]]]
    for tasks in cohorts.values():
        excluded.append(task_pairs(env, tasks))
    if replay_guard.contains(np.concatenate(excluded)).any():
        raise ValueError("Historical holdout overlaps replay labels")
    previous = np.concatenate([replay_pairs, *excluded])
    opts = config["evaluation"]
    for split in ("validation", "test"):
        tasks = sample_tasks(env, opts["distance_levels"], opts[f"{split}_per_level"], rng, layers,
                             RelationGuard(env.size, previous))
        cohorts[split] = tasks
        relations = task_pairs(env, tasks)
        excluded.append(relations)
        previous = np.concatenate((previous, relations))
    guard = RelationGuard(env.size, np.concatenate(excluded))
    cohorts["train_monitor"] = sample_tasks(env, opts["distance_levels"], opts["train_monitor_per_level"],
                                            rng, layers, guard)
    if guard.contains(replay_pairs).any():
        raise ValueError("Replay leaks a heldout relation")
    # Distance-one edges must all remain trainable under the relation split.
    excluded_pairs = np.column_stack((guard.keys // env.size, guard.keys % env.size))
    if np.any(env.distance(excluded_pairs[:, 0], excluded_pairs[:, 1]) <= 1):
        raise ValueError("Cannot hold out action-edge relations while training every edge")
    return cohorts, guard, layers


class FullStateSampler:
    def __init__(self, env, config, guard, layers):
        self.env, self.config, self.guard, self.layers = env, config, guard, layers
        self.rng = np.random.default_rng(config["data"]["seed"] + 17)
        self.seen = np.zeros(env.size, dtype=bool)
        self.histogram = np.zeros(env.diameter + 1, dtype=np.int64)
        self.parents_drawn = 0
        self.rejected_tasks = 0

    def batches(self):
        order = self.rng.permutation(self.env.size)
        width = self.config["training"]["parent_batch_size"]
        for start in range(0, len(order), width):
            yield order[start:start + width]

    def draw(self, parents):
        env, rng = self.env, self.rng
        count = len(parents)
        distance_levels = np.asarray(self.config["data"]["target_distance_levels"])
        levels = distance_levels[(np.arange(count) + self.parents_drawn) % len(distance_levels)]
        rng.shuffle(levels)
        relative = np.array([rng.choice(self.layers[level]) for level in levels])
        own_goals = env.compose(parents, relative)
        # Use another parent's target: these cross-star relations are not
        # confined to independent local collections. A cyclic shift reuses Q.
        goals = np.column_stack((own_goals, np.roll(own_goals, 1)))
        for column in range(goals.shape[1]):
            tasks = np.column_stack((parents, goals[:, column]))
            blocked = self.guard.task_blocked(env, tasks) | (tasks[:, 0] == tasks[:, 1])
            while blocked.any():
                ids = np.flatnonzero(blocked)
                self.rejected_tasks += len(ids)
                if column == 0:
                    rel = np.array([rng.choice(self.layers[levels[i]]) for i in ids])
                    goals[ids, column] = env.compose(parents[ids], rel)
                else:
                    goals[ids, column] = rng.integers(env.size, size=len(ids))
                tasks = np.column_stack((parents, goals[:, column]))
                blocked = self.guard.task_blocked(env, tasks) | (tasks[:, 0] == tasks[:, 1])
        stars = np.column_stack((parents, env.successors(parents)))
        states, inverse = np.unique(np.r_[stars.ravel(), goals.ravel()], return_inverse=True)
        star_idx = inverse[:stars.size].reshape(stars.shape)
        goal_idx = inverse[stars.size:].reshape(goals.shape)
        # Edge labels and every parent/candidate-to-goal distance enter Huber.
        edges = np.stack((np.repeat(star_idx[:, :1], stars.shape[1] - 1, axis=1), star_idx[:, 1:]), axis=-1).reshape(-1, 2)
        relation_pairs = np.stack((np.broadcast_to(star_idx[:, None], (count, 2, stars.shape[1])),
                                  np.broadcast_to(goal_idx[:, :, None], (count, 2, stars.shape[1]))), axis=-1).reshape(-1, 2)
        truth = env.distance(stars[:, None, :], goals[:, :, None])
        pairs = np.concatenate((edges, relation_pairs))
        labels = np.r_[np.ones(len(edges)), truth.ravel()].astype(np.float32)
        if self.guard.contains(states[pairs]).any():
            raise ValueError("Heldout distance label entered training")
        self.seen[parents] = True
        self.parents_drawn += count
        self.histogram += np.bincount(labels.astype(int), minlength=env.diameter + 1)
        return {"states": states, "pairs": pairs, "labels": labels,
                "children": np.repeat(star_idx[:, None, 1:], 2, axis=1).reshape(-1, stars.shape[1] - 1),
                "goals": goal_idx.ravel(), "truth": truth[:, :, 1:].reshape(-1, stars.shape[1] - 1)}

    def coverage(self):
        return {"unique_parents": int(self.seen.sum()), "state_count": self.env.size,
                "parent_exposures": self.parents_drawn,
                "directed_edges_covered": int(self.seen.sum()) * len(self.env.actions),
                "directed_edge_exposures": self.parents_drawn * len(self.env.actions),
                "absolute_label_histogram": self.histogram.tolist(),
                "rejected_heldout_or_terminal_tasks": self.rejected_tasks}


    def state_dict(self):
        return {"rng": self.rng.bit_generator.state, "seen": self.seen.copy(),
                "histogram": self.histogram.copy(), "parents_drawn": self.parents_drawn,
                "rejected_tasks": self.rejected_tasks}

    def load_state_dict(self, state):
        self.rng.bit_generator.state = state["rng"]
        self.seen = state["seen"].copy()
        self.histogram = state["histogram"].copy()
        self.parents_drawn = state["parents_drawn"]
        self.rejected_tasks = state["rejected_tasks"]


def restore_sampler(sampler, saved):
    """Resume complete epochs; reconstruct the original two-epoch legacy RNG."""
    epoch, step = int(saved["epoch"]), int(saved["step"])
    width = sampler.config["training"]["parent_batch_size"]
    if (saved["coverage"]["parent_exposures"] != epoch * sampler.env.size or
            step != epoch * ((sampler.env.size + width - 1) // width)):
        raise ValueError("Resume requires a completed epoch; use an epoch_N.pt checkpoint")
    if "sampler" in saved:
        sampler.load_state_dict(saved["sampler"])
    else:
        # Old weights/Adam are never updated during this deterministic replay.
        for completed in range(epoch):
            for parents in sampler.batches():
                sampler.draw(parents)
            print(json.dumps({"reconstructed_sampler_epoch": completed + 1}), flush=True)
    if sampler.coverage() != saved["coverage"]:
        raise ValueError("Restored sampler does not match checkpoint exposure")
    return epoch, step


def batch_losses(model, features, batch, config):
    q = model.encode(features)
    a, b = batch["pairs"].T
    predicted = model.distance(q[a], q[b])
    huber = F.smooth_l1_loss(predicted, torch.as_tensor(batch["labels"], device=q.device))
    scores = model.distance(q[batch["children"]], q[batch["goals"]][:, None])
    truth = torch.as_tensor(batch["truth"], device=q.device)
    rank = soft_rank_loss(scores, truth, torch.ones_like(truth, dtype=torch.bool), config["training"]["temperature"])
    return huber, rank


@torch.no_grad()
def evaluate(model, env, cohorts, base, replay_features, replay_tasks, config, device, seen):
    results = {}
    for name, tasks in cohorts.items():
        rows = np.column_stack((tasks, env.distance(tasks[:, 0], tasks[:, 1])))
        inputs = np.r_[tasks.ravel(), env.successors(tasks[:, 0]).ravel()]
        results[name] = {"pairs": pair_metrics(model, env, rows, device),
                         "decisions": decisions(model, env, tasks, device)["q"],
                         "rollout": rollout(model, env, tasks, device, 2,
                                            horizon=config["evaluation"]["horizon"])["q"],
                         "all_initial_inputs_seen_as_training_parents": bool(seen[inputs].all())}
        # Action accuracy needs the same distance stratification as arrival.
        results[name]["actions_by_distance"] = {str(level): decisions(model, env, tasks[rows[:, 2] == level], device)["q"]
                                                for level in np.unique(rows[:, 2])}
    results["replay"] = [metrics(model.encode(features), base, config, True) for features in replay_features]
    results["replay_unrestricted"] = rollout(model, env, replay_tasks, device, 2,
                                              horizon=config["evaluation"]["horizon"])["q"]
    parents = cohorts["train_monitor"][:, 0]
    edges = np.column_stack((np.repeat(parents, len(env.actions)), env.successors(parents).ravel(),
                             np.ones(len(parents) * len(env.actions), dtype=int)))
    results["edge_monitor"] = pair_metrics(model, env, edges, device)
    return results


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--prior-study", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--resume-run", type=Path, help="Continue its last completed-epoch checkpoint with Adam")
    parser.add_argument("--smoke-steps", type=int, default=0, help="Nonformal throughput/flow check; never reports full coverage")
    args = parser.parse_args()
    torch.set_num_threads(2)
    config = json.loads(args.contract.read_text())
    if args.seed not in config["seeds"]:
        raise ValueError("Seed not in contract")
    env = Cube2(args.inputs / "oracle_depth.npy", metric="QTM_URF_6")
    base = dict(np.load(args.inputs / "base_data.npz", allow_pickle=False))
    old = dict(np.load(args.prior_study / "data_overlap.npz", allow_pickle=False))
    if "continuation" in config and args.resume_run is None:
        raise ValueError("This continuation contract requires --resume-run")
    checkpoint = (args.resume_run / "last.pt" if args.resume_run else
                  args.prior_study / config["initialization"]["condition"] / config["initialization"]["checkpoint"])
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    expected_source = config["continuation"]["source_commit"] if args.resume_run else config["initialization"]["source_commit"]
    if saved["source_commit"] not in (expected_source, args.source_commit) or saved["seed"] != args.seed or saved["model_config"] != config["model"]:
        raise ValueError("Wrong starting checkpoint")
    cohorts, guard, layers = prepare(env, config, base, old)
    sampler = FullStateSampler(env, config, guard, layers)
    starting_epoch, starting_step = 0, 0
    if args.resume_run:
        prior_contract = json.loads((args.resume_run / "contract.json").read_text())
        for key in ("model", "data", "evaluation"):
            if prior_contract[key] != config[key]:
                raise ValueError(f"Continuation changed {key}")
        for key, value in prior_contract["training"].items():
            if key != "epochs" and config["training"][key] != value:
                raise ValueError(f"Continuation changed training.{key}")
        with np.load(args.resume_run / "relations.npz", allow_pickle=False) as prior_relations:
            for key, value in {**cohorts, "heldout_keys": guard.keys}.items():
                np.testing.assert_array_equal(value, prior_relations[key])
        starting_epoch, starting_step = restore_sampler(sampler, saved)
        if starting_epoch >= config["training"]["epochs"]:
            raise ValueError("Checkpoint already reached the requested epoch budget")
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "contract.json").write_text(json.dumps(config, indent=2) + "\n")
    np.savez_compressed(args.out / "relations.npz", **cohorts, heldout_keys=guard.keys)
    torch.manual_seed(args.seed)
    model = QMap(env.feature_dim, **config["model"], cap=env.diameter).to(args.device)
    model.load_state_dict(saved["model"])
    features = [torch.as_tensor(env.features(row), device=args.device) for row in old["rows"]]
    replay_tasks = old["rows"][:, base["initial"]].reshape(-1, 2)
    global_replay = fixed_batch(env, cohorts["train_monitor"])
    global_replay_features = torch.as_tensor(env.features(global_replay["states"]), device=args.device)
    opts = config["training"]
    optimizer = torch.optim.Adam(model.parameters(), lr=opts["learning_rate"])
    if args.resume_run:
        optimizer.load_state_dict(saved["optimizer"])
    started, step = time.monotonic(), starting_step
    result = {"source_commit": args.source_commit, "seed": args.seed, "smoke_steps": args.smoke_steps,
              "initial_checkpoint": str(checkpoint), "initial_source_commit": saved["source_commit"],
              "starting_epoch": starting_epoch, "starting_step": starting_step,
              "optimizer_restored": bool(args.resume_run),
              "heldout_relation_count": len(guard.keys), "evaluations": []}

    def save(path, epoch):
        torch.save({"model_type": MODEL_TYPE, "task": env.name, "actions": list(env.actions), "cap": env.diameter,
                    "model_config": config["model"], "source_commit": args.source_commit, "seed": args.seed,
                    "step": step, "epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "coverage": sampler.coverage(), "sampler": sampler.state_dict()}, path)

    def inspect(epoch, log):
        row = {"step": step, "epoch": epoch, "seconds": time.monotonic() - started,
               "coverage": sampler.coverage(),
               "metrics": evaluate(model, env, cohorts, base, features, replay_tasks, config, args.device, sampler.seen)}
        result["evaluations"].append(row)
        result["last"] = row
        (args.out / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
        log.write(json.dumps(row) + "\n")
        log.flush()
        print(json.dumps({"evaluation_step": step, "epoch": epoch, "seconds": row["seconds"],
                          "coverage": row["coverage"], "test": row["metrics"]["test"],
                          "old_routes": sum(m["initial_routes"]["solved"] for m in row["metrics"]["replay"])}), flush=True)

    with (args.out / "progress.jsonl").open("w") as log:
        inspect(starting_epoch, log)
        finished_epochs = starting_epoch
        for epoch in range(starting_epoch + 1, opts["epochs"] + 1):
            for parents in sampler.batches():
                batch = sampler.draw(parents)
                inputs = torch.as_tensor(env.features(batch["states"]), device=args.device)
                huber, rank = batch_losses(model, inputs, batch, config)
                pool = step % (len(features) + 1)
                if pool == len(features):
                    rh, rr = batch_losses(model, global_replay_features, global_replay, config)
                else:
                    rh, rr, _, _ = losses(model.encode(features[pool]), base, config)
                loss = huber + opts["rank_weight"] * rank + opts["replay_weight"] * (rh + opts["rank_weight"] * rr)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), opts["clip_norm"])
                optimizer.step()
                step += 1
                if step % opts["log_every"] == 0 or (args.smoke_steps and step - starting_step == args.smoke_steps):
                    row = {"step": step, "epoch": epoch, "seconds": time.monotonic() - started,
                           "global_huber": float(huber.detach()), "global_rank": float(rank.detach()),
                           "replay_huber": float(rh.detach()), "replay_rank": float(rr.detach()),
                           "coverage": sampler.coverage()}
                    log.write(json.dumps(row) + "\n")
                    log.flush()
                    print(json.dumps(row), flush=True)
                if step % opts["evaluate_every"] == 0:
                    inspect(epoch - 1 + sampler.parents_drawn / env.size % 1, log)
                if args.smoke_steps and step - starting_step >= args.smoke_steps:
                    break
            if not args.smoke_steps:
                if sampler.parents_drawn != epoch * env.size or not sampler.seen.all():
                    raise ValueError("A full epoch did not cover every physical parent")
                finished_epochs = epoch
            inspect(epoch if not args.smoke_steps else sampler.parents_drawn / env.size, log)
            save(args.out / f"epoch_{epoch}.pt", epoch)
            save(args.out / "last.pt", finished_epochs)
            if args.smoke_steps:
                break
        save(args.out / "last.pt", finished_epochs)
    result["completed_epochs"] = finished_epochs
    result["seconds"] = time.monotonic() - started
    (args.out / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    (args.out / "complete.json").write_text(json.dumps({"source_commit": args.source_commit,
        "formal_full_coverage": not bool(args.smoke_steps), "completed_epochs": finished_epochs}) + "\n")


if __name__ == "__main__":
    main()
