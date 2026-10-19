"""Bounded current-policy cycle repair with independently weighted relations.

The optional guard backtracks actual Adam proposals against training-only
correct decisions. It provides no guarantee outside that finite cohort.
"""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from grounded_llm.tasks.cube2.evaluate import pair_metrics, rollout
from grounded_llm.tasks.cube2.model import MODEL_TYPE, QMap, soft_rank_loss
from .cycle_supervision import cycle_pack
from .cycle_trial import close_axis_mask
from grounded_llm.tasks.cube2.environment import Cube2
from .fixed_goal import balanced_parents, evaluate, make_batch, prepare
from .full_state_joint import FullStateSampler, batch_losses
from .mixed_goal import HeldoutStates
from .policy_audit import policy, summarize_policy, two_cycle_basins


def cycle_pairs(pointer):
    ids = np.arange(len(pointer))
    seeds = ids[(pointer[pointer] == ids) & (ids < pointer) & (ids != 0)]
    return np.column_stack((seeds, pointer[seeds]))


def allowed_stars(env, protected):
    allowed = ~protected.copy()
    for start in range(0, env.size, 32768):
        ids = np.arange(start, min(start + 32768, env.size))
        allowed[ids] &= ~protected[env.successors(ids)].any(-1)
    allowed[0] = False
    return allowed


def current_cycles(pointer, allowed):
    pairs = cycle_pairs(pointer)
    return pairs[allowed[pairs].all(-1)]


def split_cycle_losses(model, features, batch, config):
    """Normalize exit and non-solved-goal relations separately, never by count."""
    q = model.encode(features)
    a, b = batch["pairs"].T
    predicted = model.distance(q[a], q[b])
    labels = torch.as_tensor(batch["labels"], device=q.device)
    pair_exit = (batch["states"][batch["pairs"]] == 0).any(-1)
    task_exit = batch["states"][batch["goals"]] == 0
    if not pair_exit.any() or pair_exit.all() or not task_exit.any() or task_exit.all():
        raise ValueError("A repair pack must contain both exit and multi-goal relations")
    scores = model.distance(q[batch["children"]], q[batch["goals"]][:, None])
    truth = torch.as_tensor(batch["truth"], device=q.device)
    result = {}
    for name, pmask, tmask in (("exit", pair_exit, task_exit),
                              ("relation", ~pair_exit, ~task_exit)):
        huber = F.smooth_l1_loss(predicted[pmask], labels[pmask])
        rank = soft_rank_loss(scores[tmask], truth[tmask], torch.ones_like(truth[tmask], dtype=torch.bool),
                              config["training"]["temperature"])
        result[name] = (huber, rank)
    return result


@torch.no_grad()
def full_policy(model, env, device):
    model.eval()
    goal = model.encode(torch.as_tensor(env.features([0]), device=device))
    height = np.empty(env.size, dtype=np.float32)
    for start in range(0, env.size, 8192):
        ids = np.arange(start, min(start + 8192, env.size))
        q = model.encode(torch.as_tensor(env.features(ids), device=device))
        height[ids] = model.distance(q, goal).cpu().numpy()
    pointer, _, _, margins = policy(env, height)
    return height, pointer, margins


def choose_guard(env, margins, allowed, count, fragile_fraction, rng, exclude=()):
    eligible = allowed & np.isfinite(margins) & (margins >= .02)
    eligible[np.asarray(exclude, dtype=np.int64)] = False
    pool = np.flatnonzero(eligible)
    count = min(count, len(pool))
    fragile_count = min(count, round(count * fragile_fraction))
    fragile = pool[np.argsort(margins[pool], kind="stable")[:fragile_count]]
    eligible[fragile] = False
    layers = [np.flatnonzero(eligible & (env.depth == d)) for d in range(1, env.diameter + 1)]
    layers = [layer for layer in layers if len(layer)]
    broad = []
    # Depth-balanced without duplicate stars.
    while len(broad) < count - len(fragile) and layers:
        next_layers = []
        for layer in layers:
            pick = int(rng.choice(layer))
            broad.append(pick)
            remaining = layer[layer != pick]
            if len(remaining):
                next_layers.append(remaining)
            if len(broad) == count - len(fragile):
                break
        layers = next_layers
    return np.r_[fragile, np.asarray(broad, dtype=np.int64)]


class ExitGuard:
    def __init__(self, model, env, parents, floors, device):
        self.floors = np.asarray(floors, dtype=np.float32)
        self.parents = np.asarray(parents, dtype=np.int64)
        children = env.successors(self.parents)
        states = np.unique(np.r_[0, children.ravel()])
        self.features = torch.as_tensor(env.features(states), device=device)
        self.children = np.searchsorted(states, children)
        self.goal = int(np.searchsorted(states, 0))
        self.good = torch.as_tensor(env.depth[children].astype(int) ==
                                    env.depth[self.parents, None].astype(int) - 1, device=device)
        if not len(parents) or not self.good.any(-1).all() or self.good.all(-1).any():
            raise ValueError("Guard needs nonterminal decisions with competing candidates")
        self.model = model

    def margin_tensor(self):
        q = self.model.encode(self.features)
        scores = self.model.distance(q[self.children], q[self.goal])
        return scores.masked_fill(self.good, torch.inf).amin(-1) - scores.masked_fill(~self.good, torch.inf).amin(-1)

    @torch.no_grad()
    def margins(self):
        return self.margin_tensor().cpu().numpy()

    def report(self):
        margin = self.margins()
        return {"count": len(margin), "correct": int((margin > 0).sum()),
                "floor_violations": int((margin < self.floors - 1e-6).sum()),
                "minimum_margin": float(margin.min())}

    def accepts(self):
        margin = self.margins()
        return bool(np.isfinite(margin).all() and (margin >= self.floors - 1e-6).all())


    def project(self, before, delta, iterations, slack):
        """Correct the worst violated exit using its margin derivative.

        Derivatives are evaluated at the unchanged starting parameters. The
        finite candidate is rechecked after every correction; this is not a
        certificate from a linear approximation. The caller still backtracks
        and can reject the entire displacement.
        """
        parameters = list(self.model.parameters())
        self.last_projection_corrections = 0
        for _ in range(iterations):
            with torch.no_grad():
                for p, old, movement in zip(parameters, before, delta):
                    p.copy_(old + movement)
            violation = self.floors - self.margins()
            worst = int(violation.argmax())
            if violation[worst] <= 1e-6:
                break
            with torch.no_grad():
                for p, old in zip(parameters, before):
                    p.copy_(old)
            with torch.enable_grad():
                derivative = torch.autograd.grad(self.margin_tensor()[worst], parameters)
            squared_norm = sum(g.square().sum() for g in derivative)
            if not torch.isfinite(squared_norm) or squared_norm <= 1e-12:
                break
            amount = (float(violation[worst]) + slack) / squared_norm
            delta = [movement + amount * g for movement, g in zip(delta, derivative)]
            self.last_projection_corrections += 1
        return delta


def guarded_adam_step(model, optimizer, accepts, scales, projector=None):
    """Accept a damped Adam proposal, or roll parameters AND optimizer back.

    On a damped acceptance Adam moments record the current gradient normally;
    only its actual parameter displacement is scaled. Rejected steps record
    neither a displacement nor a moment/step advance.
    """
    parameters = list(model.parameters())
    before = [p.detach().clone() for p in parameters]
    old_optimizer = deepcopy(optimizer.state_dict())
    accepted = False
    try:
        optimizer.step()
        delta = [p.detach() - old for p, old in zip(parameters, before)]
        if projector is not None:
            if accepts():
                accepted = True
                return 1.
            delta = projector(before, delta)
        for scale in scales:
            with torch.no_grad():
                for p, old, movement in zip(parameters, before, delta):
                    p.copy_(old + scale * movement)
            if accepts():
                accepted = True
                return float(scale)
        return 0.
    finally:
        if not accepted:
            with torch.no_grad():
                for p, old in zip(parameters, before):
                    p.copy_(old)
            optimizer.load_state_dict(old_optimizer)


def policy_report(env, pointer, initial_pointer, initial_solved):
    report = summarize_policy(env, pointer)
    solved = two_cycle_basins(pointer) == -1
    old, new = cycle_pairs(initial_pointer), cycle_pairs(pointer)
    old_keys = old[:, 0] * env.size + old[:, 1]
    new_keys = new[:, 0] * env.size + new[:, 1]
    report.update({"cycles_removed": len(np.setdiff1d(old_keys, new_keys)),
                   "cycles_added": len(np.setdiff1d(new_keys, old_keys)),
                   "old_solved_lost": int((initial_solved & ~solved).sum()),
                   "old_failed_gained": int((~initial_solved & solved).sum()),
                   "eventually_solved": int(solved.sum())})
    return report


def run(config, checkpoint, oracle, historical_cohorts, out, source_commit, device):
    out.mkdir(parents=True, exist_ok=False)
    (out / "contract.json").write_text(json.dumps(config, indent=2) + "\n")
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if saved["model_config"].get("symmetry") != "cube_axis_permutation_average":
        raise ValueError("Repair must start from the retained symmetric checkpoint")
    env = Cube2(oracle, metric="QTM_URF_6")
    options = config["training"]
    torch.set_num_threads(options["cpu_threads"])
    base = json.loads(Path(config["base_contract"]).read_text())
    mask, _, cohorts = prepare(env, base)
    historical = dict(np.load(historical_cohorts, allow_pickle=False))
    def fresh_model():
        m = QMap(env.feature_dim, **saved["model_config"], cap=saved["cap"]).to(device)
        m.load_state_dict(saved["model"])
        return m
    model = fresh_model()
    _, initial_pointer, initial_margins = full_policy(model, env, device)
    initial_solved = two_cycle_basins(initial_pointer) == -1
    current = cycle_pairs(initial_pointer)
    monitor_rng = np.random.default_rng(config["seed"])
    monitors = current[monitor_rng.choice(len(current), config["evaluation"]["cycles"], replace=False)]
    protected = mask > 0
    for cycles in (historical["cycles"], monitors):
        protected[cycles.ravel()] = True
        protected[env.successors(cycles.ravel()).ravel()] = True
    protected = close_axis_mask(env, protected)
    allowed = allowed_stars(env, protected)
    fixed_guard = choose_guard(env, initial_margins, allowed, config["guard"]["fixed_count"],
                               config["guard"]["fragile_fraction"], monitor_rng)
    if len(fixed_guard) != config["guard"]["fixed_count"]:
        raise ValueError("Not enough training-only correct guard decisions")
    fixed_floors = np.minimum(initial_margins[fixed_guard] * config["guard"]["retain_fraction"],
                              config["guard"]["floor_cap"])
    relations = historical["relations"]
    global_layers = [np.flatnonzero(env.depth == d) for d in range(env.diameter + 1)]
    # Fixed arbitrary goals protected from every continuation label. Their
    # labels might have appeared in pretraining, so this is continuation transfer.
    arbitrary_tasks = []
    for d in range(1, env.diameter + 1):
        remaining = config["evaluation"]["arbitrary_per_depth"]
        while remaining:
            s = monitor_rng.integers(env.size, size=512)
            g = env.compose(s, monitor_rng.choice(global_layers[d], size=len(s)))
            tasks = np.column_stack((s, g))[protected[g]][:remaining]
            arbitrary_tasks.extend(tasks.tolist())
            remaining -= len(tasks)
    arbitrary_tasks = np.asarray(arbitrary_tasks, dtype=np.int64)
    np.savez_compressed(out / "cohorts.npz", cycles=monitors, fixed_guard=fixed_guard,
                        fixed_floors=fixed_floors, relations=relations, arbitrary=arbitrary_tasks, **cohorts)
    train_states = np.flatnonzero(~protected)
    layers = [np.flatnonzero((env.depth == d) & ~protected) for d in range(env.diameter + 1)]
    result = {"source_commit": source_commit, "initial_checkpoint": str(checkpoint),
              "protected_states": int(protected.sum()), "conditions": {},
              "scope": "Same initial symmetric Q and base batches. Repair adds independently weighted labels. "
                       "Guards use allowed training stars only; protected monitors never constrain updates. "
                       "Fixed budget, last checkpoints, no test selection or automatic extension."}
    def inspect(m):
        return {"restoration": evaluate(m, env, cohorts, base, device),
                "current_cycle_routes": rollout(m, env, np.column_stack((monitors.ravel(), np.zeros(monitors.size, dtype=int))),
                                                 device, 2, horizon=28)["q"],
                "relations": pair_metrics(m, env, relations, device),
                "arbitrary_routes": rollout(m, env, arbitrary_tasks, device, 2, horizon=28)["q"]}
    result["initial"] = {"policy": policy_report(env, initial_pointer, initial_pointer, initial_solved),
                         **inspect(model)}
    del model
    for condition in config["conditions"]:
        if condition not in ("base", "repair", "repair_guard", "repair_projected"):
            raise ValueError("Unknown repair condition")
        model = fresh_model()
        optimizer = torch.optim.Adam(model.parameters(), lr=options["learning_rate"])
        rng = np.random.default_rng(config["seed"] + 1)
        cycle_rng = np.random.default_rng(config["seed"] + 2)
        guard_rng = np.random.default_rng(config["seed"] + 3)
        sampler = FullStateSampler(env, config, HeldoutStates(protected), global_layers)
        pointer, margins = initial_pointer, initial_margins
        fixed = ExitGuard(model, env, fixed_guard, fixed_floors, device)
        pool = current_cycles(pointer, allowed)
        history, scales, sampled, corrections = [], [], [], []
        start = time.monotonic()
        with (out / (condition + ".jsonl")).open("w") as log:
            guard = fixed
            for step in range(options["steps"]):
                if step % options["refresh_every"] == 0:
                    if step:
                        _, pointer, margins = full_policy(model, env, device)
                    pool = current_cycles(pointer, allowed)
                    if condition in ("repair_guard", "repair_projected"):
                        dynamic = choose_guard(env, margins, allowed, config["guard"]["dynamic_count"],
                                               config["guard"]["fragile_fraction"], guard_rng, exclude=fixed_guard)
                        floors = np.minimum(margins[dynamic] * config["guard"]["retain_fraction"],
                                            config["guard"]["floor_cap"])
                        guard = ExitGuard(model, env, np.r_[fixed_guard, dynamic], np.r_[fixed_floors, floors], device)
                        if not guard.accepts():
                            raise ValueError("A refresh found an already-violated training guard")
                    row = {"condition": condition, "step": step, "refresh": True,
                           "trainable_current_cycles": len(pool), "fixed_guard": fixed.report(),
                           "policy": policy_report(env, pointer, initial_pointer, initial_solved)}
                    history.append(row)
                    log.write(json.dumps(row) + "\n"); log.flush()
                    print(json.dumps(row), flush=True)
                model.train()
                parents = np.r_[rng.choice(train_states, options["parent_batch_size"]),
                                balanced_parents(layers, options["balanced_replay_size"], step, rng)]
                roots = rng.choice(parents, options["paths_per_batch"], replace=False)
                restore = make_batch(env, parents, protected, roots, rng)
                arbitrary = sampler.draw(parents[allowed[parents]])
                def losses(batch):
                    return batch_losses(model, torch.as_tensor(env.features(batch["states"]), device=device), batch, config)
                h, r = losses(restore)
                gh, gr = losses(arbitrary)
                gw = options["global_weight"]
                loss = (1 - gw) * (h + r) + gw * (gh + gr)
                terms = {"restore_huber": float(h.detach()), "restore_rank": float(r.detach()),
                         "global_huber": float(gh.detach()), "global_rank": float(gr.detach())}
                if condition != "base" and len(pool):
                    extra = None
                    for _ in range(options["pack_attempt_batches"]):
                        candidates = pool[cycle_rng.choice(len(pool), options["cycle_candidates"])]
                        extra = cycle_pack(env, candidates, protected, cycle_rng, options["exit_steps"],
                                           options["branch_goals"], max_packs=options["packs_per_batch"])
                        if extra is not None:
                            break
                    if extra is None:
                        raise ValueError("No complete unprotected current-policy repair pack")
                    split = split_cycle_losses(model, torch.as_tensor(env.features(extra["states"]), device=device), extra, config)
                    for family, (ch, cr) in split.items():
                        loss = loss + options[family + "_weight"] * (ch + cr)
                        terms[family + "_huber"], terms[family + "_rank"] = float(ch.detach()), float(cr.detach())
                    sampled.extend(p["cycle"] for p in extra["packs"])
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), options["clip_norm"])
                guard.last_projection_corrections = 0
                if condition in ("repair_guard", "repair_projected"):
                    projector = None
                    if condition == "repair_projected":
                        projector = lambda before, delta: guard.project(before, delta,
                            config["projection"]["max_corrections"], config["projection"]["slack"])
                    scale = guarded_adam_step(model, optimizer, guard.accepts, config["guard"]["scales"], projector)
                else:
                    optimizer.step()
                    scale = 1.
                scales.append(scale)
                corrections.append(guard.last_projection_corrections)
                row = {"condition": condition, "step": step + 1, "loss": float(loss.detach()),
                       "terms": terms, "gradient_norm_before_clip": float(norm), "accepted_scale": scale,
                       "projection_corrections": guard.last_projection_corrections}
                log.write(json.dumps(row) + "\n"); log.flush()
                if (step + 1) % options["log_every"] == 0:
                    print(json.dumps(row), flush=True)
            height, pointer, _ = full_policy(model, env, device)
            final = {"policy": policy_report(env, pointer, initial_pointer, initial_solved),
                     "fixed_guard": fixed.report(), **inspect(model)}
        payload = {"model_type": MODEL_TYPE, "task": env.name, "actions": list(env.actions),
                   "model_config": saved["model_config"], "model": model.state_dict(), "cap": saved["cap"],
                   "optimizer": optimizer.state_dict(), "source_commit": source_commit, "steps": options["steps"],
                   "condition": condition, "initial_checkpoint": str(checkpoint)}
        torch.save(payload, out / (condition + ".pt"))
        np.save(out / (condition + "_height.npy"), height)
        np.save(out / (condition + "_next_state.npy"), pointer)
        initial = result["initial"]
        criteria = {"relation_mae_lower": final["relations"]["mae"] < initial["relations"]["mae"],
                    "fewer_cycles": final["policy"]["two_cycles"] < initial["policy"]["two_cycles"],
                    "full_arrival_not_lower": final["policy"]["solved"] >= initial["policy"]["solved"],
                    "fixed_guard_no_order_flips": final["fixed_guard"]["correct"] == len(fixed_guard)}
        result["conditions"][condition] = {"final": final, "refreshes": history,
            "criteria": criteria, "all_criteria_pass": all(criteria.values()), "sampled_packs": len(sampled),
            "accepted_scales": {str(s): scales.count(s) for s in sorted(set(scales))},
            "projection_corrections": sum(corrections),
            "elapsed_seconds": time.monotonic() - start}
        np.save(out / (condition + "_sampled_cycles.npy"), np.asarray(sampled, dtype=np.int64).reshape(-1, 2))
        (out / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps({"finished": condition, "criteria": criteria, "policy": final["policy"]}), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(__doc__)
    for name in ("contract", "checkpoint", "oracle", "historical-cohorts", "out"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    config = json.loads(args.contract.read_text())
    if "repair_contract" in config:
        parent = json.loads(Path(config["repair_contract"]).read_text())
        config = {**parent, **config}
    run(config, args.checkpoint, args.oracle, args.historical_cohorts,
        args.out, args.source_commit, args.device)


if __name__ == "__main__":
    main()
