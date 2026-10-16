"""Bounded comparison of linked cycle supervision and exact axis projection."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from experiments.puzzle_qmap.src.evaluate import pair_metrics, rollout
from experiments.puzzle_qmap.src.model import MODEL_TYPE, QMap
from .cycle_supervision import cycle_pack
from .environment import Cube2
from .fixed_goal import balanced_parents, evaluate, make_batch, prepare
from .full_state_joint import FullStateSampler, batch_losses
from .mixed_goal import HeldoutStates


def close_axis_mask(env, mask):
    """No newly supervised symmetry counterpart of a protected state."""
    protected = np.flatnonzero(mask)
    closed = np.asarray(mask, dtype=bool).copy()
    for g in range(6):
        closed[env.axis_symmetry(protected, g)] = True
    return closed


@torch.no_grad()
def align_initialization(saved, env, probe_states, device, invariant_channel=126):
    """Move a learned channel into an invariant slot, without changing old D.

    Channel selection reads old Q on training inputs only, not oracle labels.
    The projection itself still changes distances and must be evaluated.
    """
    if invariant_channel not in (126, 127) or saved["model_config"]["dimension"] != 128:
        raise ValueError("The projection has exactly two invariant slots: 126, 127")
    plain = QMap(env.feature_dim, **saved["model_config"], cap=saved["cap"]).to(device)
    plain.load_state_dict(saved["model"])
    q = plain.encode(torch.as_tensor(env.features(probe_states), device=device))
    goal = plain.encode(torch.as_tensor(env.features([0]), device=device))
    counts = torch.bincount((q - goal).abs().argmax(-1), minlength=128).cpu().numpy()
    dominant = int(counts.argmax())
    permutation = torch.arange(128)
    permutation[invariant_channel], permutation[dominant] = dominant, invariant_channel
    state = {k: v.clone() for k, v in saved["model"].items()}
    for key in ("network.4.weight", "network.4.bias"):
        state[key] = state[key][permutation]
    return state, {"dominant_coordinate": dominant, "invariant_channel": invariant_channel,
                   "probe_states": len(probe_states), "dominant_share": float(counts.max() / len(probe_states)),
                   "scope": "Output-row permutation preserves every original Linf distance before projection."}


def run(config, checkpoint, oracle, policy_cache, out, source_commit, device, smoke_steps=0):
    out.mkdir(parents=True, exist_ok=False)
    (out / "contract.json").write_text(json.dumps(config, indent=2) + "\n")
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if saved["task"] != "cube2_qtm" or saved["model_config"].get("symmetry"):
        raise ValueError("Expected a recorded plain cube Q initialization")
    base = json.loads(Path(config["base_contract"]).read_text())
    env = Cube2(oracle, metric="QTM_URF_6")
    mask, _, cohorts = prepare(env, base)
    pointer = np.load(policy_cache, allow_pickle=False)
    if pointer.shape != (env.size,) or pointer.dtype.kind not in "iu" or np.any((pointer < 0) | (pointer >= env.size)):
        raise ValueError("Invalid frozen next-state cache")
    ids = np.arange(env.size)
    seeds = ids[(pointer[pointer] == ids) & (ids < pointer) & (ids != 0)]
    cycles = np.column_stack((seeds, pointer[seeds]))
    options = config["training"]
    if not set(config["conditions"]) <= {"plain_base", "plain_cycles", "symmetric_base", "symmetric_cycles"}:
        raise ValueError("Unknown comparison condition")
    torch.set_num_threads(options["cpu_threads"])
    monitor_rng = np.random.default_rng(config["seed"])
    monitor_count = min(len(cycles) // 4, config["evaluation"]["cycles"])
    if monitor_count < 1:
        raise ValueError("Need distinct training and monitor cycles")
    chosen = monitor_rng.choice(len(cycles), monitor_count, replace=False)
    monitored = cycles[chosen]
    protected = mask > 0
    protected[monitored.ravel()] = True
    protected[env.successors(monitored.ravel()).ravel()] = True
    protected = close_axis_mask(env, protected)
    cycles = np.delete(cycles, chosen, axis=0)
    stars = np.column_stack((cycles.ravel(), env.successors(cycles.ravel())))
    eligible = ~protected[stars].any(-1).reshape(-1, 2).any(-1)
    cycles = cycles[eligible]
    if not len(cycles):
        raise ValueError("No trainable cycle remains after symmetry closure")
    layers = [np.flatnonzero((env.depth == d) & ~protected) for d in range(env.diameter + 1)]
    train_states = np.flatnonzero(~protected)
    global_layers = [np.flatnonzero(env.depth == d) for d in range(env.diameter + 1)]
    monitor_tasks = np.column_stack((monitored.ravel(), np.zeros(monitored.size, dtype=np.int64)))
    selected = monitor_rng.choice(np.flatnonzero(protected), 2 * config["evaluation"]["relations"])
    relations = np.column_stack((selected.reshape(-1, 2), env.distance(*selected.reshape(-1, 2).T)))
    np.savez_compressed(out / "cohorts.npz", cycles=monitored, relations=relations, **cohorts)
    steps = smoke_steps or options["steps"]
    initial_state, alignment = saved["model"], None
    initialization = config.get("initialization")
    if initialization:
        probes = np.random.default_rng(config["seed"]).choice(train_states, initialization["probe_states"])
        initial_state, alignment = align_initialization(saved, env, probes, device,
                                                       initialization["invariant_channel"])
    result = {"source_commit": source_commit, "initial_checkpoint": str(checkpoint),
              "policy_cache": str(policy_cache), "initial_epoch": saved["epoch"],
              "smoke_steps": smoke_steps, "trainable_cycles": len(cycles),
              "protected_states": int(protected.sum()), "conditions": {}, "initialization_alignment": alignment,
              "scope": "Matched update counts and base batches. Cycle arms add labels and compute. "
                       "Both architectures initialize from the same weights with fresh Adam. "
                       "Historical and cycle monitors may have labels in pretraining; protected from this continuation, "
                       "not a claim of never-seen states/relations. No test selection or automatic extension."}

    def inspect(model):
        model.eval()
        return {"restoration_retention": evaluate(model, env, {k: v[:8] if smoke_steps else v
                                                               for k, v in cohorts.items()}, base, device),
                "protected_cycle_routes": rollout(model, env, monitor_tasks, device, 2, horizon=28)["q"],
                "protected_relations": pair_metrics(model, env, relations, device)}

    for condition in config["conditions"]:
        symmetric, augmented = condition.startswith("symmetric"), condition.endswith("cycles")
        model_config = dict(saved["model_config"])
        if symmetric:
            model_config["symmetry"] = "cube_axis_permutation_average"
        torch.manual_seed(config["seed"])
        model = QMap(env.feature_dim, **model_config, cap=saved["cap"]).to(device)
        model.load_state_dict(initial_state if symmetric else saved["model"], strict=True)
        if symmetric and not (out / "symmetric_initial.pt").exists():
            torch.save({"model_type": MODEL_TYPE, "task": env.name, "actions": list(env.actions),
                        "model_config": model_config, "model": model.state_dict(), "cap": saved["cap"],
                        "source_commit": source_commit, "steps": 0, "initial_epoch": saved["epoch"],
                        "initialization_alignment": alignment}, out / "symmetric_initial.pt")
        optimizer = torch.optim.Adam(model.parameters(), lr=options["learning_rate"])
        rng = np.random.default_rng(config["seed"] + 1)
        cycle_rng = np.random.default_rng(config["seed"] + 2)
        sampler = FullStateSampler(env, config, HeldoutStates(protected), global_layers)
        initial = inspect(model)
        counters = {"cycle_packs": 0, "cycle_pairs": 0, "cycle_rank_tasks": 0}
        with (out / f"{condition}.jsonl").open("w") as log:
            for step in range(steps):
                model.train()
                parents = np.r_[rng.choice(train_states, options["parent_batch_size"]),
                                balanced_parents(layers, options["balanced_replay_size"], step, rng)]
                roots = rng.choice(parents, options["paths_per_batch"], replace=False)
                restore = make_batch(env, parents, protected, roots, rng)
                stars = np.column_stack((parents, env.successors(parents)))
                arbitrary = sampler.draw(parents[~protected[stars].any(-1)])

                def losses(batch):
                    return batch_losses(model, torch.as_tensor(env.features(batch["states"]), device=device), batch, config)

                h, r = losses(restore)
                gh, gr = losses(arbitrary)
                gw = options["global_weight"]
                loss = (1 - gw) * (h + r) + gw * (gh + gr)
                cycle_loss = None
                if augmented:
                    extra = None
                    for _ in range(options["pack_attempt_batches"]):
                        candidates = cycles[cycle_rng.choice(len(cycles), options["cycle_candidates"])]
                        extra = cycle_pack(env, candidates, protected, cycle_rng,
                                           options["exit_steps"], options["branch_goals"],
                                           max_packs=options["packs_per_batch"])
                        if extra is not None:
                            break
                    if extra is None:
                        raise ValueError("Could not form an unprotected complete cycle pack")
                    ch, cr = losses(extra)
                    cycle_loss = ch + cr
                    loss = loss + options["cycle_weight"] * cycle_loss
                    counters["cycle_packs"] += len(extra["packs"])
                    counters["cycle_pairs"] += len(extra["pairs"])
                    counters["cycle_rank_tasks"] += len(extra["goals"])
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), options["clip_norm"])
                optimizer.step()
                if step == 0 or (step + 1) % options["log_every"] == 0:
                    row = {"condition": condition, "step": step + 1, "loss": float(loss.detach()),
                           "cycle_loss": float(cycle_loss.detach()) if cycle_loss is not None else None,
                           "gradient_norm_before_clip": float(norm)}
                    log.write(json.dumps(row) + "\n"); log.flush()
                    print(json.dumps(row), flush=True)
        payload = {"model_type": MODEL_TYPE, "task": env.name, "actions": list(env.actions),
                   "model_config": model_config, "model": model.state_dict(), "cap": saved["cap"],
                   "source_commit": source_commit, "steps": steps, "initial_epoch": saved["epoch"],
                   "optimizer": optimizer.state_dict(), "condition": condition, "smoke_steps": smoke_steps,
                   "initialization_alignment": alignment if symmetric else None}
        torch.save(payload, out / f"{condition}.pt")
        result["conditions"][condition] = {"initial": initial, "final": inspect(model), **counters}
        (out / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps({"condition": condition, "finished_steps": steps}), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(__doc__)
    for name in ("contract", "checkpoint", "oracle", "policy-cache", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--smoke-steps", type=int, default=0)
    args = parser.parse_args()
    run(json.loads(args.contract.read_text()), args.checkpoint, args.oracle, args.policy_cache,
        args.out, args.source_commit, args.device, args.smoke_steps)


if __name__ == "__main__":
    main()
