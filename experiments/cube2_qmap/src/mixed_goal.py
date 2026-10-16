"""Continue one Q with linked restoration, intermediate and arbitrary goals."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from experiments.puzzle_qmap.src.evaluate import decisions, pair_metrics, rollout
from experiments.puzzle_qmap.src.model import MODEL_TYPE, QMap
from .environment import Cube2
from .fixed_goal import balanced_parents, evaluate as restoration_metrics, make_batch, prepare
from .full_state_joint import FullStateSampler, batch_losses


class HeldoutStates:
    """Apply the existing restoration mask to all arbitrary relation labels."""
    def __init__(self, heldout):
        self.heldout = heldout

    def contains(self, pairs):
        return self.heldout[np.asarray(pairs)].any(-1)

    def task_blocked(self, env, tasks):
        stars = np.column_stack((tasks[:, 0], env.successors(tasks[:, 0])))
        return self.heldout[stars].any(-1) | self.heldout[tasks[:, 1]]


def global_cohorts(env, mask, config, layers):
    rng = np.random.default_rng(config["data"]["seed"])
    cohorts = {}
    for name, split in (("validation", 1), ("test", 2)):
        goals = np.flatnonzero(mask == split)
        rows = []
        for depth in config["evaluation"]["global_distance_levels"]:
            count = config["evaluation"][f"global_{name}_per_level"]
            selected = rng.choice(goals, count)
            source = env.compose(selected, rng.choice(layers[depth], count))
            rows.extend(zip(source, selected))
        cohorts[name] = np.asarray(rows, dtype=np.int64)
    return cohorts


def inspect(model, env, restore_cohorts, arbitrary_cohorts, base, config, device):
    model.eval()
    result = {"restoration": restoration_metrics(model, env, restore_cohorts, base, device), "arbitrary_goals": {}}
    for name, tasks in arbitrary_cohorts.items():
        rows = np.column_stack((tasks, env.distance(*tasks.T)))
        result["arbitrary_goals"][name] = {
            "pairs": pair_metrics(model, env, rows, device),
            "decisions": decisions(model, env, tasks, device)["q"],
            "rollout": rollout(model, env, tasks, device, 2,
                               horizon=config["evaluation"]["horizon"])["q"]}
    return result


def run(config, checkpoint, oracle, out, source_commit, device, smoke_steps=0):
    base = json.loads(Path(config["base_contract"]).read_text())
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if (saved["task"] != "cube2_qtm" or saved["model_config"] != config["model"] or
            saved.get("smoke_steps")):
        raise ValueError("Expected a matching formal Q checkpoint")
    torch.set_num_threads(config["training"]["cpu_threads"])
    torch.manual_seed(config["seed"])
    env = Cube2(oracle, metric=base["rules"]["metric"])
    mask, restoration_layers, restore_cohorts = prepare(env, base)
    heldout = mask > 0
    layers = [np.flatnonzero(env.depth == depth) for depth in range(env.diameter + 1)]
    arbitrary_cohorts = global_cohorts(env, mask, config, layers)
    guard = HeldoutStates(heldout)
    sampler = FullStateSampler(env, config, guard, layers)
    rng = np.random.default_rng(config["seed"] + 1)
    options = config["training"]
    model = QMap(env.feature_dim, **config["model"], cap=env.diameter).to(device)
    model.load_state_dict(saved["model"])
    optimizer = torch.optim.Adam(model.parameters(), lr=options["learning_rate"])
    optimizer.load_state_dict(saved["optimizer"])
    for group in optimizer.param_groups:
        group["lr"] = options["learning_rate"]
    out.mkdir(parents=True, exist_ok=False)
    (out / "contract.json").write_text(json.dumps(config, indent=2) + "\n")
    np.savez_compressed(out / "split.npz", mask=mask, **{f"restore_{k}":v for k,v in restore_cohorts.items()},
                        **{f"global_{k}":v for k,v in arbitrary_cohorts.items()})
    result = {"source_commit": source_commit, "initial_checkpoint": str(checkpoint),
              "initial_source_commit": saved["source_commit"], "initial_epoch": saved["epoch"],
              "source_protected_fixed_goal_mask": saved.get("fixed_goal") == 0,
              "smoke_steps": smoke_steps, "evaluations": []}
    seen = np.zeros(env.size, dtype=bool)
    exposed = np.zeros(env.diameter + 1, dtype=np.int64)
    path_labels = goal_labels = step = 0
    started = time.monotonic()

    def record(epoch, log):
        row = {"epoch": epoch, "step": step, "seconds": time.monotonic() - started,
               "unique_parents": int(seen.sum()), "global_coverage": sampler.coverage(),
               "restoration_labels": goal_labels, "path_labels": path_labels,
               "restoration_depth_exposures": exposed.tolist(),
               "metrics": inspect(model, env, restore_cohorts, arbitrary_cohorts, base, config, device)}
        result["evaluations"].append(row)
        result["last"] = row
        (out / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
        log.write(json.dumps(row) + "\n")
        log.flush()
        print(json.dumps({"epoch": epoch, "step": step,
                          "restoration_validation_solved": row["metrics"]["restoration"]["validation"]["rollout"]["solved"],
                          "global_validation_solved": row["metrics"]["arbitrary_goals"]["validation"]["rollout"]["solved"]}), flush=True)

    with (out / "progress.jsonl").open("w") as log:
        record(0, log)
        for epoch in range(1, options["epochs"] + 1):
            for parents in sampler.batches():
                replay = balanced_parents(restoration_layers, options["balanced_replay_size"],
                                           step * options["balanced_replay_size"], rng)
                all_parents = np.r_[parents, replay]
                roots = rng.choice(all_parents, min(len(all_parents), options["paths_per_batch"]), replace=False)
                restore = make_batch(env, all_parents, heldout, roots, rng)
                h, r = batch_losses(model, torch.as_tensor(env.features(restore["states"]), device=device), restore, config)
                stars = np.column_stack((all_parents, env.successors(all_parents)))
                eligible = all_parents[~heldout[stars].any(-1)]
                if not len(eligible):
                    raise ValueError("No unmasked global star in training batch")
                arbitrary = sampler.draw(eligible)
                gh, gr = batch_losses(model, torch.as_tensor(env.features(arbitrary["states"]), device=device), arbitrary, config)
                weight = options["global_weight"]
                huber = (1 - weight) * h + weight * gh
                rank = (1 - weight) * r + weight * gr
                loss = huber + options["rank_weight"] * rank
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), options["clip_norm"])
                optimizer.step()
                seen[parents] = True
                exposed += np.bincount(env.depth[all_parents], minlength=env.diameter + 1)
                goal_labels += restore["goal_pair_count"]
                path_labels += restore["path_pair_count"]
                step += 1
                if step % options["log_every"] == 0:
                    row = {"epoch": epoch, "step": step, "seconds": time.monotonic() - started,
                           "restore_huber": float(h.detach()), "restore_rank": float(r.detach()),
                           "global_huber": float(gh.detach()), "global_rank": float(gr.detach())}
                    log.write(json.dumps(row) + "\n");log.flush()
                    print(json.dumps(row), flush=True)
                if smoke_steps and step >= smoke_steps:
                    break
            complete_epoch = epoch if not smoke_steps else 0
            record(complete_epoch, log)
            payload = {"model_type": MODEL_TYPE, "task": env.name, "actions": list(env.actions),
                       "cap": env.diameter, "model_config": config["model"], "source_commit": source_commit,
                       "initial_source_commit": saved["source_commit"], "seed": config["seed"],
                       "epoch": complete_epoch, "step": step, "smoke_steps": smoke_steps,
                       "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                       "rng": rng.bit_generator.state, "global_rng": sampler.rng.bit_generator.state,
                       "seen": seen, "global_seen": sampler.seen, "config": config}
            temporary = out / "last.pt.tmp"
            torch.save(payload, temporary);temporary.replace(out / "last.pt")
            if not smoke_steps:
                torch.save(payload, out / f"epoch_{epoch}.pt")
            if smoke_steps:
                break
    (out / "complete.json").write_text(json.dumps({"formal": not bool(smoke_steps), "epochs": complete_epoch, "steps": step}) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(__doc__)
    for name in ("contract", "checkpoint", "oracle", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--smoke-steps", type=int, default=0)
    args = parser.parse_args()
    run(json.loads(args.contract.read_text()), args.checkpoint, args.oracle, args.out,
        args.source_commit, args.device, args.smoke_steps)


if __name__ == "__main__":
    main()
