"""Matched small continuation: retain restoration supervision, add waypoint stars."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from experiments.puzzle_qmap.src.model import QMap
from .environment import Cube2
from .fixed_goal import balanced_parents, evaluate, make_batch, prepare
from .full_state_joint import batch_losses
from .landmarks import diagnostics, landmark_batch


def run(config, checkpoint, oracle, out, source_commit):
    out.mkdir(parents=True, exist_ok=False)
    (out / "contract.json").write_text(json.dumps(config, indent=2) + "\n")
    base = json.loads(Path(config["base_contract"]).read_text())
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if (saved["task"] != "cube2_qtm" or saved["fixed_goal"] != 0 or
            saved["model_config"] != base["model"] or saved["smoke_steps"]):
        raise ValueError("Expected a completed matching fixed-goal checkpoint")
    options = config["training"]
    torch.set_num_threads(options["cpu_threads"])
    env = Cube2(oracle, metric=base["rules"]["metric"])
    mask, layers, cohorts = prepare(env, base)
    heldout = mask > 0
    bank = np.random.default_rng(config["seed"]).choice(
        layers[config["landmarks"]["goal_depth"]], config["evaluation"]["bank_size"], replace=False)
    np.savez_compressed(out / "cohorts.npz", bank=bank, **cohorts)
    result = {"source_commit": source_commit, "checkpoint": str(checkpoint),
              "initial_source_commit": saved["source_commit"], "initial_epoch": saved["epoch"],
              "comparison": "matched base batches and updates; augmentation adds labels and compute", "conditions": {}}

    def inspect(model):
        metrics = evaluate(model, env, cohorts, base, "cpu")
        probes = {}
        for split in ("validation", "test"):
            starts = cohorts[split]
            starts = starts[env.depth[starts] == config["evaluation"]["starting_depth"]]
            probes[split] = diagnostics(model, env, starts, bank, config)
        return {"restoration": metrics, "ten_to_seven": probes}

    for condition in ("control", "landmarks"):
        torch.manual_seed(config["seed"])
        model = QMap(env.feature_dim, **saved["model_config"], cap=saved["cap"])
        model.load_state_dict(saved["model"])
        optimizer = torch.optim.Adam(model.parameters(), lr=options["learning_rate"])
        optimizer.load_state_dict(saved["optimizer"])
        for group in optimizer.param_groups:
            group["lr"] = options["learning_rate"]
        rng = np.random.default_rng(config["seed"] + 1)
        landmark_rng = np.random.default_rng(config["seed"] + 2)
        if "initial" not in result:
            result["initial"] = inspect(model)
        started = time.monotonic()
        point_labels = rank_tasks = 0
        with (out / f"{condition}.jsonl").open("w") as log:
            for step in range(options["steps"]):
                model.train()
                parents = np.r_[rng.integers(env.size, size=options["parent_batch_size"]),
                                balanced_parents(layers, options["balanced_replay_size"],
                                                 step * options["balanced_replay_size"], rng)]
                roots = rng.choice(parents, options["paths_per_batch"], replace=False)
                batch = make_batch(env, parents, heldout, roots, rng)
                features = torch.as_tensor(env.features(batch["states"]))
                huber, rank = batch_losses(model, features, batch, config)
                loss = huber + options["rank_weight"] * rank
                lm_huber = lm_rank = None
                if condition == "landmarks":
                    opts = config["landmarks"]
                    levels = landmark_rng.choice(opts["root_depths"], opts["roots_per_batch"])
                    roots = np.asarray([landmark_rng.choice(layers[d]) for d in levels])
                    extra = landmark_batch(env, roots, heldout, landmark_rng, opts["goal_depth"])
                    if extra is not None:
                        lm_huber, lm_rank = batch_losses(model, torch.as_tensor(env.features(extra["states"])), extra, config)
                        loss = loss + options["landmark_weight"] * (lm_huber + options["rank_weight"] * lm_rank)
                        point_labels += len(extra["labels"])
                        rank_tasks += len(extra["goals"])
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), options["clip_norm"])
                optimizer.step()
                if step == 0 or (step + 1) % options["log_every"] == 0:
                    row = {"condition": condition, "step": step + 1,
                           "seconds": round(time.monotonic() - started, 2),
                           "huber": float(huber.detach()), "rank": float(rank.detach()),
                           "landmark_huber": float(lm_huber.detach()) if lm_huber is not None else None,
                           "landmark_rank": float(lm_rank.detach()) if lm_rank is not None else None}
                    log.write(json.dumps(row) + "\n")
                    log.flush()
                    print(json.dumps(row), flush=True)
        torch.save({"model": model.state_dict(), "model_config": saved["model_config"],
                    "cap": saved["cap"], "source_commit": source_commit,
                    "initial_epoch": saved["epoch"], "steps": options["steps"],
                    "condition": condition, "optimizer": optimizer.state_dict()}, out / f"{condition}.pt")
        result["conditions"][condition] = {"seconds": round(time.monotonic() - started, 2),
                                          "landmark_labels": point_labels, "landmark_rank_tasks": rank_tasks,
                                          **inspect(model)}
        (out / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps({"condition": condition,
                          "ten_to_seven": result["conditions"][condition]["ten_to_seven"]}), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(__doc__)
    for key in ("contract", "checkpoint", "oracle", "out"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    args = parser.parse_args()
    run(json.loads(args.contract.read_text()), args.checkpoint, args.oracle, args.out, args.source_commit)


if __name__ == "__main__":
    main()
