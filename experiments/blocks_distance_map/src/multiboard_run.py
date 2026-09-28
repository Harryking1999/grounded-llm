"""Shared 1000-board Q training with intermediate checkpoints and sealed OOD tests."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from .model import BoardEncoder, board_bits, distance, strata
from .multiboard_eval import (contrast_metrics, encoded_values, goal_rollouts,
                              listwise_groups, pair_scores, split_metrics)


def draw(groups, size, device):
    parts, remaining = [], size
    for index, (ids, weight) in enumerate(groups):
        count = remaining if index == len(groups) - 1 else round(size * weight)
        remaining -= count
        parts.append(ids[torch.randint(len(ids), (count,), device=device)])
    return torch.cat(parts)


def rank_loss(model, bits, pairs, groups, cap, count, temperature):
    chosen = groups[torch.randint(len(groups), (count,), device=groups.device)]
    rows = pairs[chosen]
    predicted = model(bits, rows[..., :2].reshape(-1, 2)).reshape(-1, chosen.shape[1])
    truth = torch.where(rows[..., 2] >= 0, rows[..., 2], cap).float()
    target = torch.softmax(-truth / temperature, dim=1)
    return -(target * torch.log_softmax(-predicted / temperature, dim=1)).sum(1).mean()


def atomic_save(payload, path):
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def run(config, out, source_commit, device):
    from .multiboard_data import prepare

    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    (out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    (out / "source_commit.txt").write_text(source_commit + "\n")
    prepare(config, out / "data")
    train_prepared(config, out, source_commit, device)


def train_prepared(config, out, source_commit, device):
    """Train the original Q/loss on an already frozen relation dataset."""
    out = Path(out)
    data = dict(np.load(out / "data" / "data.npz", allow_pickle=False))
    options = config["training"]
    torch.manual_seed(options["seed"])
    torch.set_num_threads(options["cpu_threads"])
    device = torch.device(device)
    bits = board_bits(data["states"]).to(device)
    cap = int(data["cap"])
    metric = config["model"]["metric"]
    model = BoardEncoder(config["model"]["state_dim"], config["model"]["encoder_hidden_dim"],
                         cap, metric).to(device)
    train_pairs_np = data["train"]
    train_pairs = torch.as_tensor(train_pairs_np, dtype=torch.long, device=device)
    groups = [(ids.to(device), weight) for ids, weight in strata(train_pairs)]
    rank_train = torch.as_tensor(listwise_groups(train_pairs_np, options["seed"],
                                                   per_anchor=options["listwise_groups_per_anchor"]), device=device)
    rank_val = listwise_groups(data["validation_unseen_state"], options["seed"] + 1,
                               per_anchor=options["listwise_groups_per_anchor"])
    if not len(rank_train) or not len(rank_val):
        raise ValueError("Multi-goal listwise supervision or validation is empty")
    target = torch.where(train_pairs[:, 2] >= 0, train_pairs[:, 2], cap).float()
    optimizer = torch.optim.Adam(model.parameters(), lr=options["learning_rate"])
    checkpoints = out / "checkpoints"
    checkpoints.mkdir()
    best_score, best_step, stale = float("inf"), 0, 0
    started = time.monotonic()

    def checkpoint(step, destination):
        atomic_save({"model_type": "board_mlp", "model": model.state_dict(),
                     "optimizer": optimizer.state_dict(), "step": step, "metric": metric,
                     "cap": cap, "config": config, "source_commit": source_commit}, destination)

    with (out / "progress.jsonl").open("w") as log:
        for step in range(1, options["steps"] + 1):
            model.train()
            ids = draw(groups, options["batch_size"], device)
            predicted = model(bits, train_pairs[ids, :2])
            pointwise = F.smooth_l1_loss(predicted, target[ids])
            listwise = rank_loss(model, bits, train_pairs, rank_train, cap,
                                 options["listwise_batch_size"], options["temperature"])
            loss = pointwise + options["listwise_weight"] * listwise
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            if step == 1 or step % options["eval_every"] == 0 or step == options["steps"]:
                values = encoded_values(model, bits)
                val_scores = pair_scores(values, data["validation_unseen_state"], metric)
                validation = split_metrics(data["validation_unseen_state"], val_scores, rank_val, cap)
                objective = validation["distance_loss"] + options["listwise_weight"] * validation["listwise_loss"]
                improved = objective < best_score - options["min_improvement"]
                if improved:
                    best_score, best_step, stale = objective, step, 0
                    checkpoint(step, out / "best.pt")
                else:
                    stale += 1
                record = {"step": step, "seconds": round(time.monotonic() - started, 2),
                          "pointwise_batch_loss": float(pointwise.detach()),
                          "listwise_batch_loss": float(listwise.detach()),
                          "validation_objective": objective, "best_step": best_step,
                          "validation": validation}
                log.write(json.dumps(record) + "\n")
                log.flush()
                print(json.dumps(record), flush=True)
                if step % options["checkpoint_every"] == 0:
                    checkpoint(step, checkpoints / f"step_{step:06d}.pt")
                if stale >= options["patience_evaluations"]:
                    break
    saved = torch.load(out / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(saved["model"])
    values = encoded_values(model, bits)
    splits = {}
    for name in ("train", "validation_unseen_state", "test_seen_pair",
                 "test_unseen_goal", "test_isolated_goal", "test_ood_board"):
        pairs = data[name]
        ranking = listwise_groups(pairs, options["seed"] + len(name),
                                  per_anchor=options["listwise_groups_per_anchor"])
        splits[name] = split_metrics(pairs, pair_scores(values, pairs, metric), ranking, cap)
    contrast_tasks = [(parent, goal) for parent, _, _, ga, gb in data["contrast_test"]
                      for goal in (ga, gb)]
    result = {"source_commit": source_commit, "best_step": best_step, "last_step": step,
              "training_seconds": time.monotonic() - started,
              "torch_version": torch.__version__, "numpy_version": np.__version__,
              "splits": splits,
              "contrast_train": contrast_metrics(values, data["contrast_train"], metric),
              "contrast_test": contrast_metrics(values, data["contrast_test"], metric),
              "heldout_goal_rollout": goal_rollouts(model, contrast_tasks, data["states"], metric, device),
              "isolated_goal_rollout": goal_rollouts(model, data["isolated_goal_tasks"][:, :2],
                                                     data["states"], metric, device)}
    area = torch.as_tensor([[int(mask).bit_count()] for mask in data["states"]],
                           dtype=torch.float32, device=device)
    result["area_only_baseline"] = {"splits": {},
                                    "contrast_test": contrast_metrics(area, data["contrast_test"], "directed_sum")}
    for name in ("test_seen_pair", "test_unseen_goal", "test_isolated_goal", "test_ood_board"):
        pairs = data[name]
        ranking = listwise_groups(pairs, options["seed"] + len(name),
                                  per_anchor=options["listwise_groups_per_anchor"])
        result["area_only_baseline"]["splits"][name] = split_metrics(
            pairs, pair_scores(area, pairs, "directed_sum"), ranking, cap)
    (out / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"complete": str(out), "best_step": best_step,
                      "splits": {name: scores["pairs"] for name, scores in splits.items()},
                      "contrast_test": result["contrast_test"]}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    run(json.loads(args.config.read_text()), args.out, args.source_commit, args.device)


if __name__ == "__main__":
    main()
