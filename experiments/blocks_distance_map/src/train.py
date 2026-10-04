"""Train Q only. Test labels and held-out goal decisions never select checkpoints."""
import argparse
import copy
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from .evaluate import evaluate_split, final_metrics, predict_pairs
from .model import QMap, strata


def draw(groups, size, device):
    parts, remaining = [], size
    for i, (ids, weight) in enumerate(groups):
        count = remaining if i == len(groups) - 1 else round(size * weight)
        remaining -= count
        parts.append(ids[torch.randint(len(ids), (count,), device=device)])
    return torch.cat(parts)


def train(config, data_path, out, metric, seed, device, source_commit):
    out.mkdir(parents=True, exist_ok=False)
    options = config["training"]
    torch.manual_seed(seed)
    torch.set_num_threads(options["cpu_threads"])
    data = dict(np.load(data_path, allow_pickle=False))
    cap = int(data["cap"])
    model = QMap(len(data["states"]), config["model"]["state_dim"], cap, metric,
                 config["model"]["init_std"]).to(device)
    pairs = torch.as_tensor(data["pairs"], dtype=torch.long, device=device)
    train_ids = torch.as_tensor(np.flatnonzero(data["split"] == 0), device=device)
    groups = [(train_ids[ids], weight) for ids, weight in strata(pairs[train_ids])]
    rank_groups = [torch.as_tensor(data[f"train_{axis}"], device=device)
                   for axis in ("outgoing", "incoming")]
    rank_groups = [rows for rows in rank_groups if len(rows)]
    target = torch.where(pairs[:, 2] >= 0, pairs[:, 2], cap).float()
    optimizer = torch.optim.Adam(model.parameters(), lr=options["learning_rate"])
    # No test or decision labels are read by the training objective.
    initial_scores = predict_pairs(model, data["pairs"])
    initial_validation = evaluate_split(data, "validation", initial_scores, options["rank_margin"])
    initial = final_metrics(model, data, options["rank_margin"])
    best = initial_validation["distance_loss"] + options["rank_weight"] * initial_validation["ranking_loss"]
    best_state, best_step, stale = copy.deepcopy(model.state_dict()), 0, 0
    started = time.monotonic()
    history = []
    with (out / "progress.jsonl").open("w") as log:
        for step in range(1, options["steps"] + 1):
            model.train()
            ids = draw(groups, options["batch_size"], device)
            regression = F.smooth_l1_loss(model(pairs[ids]), target[ids])
            rank_losses = []
            for rows in rank_groups:
                sampled = rows[torch.randint(len(rows), (options["rank_batch_size"] // len(rank_groups),), device=device)]
                near, far = model(pairs[sampled[:, 0]]), model(pairs[sampled[:, 1]])
                rank_losses.append((options["rank_margin"] + near - far).clamp_min(0).mean())
            ranking = torch.stack(rank_losses).mean() if rank_losses else regression * 0
            loss = regression + options["rank_weight"] * ranking
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            model.project()
            if step == 1 or step % options["eval_every"] == 0 or step == options["steps"]:
                model.eval()
                scores = predict_pairs(model, data["pairs"])
                validation = evaluate_split(data, "validation", scores, options["rank_margin"])
                score = validation["distance_loss"] + options["rank_weight"] * validation["ranking_loss"]
                improved = score < best - options["min_improvement"]
                if improved:
                    best, best_step, stale = score, step, 0
                    best_state = copy.deepcopy(model.state_dict())
                else:
                    stale += 1
                record = {"step": step, "seconds": time.monotonic() - started,
                          "batch_distance_loss": float(regression.detach()),
                          "batch_rank_loss": float(ranking.detach()),
                          "validation_objective": score, "best_step": best_step,
                          "validation": validation}
                history.append(record)
                log.write(json.dumps(record) + "\n")
                log.flush()
                print(json.dumps({"metric": metric, "seed": seed, **record}), flush=True)
                if stale >= options["patience"]:
                    break
    model.load_state_dict(best_state)
    model.eval()
    result = {"metric": metric, "seed": seed, "source_commit": source_commit,
              "device": str(device), "torch_version": torch.__version__,
              "numpy_version": np.__version__, "best_step": best_step, "last_step": step,
              "training_seconds": time.monotonic() - started,
              "initial": initial, "final": final_metrics(model, data, options["rank_margin"])}
    torch.save({"model": model.cpu().state_dict(), "metric": metric, "cap": cap,
                "state_count": len(data["states"]), "config": config,
                "seed": seed, "source_commit": source_commit}, out / "q.pt")
    (out / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"complete": str(out), "best_step": best_step,
                      "test": result["final"]["test"],
                      "heldout_goal_decisions": result["final"]["heldout_goal_decisions"]}), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--metric", required=True, choices=("directed_max", "directed_sum", "euclidean"))
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--source-commit", required=True)
    args = parser.parse_args()
    train(json.loads(args.config.read_text()), args.data, args.out, args.metric,
          args.seed, args.device, args.source_commit)


if __name__ == "__main__":
    main()
