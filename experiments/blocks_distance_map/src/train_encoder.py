"""Q-only board encoder trained on the same certified pair relations as the table."""
import argparse
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import time

import numpy as np
import torch
from torch.nn import functional as F

from .evaluate import evaluate_split, final_metrics, predict_pairs
from .model import BoardEncoder, board_bits, distance, strata
from .train import draw


class EncodedView:
    """Expose frozen encoded Q rows to the shared evaluation routines."""
    def __init__(self, values, metric):
        self.q = SimpleNamespace(weight=values)
        self.metric = metric

    def __call__(self, pairs):
        return distance(self.q.weight[pairs[:, 0]], self.q.weight[pairs[:, 1]], self.metric)


@torch.no_grad()
def encoded_view(model, bit_table):
    model.eval()
    values = torch.cat([model.encode(chunk) for chunk in bit_table.split(2048)], dim=0)
    return EncodedView(values, model.metric)


def train(config, data_path, out, seed, device, source_commit):
    out.mkdir(parents=True, exist_ok=False)
    options = config["training"]
    torch.manual_seed(seed)
    torch.set_num_threads(options["cpu_threads"])
    data = dict(np.load(data_path, allow_pickle=False))
    metric = options["metrics"][0]
    cap = int(data["cap"])
    model = BoardEncoder(config["model"]["state_dim"], config["model"]["encoder_hidden_dim"],
                         cap, metric).to(device)
    bits = board_bits(data["states"]).to(device)
    pairs = torch.as_tensor(data["pairs"], dtype=torch.long, device=device)
    train_ids = torch.as_tensor(np.flatnonzero(data["split"] == 0), device=device)
    groups = [(train_ids[ids], weight) for ids, weight in strata(pairs[train_ids])]
    rank_groups = [torch.as_tensor(data[f"train_{axis}"], device=device)
                   for axis in ("outgoing", "incoming")]
    rank_groups = [rows for rows in rank_groups if len(rows)]
    target = torch.where(pairs[:, 2] >= 0, pairs[:, 2], cap).float()
    optimizer = torch.optim.Adam(model.parameters(), lr=options["learning_rate"])
    view = encoded_view(model, bits)
    initial_scores = predict_pairs(view, data["pairs"])
    initial_validation = evaluate_split(data, "validation", initial_scores, options["rank_margin"])
    initial = final_metrics(view, data, options["rank_margin"])
    best = initial_validation["distance_loss"] + options["rank_weight"] * initial_validation["ranking_loss"]
    best_state, best_step, stale = copy.deepcopy(model.state_dict()), 0, 0
    started = time.monotonic()
    with (out / "progress.jsonl").open("w") as log:
        for step in range(1, options["steps"] + 1):
            model.train()
            ids = draw(groups, options["batch_size"], device)
            regression = F.smooth_l1_loss(model(bits, pairs[ids]), target[ids])
            rank_losses = []
            for rows in rank_groups:
                sampled = rows[torch.randint(len(rows),
                    (options["rank_batch_size"] // len(rank_groups),), device=device)]
                near = model(bits, pairs[sampled[:, 0]])
                far = model(bits, pairs[sampled[:, 1]])
                rank_losses.append((options["rank_margin"] + near - far).clamp_min(0).mean())
            ranking = torch.stack(rank_losses).mean() if rank_losses else regression * 0
            loss = regression + options["rank_weight"] * ranking
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            if step == 1 or step % options["eval_every"] == 0 or step == options["steps"]:
                view = encoded_view(model, bits)
                scores = predict_pairs(view, data["pairs"])
                validation = evaluate_split(data, "validation", scores, options["rank_margin"])
                score = validation["distance_loss"] + options["rank_weight"] * validation["ranking_loss"]
                if score < best - options["min_improvement"]:
                    best, best_step, stale = score, step, 0
                    best_state = copy.deepcopy(model.state_dict())
                else:
                    stale += 1
                record = {"step": step, "seconds": time.monotonic() - started,
                          "batch_distance_loss": float(regression.detach()),
                          "batch_rank_loss": float(ranking.detach()),
                          "validation_objective": score, "best_step": best_step,
                          "validation": validation}
                log.write(json.dumps(record) + "\n")
                log.flush()
                print(json.dumps({"metric": metric, "seed": seed, **record}), flush=True)
                if stale >= options["patience"]:
                    break
    model.load_state_dict(best_state)
    view = encoded_view(model, bits)
    result = {"model_type": "board_mlp", "metric": metric, "seed": seed,
              "source_commit": source_commit, "device": str(device), "torch_version": torch.__version__,
              "numpy_version": np.__version__, "best_step": best_step, "last_step": step,
              "training_seconds": time.monotonic() - started,
              "initial": initial, "final": final_metrics(view, data, options["rank_margin"])}
    torch.save({"model_type": "board_mlp", "model": model.cpu().state_dict(),
                "metric": metric, "cap": cap, "config": config, "seed": seed,
                "source_commit": source_commit}, out / "q.pt")
    (out / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"complete": str(out), "best_step": best_step,
                      "test": result["final"]["test"],
                      "heldout_goal_decisions": result["final"]["heldout_goal_decisions"]}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--source-commit", required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if config["training"]["metrics"] != ["directed_sum"]:
        raise ValueError("Encoder pilot is contracted to directed_sum")
    train(config, args.data, args.out, args.seed, args.device, args.source_commit)


if __name__ == "__main__":
    main()
