"""Resume the same Q and Adam state with clear-goal and long-fork supervision."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from .model import BoardEncoder, board_bits, distance, strata
from .multiboard_eval import encoded_values, listwise_groups, pair_scores, split_metrics
from .multiboard_run import atomic_save, rank_loss
from .train import draw
from .tree_continue_data import prepare


def development_clear(values, data, metric, goal_id):
    ids = torch.as_tensor(data["dev_clear_child_ids"], device=values.device)
    scores = distance(values[ids], values[goal_id].expand(len(ids), -1), metric)
    scores = scores.cpu().numpy()
    truth = data["dev_clear_child_distances"]
    offsets = data["dev_clear_offsets"]
    parent_steps = data["dev_clear_parent_distances"]
    reachable = shortest = 0
    for index, parent_distance in enumerate(parent_steps):
        start, end = int(offsets[index]), int(offsets[index + 1])
        choice = start + int(scores[start:end].argmin())
        reachable += truth[choice] >= 0
        shortest += truth[choice] == parent_distance - 1
    return {"boards": len(parent_steps), "reachable": int(reachable),
            "shortest": int(shortest), "reachable_acc": reachable / len(parent_steps),
            "shortest_acc": shortest / len(parent_steps)}


def train(contract, out, source_commit, device):
    out = Path(out)
    data = dict(np.load(out / "data" / "data.npz", allow_pickle=False))
    parent_path = Path(contract["parent_run"]) / contract["parent_checkpoint"]
    saved = torch.load(parent_path, map_location=device, weights_only=False)
    options = contract["training"]
    torch.manual_seed(options["seed"])
    torch.set_num_threads(options["cpu_threads"])
    device = torch.device(device)
    model_config = saved["config"]["model"]
    model = BoardEncoder(model_config["state_dim"], model_config["encoder_hidden_dim"],
                         saved["cap"], saved["metric"]).to(device)
    model.load_state_dict(saved["model"])
    optimizer = torch.optim.Adam(model.parameters(), lr=options["learning_rate"])
    optimizer.load_state_dict(saved["optimizer"])
    for group in optimizer.param_groups:
        group["lr"] = options["learning_rate"]
    metric, cap = saved["metric"], saved["cap"]
    parent_step = int(saved["step"])
    bits = board_bits(data["states"]).to(device)
    pairs_np = data["train"]
    pairs = torch.as_tensor(pairs_np, dtype=torch.long, device=device)
    target = torch.where(pairs[:, 2] >= 0, pairs[:, 2], cap).float()
    old_count = json.loads((out / "data" / "summary.json").read_text())["old_train_pairs"]
    old_strata = [(ids.to(device), weight) for ids, weight in strata(pairs[:old_count])]
    clear_pair_ids = torch.as_tensor(data["clear_pair_ids"], device=device)
    long_pair_ids = torch.as_tensor(data["long_pair_ids"], device=device)
    old_rank = torch.as_tensor(listwise_groups(
        pairs_np[:old_count], options["seed"], per_anchor=4), device=device)
    clear_rank = torch.as_tensor(data["clear_rank_groups"], device=device)
    long_rank = torch.as_tensor(data["long_rank_groups"], device=device)
    val_rows = data["validation_unseen_state"]
    val_rank = listwise_groups(val_rows, options["seed"] + 1, per_anchor=4)
    goal_id = next(index for index, mask in enumerate(data["states"]) if int(mask) == 0)
    checkpoints = out / "checkpoints"
    checkpoints.mkdir(exist_ok=True)
    model_record = {"model": model_config, "training": options,
                    "parent_checkpoint": str(parent_path)}

    def save(step, path):
        atomic_save({"model_type": "board_mlp", "model": model.state_dict(),
                     "optimizer": optimizer.state_dict(), "step": parent_step + step,
                     "continuation_step": step, "parent_step": parent_step,
                     "metric": metric, "cap": cap, "config": model_record,
                     "source_commit": source_commit}, path)

    def evaluate(step):
        values = encoded_values(model, bits)
        val = split_metrics(val_rows, pair_scores(values, val_rows, metric), val_rank, cap)
        clear = development_clear(values, data, metric, goal_id)
        # A shared Q must retain multi-goal geometry while improving the
        # previously missing empty-goal first-step decision.
        selection = .5 * val["listwise_best_accuracy"] + .5 * clear["shortest_acc"]
        return selection, val, clear

    best_score, initial_val, initial_clear = evaluate(0)
    best_step, stale = 0, 0
    save(0, out / "best.pt")
    print(json.dumps({"continuation_step": 0, "selection": best_score,
                      "validation": initial_val, "dev_clear": initial_clear}), flush=True)
    started = time.monotonic()
    normal_fraction = 1 - options["clear_pair_fraction"] - options["long_pair_fraction"]
    normal_rank_fraction = 1 - options["clear_group_fraction"] - options["long_group_fraction"]
    with (out / "progress.jsonl").open("w") as log:
        for step in range(1, options["steps"] + 1):
            model.train()
            batch = options["batch_size"]
            old_ids = draw(old_strata, round(batch * normal_fraction), device)
            clear_ids = clear_pair_ids[torch.randint(len(clear_pair_ids),
                                                     (round(batch * options["clear_pair_fraction"]),),
                                                     device=device)]
            long_ids = long_pair_ids[torch.randint(len(long_pair_ids),
                                                   (batch - len(old_ids) - len(clear_ids),),
                                                   device=device)]
            ids = torch.cat([old_ids, clear_ids, long_ids])
            pointwise = F.smooth_l1_loss(model(bits, pairs[ids, :2]), target[ids])
            rank_batch = options["listwise_batch_size"]
            rank_parts = [
                (old_rank, round(rank_batch * normal_rank_fraction)),
                (clear_rank, round(rank_batch * options["clear_group_fraction"])),
                (long_rank, rank_batch - round(rank_batch * normal_rank_fraction)
                 - round(rank_batch * options["clear_group_fraction"]))]
            listwise = sum(rank_loss(model, bits, pairs, group, cap, count,
                                     options["temperature"]) * count / rank_batch
                           for group, count in rank_parts if count)
            loss = pointwise + options["listwise_weight"] * listwise
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            if step % options["eval_every"] == 0 or step == options["steps"]:
                selection, val, clear = evaluate(step)
                if selection > best_score + 1e-4:
                    best_score, best_step, stale = selection, step, 0
                    save(step, out / "best.pt")
                else:
                    stale += 1
                record = {"continuation_step": step, "total_step": parent_step + step,
                          "seconds": round(time.monotonic() - started, 1),
                          "pointwise_loss": float(pointwise.detach()),
                          "listwise_loss": float(listwise.detach()),
                          "selection": selection, "best_step": best_step,
                          "validation": val, "dev_clear": clear}
                log.write(json.dumps(record) + "\n")
                log.flush()
                print(json.dumps(record), flush=True)
                if step % options["checkpoint_every"] == 0:
                    save(step, checkpoints / f"step_{step:06d}.pt")
                if stale >= options["patience_evaluations"]:
                    break
    result = {"source_commit": source_commit, "parent_checkpoint": str(parent_path),
              "parent_step": parent_step, "best_continuation_step": best_step,
              "last_continuation_step": step, "training_seconds": time.monotonic() - started,
              "initial_validation": initial_val, "initial_dev_clear": initial_clear,
              "best_selection": best_score, "data": json.loads(
                  (out / "data" / "summary.json").read_text())}
    (out / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"complete": str(out), "best_step": best_step}), flush=True)


def run(contract, out, source_commit, device, max_boards=None, prepare_only=False,
        train_only=False):
    out = Path(out)
    if not train_only:
        out.mkdir(parents=True, exist_ok=False)
        (out / "config.json").write_text(json.dumps(contract, indent=2) + "\n")
        (out / "source_commit.txt").write_text(source_commit + "\n")
        prepare(Path(contract["parent_run"]) / "data" / "data.npz",
                Path(contract["official_dataset"]), out / "data", contract,
                max_boards=max_boards)
    if not prepare_only and max_boards is None:
        train(contract, out, source_commit, device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-boards", type=int)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--train-only", action="store_true")
    args = parser.parse_args()
    run(json.loads(args.config.read_text()), args.out, args.source_commit, args.device,
        args.max_boards, args.prepare_only, args.train_only)


if __name__ == "__main__":
    main()
