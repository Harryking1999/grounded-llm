"""Fine-tune Q on the easiest certified mistakes among all legal successors."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from experiments.gcml_counterexamples.src import blocks
from .model import BoardEncoder, board_bits, distance, strata
from .multiboard_eval import goal_rollouts, listwise_groups
from .multiboard_run import atomic_save, rank_loss
from .oracle import successors
from .train import draw


def decision_groups(data):
    """One reachable child versus every certified bad child for each train case."""
    masks = [int(mask) for mask in data["states"]]
    ids = {mask: index for index, mask in enumerate(masks)}
    trained = set(data["train_state_ids"].tolist())
    protected = {masks[int(index)]
                 for name in ("validation_unseen_state", "test_ood_board")
                 for row in data[name] for index in row[:2] if index not in trained}
    protected.update(masks[int(index)]
                     for name in ("hard_validation_cases", "hard_ood_board_cases",
                                  "hard_unseen_goal_cases")
                     for row in data[name] for index in row[:4] if index not in trained)
    records = []
    counts = {"removed_goal_cell": 0, "local_coverage_failure": 0,
              "globally_unreachable": 0, "excluded_heldout_state": 0}
    for parent_id, good_id, bad_id, goal_id, _ in data["hard_train_cases"]:
        parent, good, bad, goal = (masks[int(index)]
                                  for index in (parent_id, good_id, bad_id, goal_id))
        negatives = set()
        for _, child in successors(parent):
            if child == good or child in protected:
                counts["excluded_heldout_state"] += child in protected
                continue
            if child == bad:
                kind = "globally_unreachable"
            elif child & goal != goal:
                kind = "removed_goal_cell"
            elif not blocks.locally_supported(child ^ goal):
                kind = "local_coverage_failure"
            else:
                continue
            negatives.add(child)
            counts[kind] += 1
        if not negatives:
            continue
        for child in negatives:
            if child not in ids:
                ids[child] = len(masks)
                masks.append(child)
        records.append((int(good_id), int(goal_id), sorted(ids[child] for child in negatives)))
    width = max(map(lambda record: len(record[2]), records))
    positive = np.asarray([[good, goal] for good, goal, _ in records], dtype=np.int32)
    negative = np.full((len(records), width), -1, dtype=np.int32)
    for index, (_, _, bad_ids) in enumerate(records):
        negative[index, :len(bad_ids)] = bad_ids
    return np.asarray(masks, dtype=object), positive, negative, counts


def decision_loss(model, bits, positive, negative, selected, metric, margin):
    rows = positive[selected]
    bad_ids = negative[selected]
    good = model.encode(bits[rows[:, 0]])
    goal = model.encode(bits[rows[:, 1]])
    bad = model.encode(bits[bad_ids.clamp_min(0)])
    good_score = distance(good, goal, metric)
    bad_scores = distance(bad, goal[:, None, :], metric)
    hardest_bad = bad_scores.masked_fill(bad_ids < 0, float("inf")).amin(1)
    return F.softplus(margin + good_score - hardest_bad).mean()


def run(base_dir, out, config, source_commit, device):
    base_dir, out = Path(base_dir), Path(out)
    out.mkdir(parents=True, exist_ok=False)
    (out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    (out / "source_commit.txt").write_text(source_commit + "\n")
    data = dict(np.load(base_dir / "data" / "data.npz", allow_pickle=False))
    saved = torch.load(base_dir / "best.pt", map_location=device, weights_only=True)
    if saved["source_commit"] != config["base_source_commit"]:
        raise ValueError("Base checkpoint does not match formal contract")
    torch.manual_seed(config["seed"])
    torch.set_num_threads(config["cpu_threads"])
    device = torch.device(device)
    model = BoardEncoder(saved["config"]["model"]["state_dim"],
                         saved["config"]["model"]["encoder_hidden_dim"],
                         saved["cap"], saved["metric"]).to(device)
    model.load_state_dict(saved["model"])
    masks, positive_np, negative_np, counts = decision_groups(data)
    bits = board_bits(masks).to(device)
    positive = torch.as_tensor(positive_np, device=device, dtype=torch.long)
    negative = torch.as_tensor(negative_np, device=device, dtype=torch.long)
    pairs_np = data["train"]
    pairs = torch.as_tensor(pairs_np, device=device, dtype=torch.long)
    strata_groups = [(ids.to(device), weight) for ids, weight in strata(pairs)]
    rank_groups = torch.as_tensor(listwise_groups(
        pairs_np, config["seed"], per_anchor=config["listwise_groups_per_anchor"]),
        device=device)
    targets = torch.where(pairs[:, 2] >= 0, pairs[:, 2], saved["cap"]).float()
    validation = data["hard_validation_cases"][:, [0, 3]]
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    (out / "checkpoints").mkdir()

    def evaluate():
        model.eval()
        return goal_rollouts(model, validation, data["states"], saved["metric"], device)

    baseline = evaluate()
    best_step, best_reached = 0, baseline["reached_goal"]
    atomic_save({**saved, "finetune_config": config,
                 "finetune_source_commit": source_commit,
                 "finetune_step": 0}, out / "best.pt")
    progress = [{"step": 0, "validation_reached": best_reached}]
    for step in range(1, config["steps"] + 1):
        model.train()
        pair_ids = draw(strata_groups, config["pair_batch_size"], device)
        pair_loss = F.smooth_l1_loss(model(bits, pairs[pair_ids, :2]), targets[pair_ids])
        regular_rank = rank_loss(model, bits, pairs, rank_groups, saved["cap"],
                                 config["listwise_batch_size"], config["temperature"])
        selected = torch.randint(len(positive), (config["decision_batch_size"],), device=device)
        choice_loss = decision_loss(model, bits, positive, negative, selected,
                                    saved["metric"], config["margin"])
        loss = pair_loss + config["listwise_weight"] * regular_rank + \
            config["decision_weight"] * choice_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step % config["eval_every"] == 0 or step == config["steps"]:
            score = evaluate()
            entry = {"step": step, "validation_reached": score["reached_goal"],
                     "pair_loss": float(pair_loss.detach()),
                     "rank_loss": float(regular_rank.detach()),
                     "choice_loss": float(choice_loss.detach())}
            progress.append(entry)
            print(json.dumps(entry), flush=True)
            payload = {**saved, "model": model.state_dict(),
                       "optimizer": optimizer.state_dict(), "finetune_config": config,
                       "finetune_source_commit": source_commit,
                       "finetune_step": step}
            if score["reached_goal"] > best_reached:
                best_step, best_reached = step, score["reached_goal"]
                atomic_save(payload, out / "best.pt")
            if step % config["checkpoint_every"] == 0:
                atomic_save(payload, out / "checkpoints" / f"step_{step:06d}.pt")
    chosen = torch.load(out / "best.pt", map_location=device, weights_only=True)
    model.load_state_dict(chosen["model"])
    result = {"source_commit": source_commit,
              "base_source_commit": saved["source_commit"],
              "decision_groups": len(positive), "additional_states": len(masks) - len(data["states"]),
              "negative_families": counts, "best_step": best_step,
              "validation_base": baseline, "validation_best": evaluate(),
              "progress": progress,
              "tests": {name: goal_rollouts(model, data[f"hard_{name}_cases"][:, [0, 3]],
                                            data["states"], saved["metric"], device)
                        for name in ("ood_board", "unseen_goal")}}
    (out / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    run(args.base_run, args.out, json.loads(args.config.read_text()),
        args.source_commit, args.device)


if __name__ == "__main__":
    main()
