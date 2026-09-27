"""Fine-tune a frozen-run Q encoder on exact rankings of sibling actions."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .model import BoardEncoder, board_bits, distance, strata
from .multiboard_eval import encoded_values, goal_rollouts, listwise_groups, pair_scores, split_metrics
from .multiboard_post_eval import matched_supported_tasks
from .multiboard_run import rank_loss, atomic_save
from .oracle import DistanceOracle, successors
from .train import draw


def decision_groups(data, count, seed, oracle):
    """One true best successor and three alternatives for a reachable nonempty goal."""
    rng = np.random.default_rng(seed)
    masks = [int(mask) for mask in data["states"]]
    ids = {mask: index for index, mask in enumerate(masks)}
    pairs = data["train"]
    eligible = np.flatnonzero((pairs[:, 2] >= 2) & (data["states"][pairs[:, 1]] != "0"))
    rng.shuffle(eligible)
    groups = []
    families = {"removed_goal_cell": 0, "uncoverable_difference": 0,
                "reachable_detour": 0}
    for index in eligible:
        source_id, goal_id, steps, _ = pairs[index]
        source, goal = masks[source_id], masks[goal_id]
        choices = successors(source)
        optimal, alternatives = [], {name: [] for name in families}
        for _, next_mask in choices:
            if next_mask & goal != goal:
                alternatives["removed_goal_cell"].append(next_mask)
                continue
            next_distance = oracle.distance(next_mask, goal)
            if next_distance == steps - 1:
                optimal.append(next_mask)
            elif next_distance < 0:
                alternatives["uncoverable_difference"].append(next_mask)
            else:
                alternatives["reachable_detour"].append(next_mask)
        if not optimal or sum(map(len, alternatives.values())) < 3:
            continue
        chosen = [optimal[int(rng.integers(len(optimal)))]]
        for family, pool in alternatives.items():
            if pool and len(chosen) < 4:
                chosen.append(pool[int(rng.integers(len(pool)))])
                families[family] += 1
        remaining = [mask for pool in alternatives.values() for mask in pool if mask not in chosen]
        rng.shuffle(remaining)
        chosen.extend(remaining[:4 - len(chosen)])
        if len(chosen) != 4:
            continue
        for mask in chosen:
            if mask not in ids:
                ids[mask] = len(masks)
                masks.append(mask)
        groups.append([*(ids[mask] for mask in chosen), int(goal_id)])
        if len(groups) == count:
            break
    if len(groups) < count:
        raise ValueError(f"Only {len(groups)} complete decision groups for requested {count}")
    return np.asarray(groups, dtype=np.int32), masks, families


def evaluation_tasks(data, seed):
    isolated = data["isolated_goal_tasks"]
    states = data["states"]
    nonempty = lambda rows: rows[(rows[:, 2] >= 2) & (states[rows[:, 1]] != "0")]
    known_targets = set(data["train"][:, 1].tolist())
    seen = nonempty(data["test_seen_pair"])
    seen = seen[np.isin(seen[:, 1], list(known_targets))]
    isolated_targets = set(data["test_isolated_goal"][:, 1].tolist())
    supported = data["test_unseen_goal"]
    supported = nonempty(supported[~np.isin(supported[:, 1], list(isolated_targets))])
    pools = {"validation": nonempty(data["validation_unseen_state"]),
             "known_goal_unseen_pair": seen, "supported_unseen_goal": supported,
             "isolated_unseen_goal": isolated, "ood_board": nonempty(data["test_ood_board"])}
    tasks = {"isolated_unseen_goal": isolated}
    for offset, (name, pool) in enumerate(pools.items()):
        if name == "isolated_unseen_goal":
            continue
        chosen = matched_supported_tasks(pool, isolated, seed + offset)
        if len(chosen) != len(isolated):
            raise ValueError(f"Could not match {name} to isolated task distances")
        tasks[name] = chosen
    return tasks


def run(base_dir, out, config, source_commit, device):
    base_dir, out = Path(base_dir), Path(out)
    out.mkdir(parents=True, exist_ok=False)
    (out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    data = dict(np.load(base_dir / "data" / "data.npz", allow_pickle=False))
    saved = torch.load(base_dir / "best.pt", map_location=device, weights_only=True)
    if saved["source_commit"] != config["base_source_commit"]:
        raise ValueError("Base model source commit differs from contract")
    torch.manual_seed(config["seed"])
    torch.set_num_threads(4)
    device = torch.device(device)
    metric = saved["metric"]
    model = BoardEncoder(saved["config"]["model"]["state_dim"],
                         saved["config"]["model"]["encoder_hidden_dim"],
                         saved["cap"], metric).to(device)
    model.load_state_dict(saved["model"])
    oracle = DistanceOracle(cache_limit=config["oracle_cache_limit"],
                            seconds=config["oracle_seconds"])
    groups_np, masks, families = decision_groups(data, config["decision_groups"],
                                                  config["seed"], oracle)
    del oracle
    bits = board_bits(masks).to(device)
    groups = torch.as_tensor(groups_np, device=device, dtype=torch.long)
    train_pairs = torch.as_tensor(data["train"], device=device, dtype=torch.long)
    pair_strata = [(ids.to(device), weight) for ids, weight in strata(train_pairs)]
    rank_train = torch.as_tensor(listwise_groups(data["train"], config["seed"],
                                                per_anchor=4), device=device)
    rank_val = listwise_groups(data["validation_unseen_state"], config["seed"] + 1,
                               per_anchor=4)
    target = torch.where(train_pairs[:, 2] >= 0, train_pairs[:, 2], saved["cap"]).float()
    tasks = evaluation_tasks(data, config["seed"])
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])

    def validation():
        values = encoded_values(model, bits[:len(data["states"])])
        scores = pair_scores(values, data["validation_unseen_state"], metric)
        metrics = split_metrics(data["validation_unseen_state"], scores, rank_val, saved["cap"])
        rollout = goal_rollouts(model, tasks["validation"][:, :2], data["states"], metric, device)
        return metrics["distance_loss"] + metrics["listwise_loss"], rollout["reached_goal"]

    base_val_loss, base_val_reached = validation()
    best_step, best_val = 0, (base_val_reached, -base_val_loss)
    atomic_save({**saved, "finetune_source_commit": source_commit, "finetune_step": 0}, out / "best.pt")
    progress = [{"step": 0, "validation_objective": base_val_loss,
                 "validation_reached": base_val_reached}]
    for step in range(1, config["steps"] + 1):
        model.train()
        pair_ids = draw(pair_strata, config["pair_batch_size"], device)
        pair_loss = F.smooth_l1_loss(model(bits, train_pairs[pair_ids, :2]), target[pair_ids])
        old_rank_loss = rank_loss(model, bits, train_pairs, rank_train, saved["cap"],
                                  config["listwise_batch_size"], 1.0)
        selected = groups[torch.randint(len(groups), (config["decision_batch_size"],), device=device)]
        choices = model.encode(bits[selected[:, :4]])
        goal = model.encode(bits[selected[:, 4]]).unsqueeze(1)
        scores = distance(choices, goal, metric)
        decision_loss = F.cross_entropy(-scores, torch.zeros(len(selected),
                                                             device=device, dtype=torch.long))
        loss = pair_loss + config["listwise_weight"] * old_rank_loss + \
            config["decision_weight"] * decision_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step % config["eval_every"] == 0 or step == config["steps"]:
            model.eval()
            val_loss, val_reached = validation()
            progress.append({"step": step, "validation_objective": val_loss,
                             "validation_reached": val_reached})
            if (val_reached, -val_loss) > best_val:
                best_step, best_val = step, (val_reached, -val_loss)
                atomic_save({**saved, "model": model.state_dict(),
                             "finetune_source_commit": source_commit,
                             "finetune_step": step}, out / "best.pt")
    (out / "progress.json").write_text(json.dumps(progress, indent=2) + "\n")
    baseline_model = BoardEncoder(saved["config"]["model"]["state_dim"],
                                  saved["config"]["model"]["encoder_hidden_dim"],
                                  saved["cap"], metric).to(device)
    baseline_model.load_state_dict(saved["model"])
    tuned = torch.load(out / "best.pt", map_location=device, weights_only=True)
    model.load_state_dict(tuned["model"])
    result = {"base_source_commit": saved["source_commit"],
              "finetune_source_commit": source_commit, "decision_groups": len(groups),
              "group_families": families, "best_step": best_step,
              "base_validation": progress[0],
              "best_validation": next(item for item in progress if item["step"] == best_step),
              "rollouts": {}}
    for name, rows in tasks.items():
        result["rollouts"][name] = {
            "base": goal_rollouts(baseline_model, rows[:, :2], data["states"], metric, device),
            "tuned": goal_rollouts(model, rows[:, :2], data["states"], metric, device)}
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
    run(args.base_run, args.out, json.loads(args.config.read_text()), args.source_commit,
        args.device)


if __name__ == "__main__":
    main()
