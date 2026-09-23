"""Train state-conditioned residual dynamics on sampled legal Blocks transitions."""

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from experiments.external_map_interface.src.blocks_dynamics import BlocksDynamics
from experiments.external_map_interface.src.blocks_diagnostics import (
    candidate_diagnostic, q_scale, rollout_diagnostic, transition_metrics)
from experiments.external_map_interface.src.blocks_q_map import (
    collect_transitions, coverage_diagnostic, immediate_dead_diagnostic)
from experiments.sol_dag_blocks.src.tasks import TASKS


def split_transitions(rows, state_count, fraction, seed):
    """Hold out edges, retaining a spanning forest and every observed action.

    All Q rows remain connected by training constraints. This is held-out
    transition evaluation among observed states, not unseen-board generalization.
    """
    rng = np.random.default_rng(seed)
    parent = list(range(state_count))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    required, seen_actions = set(), set()
    for i in rng.permutation(len(rows)):
        source, action, dest = map(int, rows[i])
        left, right = root(source), root(dest)
        if left != right or action not in seen_actions:
            required.add(int(i))
            parent[left] = right
            seen_actions.add(action)
    candidates = [i for i in rng.permutation(len(rows)) if i not in required]
    heldout = np.array(sorted(candidates[:round(len(rows) * fraction)]), dtype=np.int64)
    train = np.array(sorted(set(range(len(rows))) - set(heldout)), dtype=np.int64)
    return train, heldout


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if config["objective"] != "transition_mse_only":
        raise ValueError("This trainer implements only the approved transition objective")
    if args.out.exists():
        raise FileExistsError(args.out)
    suite = json.loads(args.suite.read_text(encoding="utf-8"))
    case, = [c for c in suite["cases"] if c["id"] == args.case_id]
    if case["condition"] not in ("blocks8", "blocks12"):
        raise ValueError("Expected a Blocks case")
    task = TASKS[case["condition"]]
    states, actions, rows = collect_transitions(task, case, config["episodes"], config["seed"])
    train_ids, heldout_ids = split_transitions(rows, len(states), config["heldout_fraction"], config["seed"])
    torch.manual_seed(config["seed"])
    torch.set_num_threads(config["cpu_threads"])
    model = BlocksDynamics(len(states), len(actions), config).to(args.device)
    tensor_rows = torch.as_tensor(rows.astype(np.int64), device=args.device)
    train = tensor_rows[torch.as_tensor(train_ids, device=args.device)]
    heldout = tensor_rows[torch.as_tensor(heldout_ids, device=args.device)]
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    initial_scale = q_scale(model)
    initial_metrics = {"train": transition_metrics(model, train),
                       "heldout": transition_metrics(model, heldout)}
    args.out.mkdir(parents=True)
    np.savez_compressed(args.out / "transitions.npz", states=np.array([str(s) for s in states]),
                        actions=actions, transitions=rows, train_ids=train_ids, heldout_ids=heldout_ids)
    (args.out / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    started = time.monotonic()
    print(json.dumps({"states": len(states), "train": len(train), "heldout": len(heldout)}), flush=True)
    with (args.out / "progress.jsonl").open("w", encoding="utf-8") as progress:
        for epoch in range(1, config["epochs"] + 1):
            order = torch.randperm(len(train), device=args.device)
            model.train()
            for batch in order.split(config["batch_size"]):
                source, action, dest = train[batch].unbind(dim=1)
                # Both endpoint Q rows and the shared MLP receive gradients.
                loss = (model(source, action) - model.q(dest)).square().mean()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            if epoch == 1 or epoch % config["report_every"] == 0 or epoch == config["epochs"]:
                model.eval()
                record = {"epoch": epoch, "elapsed_seconds": time.monotonic() - started,
                          "train": transition_metrics(model, train),
                          "heldout": transition_metrics(model, heldout), "q_scale": q_scale(model)}
                line = json.dumps(record)
                progress.write(line + "\n")
                progress.flush()
                print(line, flush=True)
    model.eval()
    torch.save({"config": config, "state_count": len(states), "action_count": len(actions),
                "model": model.cpu().state_dict()}, args.out / "model.pt")
    q = model.q.weight.detach().numpy()
    coverage = coverage_diagnostic(task, states, rows[train_ids])
    coverage["observed_training_action_ids"] = coverage.pop("trained_action_rows")
    coverage["legal_candidates_with_observed_training_action"] = coverage.pop("legal_candidates_with_trained_V")
    summary = {"case_id": args.case_id, "states": len(states), "action_count": len(actions),
        "sampled_unique_transitions": len(rows), "initial_metrics": initial_metrics,
        "train": transition_metrics(model, torch.from_numpy(rows[train_ids].astype(np.int64))),
        "heldout": transition_metrics(model, torch.from_numpy(rows[heldout_ids].astype(np.int64))),
        "initial_q_scale": initial_scale, "final_q_scale": q_scale(model),
        "training_seconds": time.monotonic() - started,
        "coverage": coverage,
        "diagnostic": immediate_dead_diagnostic(task, case, states, q),
        "candidate_diagnostic": candidate_diagnostic(model, task, states, rows),
        "rollout_by_depth": rollout_diagnostic(model, task, case, states, rows[train_ids],
            config["diagnostic_episodes"], config["diagnostic_seed"])}
    (args.out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
                                          encoding="utf-8")
    print(json.dumps({"complete": str(args.out), "train": summary["train"],
                      "heldout": summary["heldout"]}), flush=True)


if __name__ == "__main__":
    main()
