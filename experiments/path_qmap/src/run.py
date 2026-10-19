"""Train the shared state MLP using distance and full candidate soft ranking."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from grounded_llm.core.qmap.model import StateEncoder
from grounded_llm.tasks.blocks.sampling import strata
from grounded_llm.core.qmap.model import soft_rank_loss
from grounded_llm.core.files import atomic_save
from grounded_llm.core.qmap.sampling import draw
from grounded_llm.tasks.path.environment import environment_from_suite

from grounded_llm.tasks.path.data import prepare
from grounded_llm.tasks.path.evaluate import distance_table, final_metrics, relation_metrics, rank_objective


def train(config, out, source_commit, device="cpu"):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    (out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    env = environment_from_suite(json.loads(Path(config["graph_suite"]).read_text()))
    data = prepare(env.adjacency, config)
    np.savez_compressed(out / "data.npz", **data, actions=env.actions)
    options, spec = config["training"], config["model"]
    torch.set_num_threads(options["cpu_threads"])
    torch.manual_seed(options["seed"])
    size = len(env.adjacency)
    cap = int(data["distances"].max())
    model = StateEncoder(spec["state_dim"], spec["encoder_hidden_dim"], cap,
                         spec["metric"], input_dim=size).to(device)
    features = torch.eye(size, device=device)
    pairs = torch.as_tensor(data["train_pairs"], dtype=torch.long, device=device)
    groups = [(ids.to(device), weight) for ids, weight in strata(pairs)]
    tasks = torch.as_tensor(data["train_tasks"], dtype=torch.long, device=device)
    children = torch.as_tensor(data["children"], dtype=torch.long, device=device)
    truth = torch.as_tensor(data["distances"], device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=options["learning_rate"])
    initial = final_metrics(distance_table(model, features), data, config)
    best, best_step = float("inf"), 0
    started = time.monotonic()

    def save(step, filename):
        atomic_save({"model_type": "state_q_path", "model": model.state_dict(),
                     "optimizer": optimizer.state_dict(), "step": step, "cap": cap,
                     "config": config, "adjacency": torch.from_numpy(env.adjacency.copy()),
                     "source_commit": source_commit}, out / filename)

    with (out / "progress.jsonl").open("w") as log:
        for step in range(1, options["steps"] + 1):
            model.train()
            q = model.encode(features)
            rows = pairs[draw(groups, options["batch_size"], device)]
            pointwise = F.smooth_l1_loss(model.distance(q[rows[:, 0]], q[rows[:, 1]]), rows[:, 2].float())
            batch = tasks[torch.randint(len(tasks), (options["listwise_batch_size"],), device=device)]
            candidates = children[batch[:, 0]]
            valid = candidates >= 0
            safe = candidates.clamp_min(0)
            scores = model.distance(q[safe], q[batch[:, 1, None]])
            ranking = soft_rank_loss(scores, truth[safe, batch[:, 1, None]], valid, options["temperature"])
            loss = pointwise + options["listwise_weight"] * ranking
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradient_norm = sum(float(p.grad.square().sum()) for p in model.parameters() if p.grad is not None) ** .5
            optimizer.step()
            if step == 1 or step % options["eval_every"] == 0 or step == options["steps"]:
                predicted = distance_table(model, features)
                validation = relation_metrics(predicted, data["validation_pairs"])
                val_rank = rank_objective(predicted, data, data["validation_tasks"], options["temperature"])
                objective = validation["huber"] + options["listwise_weight"] * val_rank
                if objective < best:
                    best, best_step = objective, step
                    save(step, "best.pt")
                record = {"step": step, "seconds": round(time.monotonic() - started, 2),
                          "distance_loss": float(pointwise.detach()), "rank_loss": float(ranking.detach()),
                          "validation_mae": validation["mae"], "validation_objective": objective,
                          "gradient_norm_before_step": gradient_norm,
                          "learning_rate": optimizer.param_groups[0]["lr"], "best_step": best_step}
                log.write(json.dumps(record) + "\n")
                log.flush()
                print(json.dumps(record), flush=True)
    save(step, "last.pt")
    model.load_state_dict(torch.load(out / "best.pt", map_location=device, weights_only=True)["model"])
    predicted = distance_table(model, features)
    result = {"source_commit": source_commit, "best_step": best_step, "last_step": step,
              "training_seconds": round(time.monotonic() - started, 2),
              "state_count": size, "directed_edges": int(env.adjacency.sum()), "diameter": cap,
              "train_pairs": len(pairs), "train_decisions": len(tasks),
              "initial": initial, "final": final_metrics(predicted, data, config),
              "oracle_check": final_metrics(data["distances"].astype(np.float32), data, config)["all_routes"]}
    np.savez_compressed(out / "q_vectors.npz", q=model.encode(features).detach().cpu().numpy())
    (out / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"complete": str(out), "best_step": best_step,
                      "all_routes": result["final"]["all_routes"],
                      "heldout_routes": result["final"]["test"]["routes"]}), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    train(json.loads(args.config.read_text()), args.out, args.source_commit, args.device)


if __name__ == "__main__":
    main()
