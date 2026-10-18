"""Puzzle8-only complete-state relation training with the established Qmap objective."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from torch.nn import functional as F
from grounded_llm.qmap_sampling import CoverageSampler, task_indices
from grounded_llm.qmap_metrics import pair_metrics, decisions, ranking_loss
from .environment import Puzzle8
from .data import build
from .model import QMap, MODEL_TYPE, soft_rank_loss
from .evaluate import evaluate


def train(env, config, data, out, device, source_commit):
    torch.manual_seed(config["seed"])
    options = config["training"]
    torch.set_num_threads(options["cpu_threads"])
    rng = np.random.default_rng(config["seed"] + 1)
    model = QMap(**config["model"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=options["learning_rate"])
    pairs, tasks = data["pairs_train"], data["tasks_train"]
    ps = CoverageSampler(pairs[:, 2], rng, len(pairs), .5)
    rs = CoverageSampler(data["parent_distances_train"], rng, len(tasks), .5)
    for sampler, batch in ((ps, options["batch_size"]), (rs, options["decision_batch_size"])):
        if options["steps"] * round(batch * .5) < sampler.broad_count:
            raise ValueError("Budget must cover every training relation and decision")
    features = torch.as_tensor(env.features(np.arange(env.size)), device=device)
    def q(ids):
        return model.encode(features[torch.as_tensor(ids, device=device, dtype=torch.long)])
    idx = task_indices(env, data, "validation", config["evaluation"]["decision_count"], config)
    vt = data["tasks_validation"][idx]
    best, best_step = float("inf"), 0
    started = time.monotonic()
    def save(step, path):
        torch.save({"model_type": MODEL_TYPE, "task": env.name, "actions": list(env.actions),
                    "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "config": config, "source_commit": source_commit, "step": step,
                    "coverage": {"pair_passes": ps.passes, "decision_passes": rs.passes}}, path)
    with (out / "progress.jsonl").open("w") as log:
        for step in range(1, options["steps"] + 1):
            model.train()
            rows = pairs[ps.draw(options["batch_size"])]
            pair_loss = F.smooth_l1_loss(model.distance(q(rows[:, 0]), q(rows[:, 1])),
                                        torch.as_tensor(rows[:, 2], device=device, dtype=torch.float32))
            sampled = rs.draw(options["decision_batch_size"])
            children = data["children_train"][sampled]
            valid = torch.as_tensor(children >= 0, device=device)
            scores = model.distance(q(np.maximum(children, 0)), q(tasks[sampled, 1])[:, None])
            truth = torch.as_tensor(data["child_distances_train"][sampled], device=device)
            rank_loss = soft_rank_loss(scores, truth, valid, options["rank_temperature"])
            loss = pair_loss + options["rank_weight"] * rank_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step()
            if step == 1 or step % options["eval_every"] == 0 or step == options["steps"]:
                model.eval()
                pm = pair_metrics(model, env, data["pairs_validation"], device)
                rank = ranking_loss(model, env, vt, data["child_distances_validation"][idx],
                                     device, options["rank_temperature"])
                score = pm["huber"] + options["rank_weight"] * rank
                eligible = min(ps.passes, rs.passes) >= 1
                if eligible and score < best:
                    best, best_step = score, step
                    save(step, out / "best.pt")
                record = {"step": step, "seconds": round(time.monotonic() - started, 2),
                          "distance_loss": float(pair_loss.detach()), "rank_loss": float(rank_loss.detach()),
                          "validation_mae": pm["mae"], "validation_objective": score,
                          "validation_shortest_action_rate": decisions(model, env, vt, device)["q"]["shortest_action_rate"],
                          "pair_passes": ps.passes, "decision_passes": rs.passes,
                          "selection_eligible": eligible, "best_step": best_step}
                log.write(json.dumps(record) + "\n"); log.flush()
                print(json.dumps(record), flush=True)
    save(step, out / "last.pt")
    saved = torch.load(out / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(saved["model"])
    summary = {"task": env.name, "source_commit": source_commit, "best_step": best_step,
               "last_step": step, "training_seconds": round(time.monotonic() - started, 2),
               "selection": "heldout intermediate goal distance Huber + full soft rank cross entropy after one coverage pass",
               "torch_version": torch.__version__, "numpy_version": np.__version__,
               "final": evaluate(model, env, data, config, device)}
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if config["task"] != "puzzle8":
        raise ValueError("Puzzle8 trainer requires its own task contract")
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    env = Puzzle8()
    data, summary = build(env, config, args.out)
    print(json.dumps({"data": summary}), flush=True)
    train(env, config, data, args.out, args.device, args.source_commit)


if __name__ == "__main__":
    main()
