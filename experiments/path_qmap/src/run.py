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
from grounded_llm.tasks.path.evaluate import distance_table, final_metrics, relation_metrics, rank_objective, candidate_scores


def training_hard_indices(predicted, data):
    """Mine only complete training stars; protected candidates are never labels."""
    tasks = data['train_tasks']; children = data['children'][tasks[:,0]]
    valid = children >= 0; safe = np.maximum(children,0)
    if ((data['membership'][tasks[:,0],tasks[:,1]] != 0).any() or
            ((data['membership'][safe,tasks[:,1,None]] != 0)&valid).any()):
        raise ValueError('Hard replay contains a protected complete decision')
    scores,_ = candidate_scores(predicted,data['children'],tasks)
    selected = children[np.arange(len(tasks)),scores.argmin(-1)]
    return np.flatnonzero(data['distances'][selected,tasks[:,1]] !=
                         data['distances'][tasks[:,0],tasks[:,1]]-1)


def train(config, out, source_commit, device="cpu", checkpoint=None, prepared_run=None):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    (out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    if prepared_run is None:
        env = environment_from_suite(json.loads(Path(config["graph_suite"]).read_text()))
        data = prepare(env.adjacency, config); actions = env.actions
    else:
        with np.load(Path(prepared_run)/'data.npz') as stored:
            data = {key:stored[key] for key in stored.files if key != 'actions'}
            actions = stored['actions']
    adjacency = data['adjacency']
    np.savez_compressed(out / "data.npz", **data, actions=actions)
    options, spec = config["training"], config["model"]
    torch.set_num_threads(options["cpu_threads"])
    torch.manual_seed(options["seed"])
    size = len(adjacency)
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
    if checkpoint is not None:
        parent = torch.load(checkpoint,map_location=device,weights_only=False)
        if (parent['model_type'] != 'state_q_path' or parent['config']['model'] != spec or
                parent['cap'] != cap or not np.array_equal(parent['adjacency'].cpu().numpy(),adjacency)):
            raise ValueError('Continuation must use the matching pure Q and graph')
        model.load_state_dict(parent['model']); optimizer.load_state_dict(parent['optimizer'])
        for group in optimizer.param_groups: group['lr'] = options['learning_rate']
    initial_prediction = distance_table(model, features)
    # A new stage fixes selection before inspecting the old test cohort again.
    initial = final_metrics(initial_prediction, data, config) if checkpoint is None else None
    best, best_step = float("inf"), 0
    started = time.monotonic()
    replay_fraction = options.get('hard_replay_fraction',0)
    if not 0 <= replay_fraction <= 1: raise ValueError('Invalid replay fraction')
    replay_rng = torch.Generator(device=device).manual_seed(options['seed']+1)
    hard_ids = np.empty(0,dtype=np.int64); mining = []

    def save(step, filename):
        atomic_save({"model_type": "state_q_path", "model": model.state_dict(),
                     "optimizer": optimizer.state_dict(), "step": step, "cap": cap,
                     "config": config, "adjacency": torch.from_numpy(adjacency.copy()),
                     "source_commit": source_commit, "initial_checkpoint":None if checkpoint is None else str(checkpoint),
                     "prepared_run":None if prepared_run is None else str(prepared_run),
                     "torch_rng":torch.get_rng_state(),
                     "cuda_rng":torch.cuda.get_rng_state_all() if str(device).startswith('cuda') else [],
                     "replay_rng":replay_rng.get_state(), "hard_indices":torch.from_numpy(hard_ids.copy())}, out / filename)

    if checkpoint is not None:
        val = relation_metrics(initial_prediction,data['validation_pairs'])
        best = val['huber'] + options['listwise_weight'] * rank_objective(
            initial_prediction,data,data['validation_tasks'],options['temperature'])
        save(0,'best.pt')

    with (out / "progress.jsonl").open("w") as log:
        for step in range(1, options["steps"] + 1):
            if replay_fraction and (step == 1 or (step-1)%options['hard_refresh_every'] == 0):
                hard_ids = training_hard_indices(distance_table(model,features),data)
                mining.append(dict(step=step,training_hard_decisions=len(hard_ids)))
            model.train()
            q = model.encode(features)
            rows = pairs[draw(groups, options["batch_size"], device)]
            pointwise = F.smooth_l1_loss(model.distance(q[rows[:, 0]], q[rows[:, 1]]), rows[:, 2].float())
            batch = tasks[torch.randint(len(tasks), (options["listwise_batch_size"],), device=device)].clone()
            replay_count = round(replay_fraction*len(batch)) if len(hard_ids) else 0
            if replay_count:
                ids = torch.randint(len(hard_ids),(replay_count,),generator=replay_rng,device=device)
                batch[:replay_count] = tasks[torch.as_tensor(hard_ids,device=device)[ids]]
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
                if replay_fraction: record.update(hard_pool=len(hard_ids),replayed_decisions=replay_count)
                log.write(json.dumps(record) + "\n")
                log.flush()
                print(json.dumps(record), flush=True)
    save(step, "last.pt")
    model.load_state_dict(torch.load(out / "best.pt", map_location=device, weights_only=True)["model"])
    predicted = distance_table(model, features)
    result = {"source_commit": source_commit, "best_step": best_step, "last_step": step,
              "training_seconds": round(time.monotonic() - started, 2),
              "state_count": size, "directed_edges": int(adjacency.sum()), "diameter": cap,
              "train_pairs": len(pairs), "train_decisions": len(tasks),
              "initial": initial if initial is not None else final_metrics(initial_prediction,data,config),
              "final": final_metrics(predicted, data, config), "training_only_mining":mining,
              "stage_sampling": "New fixed seeded stream; inherited model/Adam only, not exact historical resume" if checkpoint is not None else None,
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
    parser.add_argument('--checkpoint',type=Path)
    parser.add_argument('--prepared-run',type=Path)
    parser.add_argument('--arm',choices=['control','replay'])
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if args.arm:
        config['training'].update(config['arms'][args.arm]); config['arm'] = args.arm
    train(config, args.out, args.source_commit, args.device,
          args.checkpoint,args.prepared_run)


if __name__ == "__main__":
    main()
