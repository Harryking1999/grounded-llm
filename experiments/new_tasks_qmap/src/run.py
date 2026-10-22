"""Fixed-goal Qmap pilot. Each invocation builds and trains exactly one task."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from torch.nn import functional as F
from grounded_llm.core.files import atomic_json, atomic_save
from grounded_llm.core.qmap.model import StateEncoder, soft_rank_loss
from grounded_llm.core.qmap.sampling import StratifiedEpochSampler
from .data import prepare
from .evaluate import predictions, evaluate


def train(config, out, source_commit, device):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    atomic_json(out / 'config.json', config)
    data = prepare(config, out)
    torch.set_num_threads(config['training']['cpu_threads'])
    torch.manual_seed(config['seed'])
    rng = np.random.default_rng(config['seed'] + 2)
    options = config['training']
    model = StateEncoder(input_dim=data['features'].shape[1], **config['model']).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=options['learning_rate'])
    x = torch.as_tensor(data['features'], device=device)
    goals = torch.as_tensor(data['goals'], device=device)
    goal_ids = torch.as_tensor(data['goal_ids'], dtype=torch.long, device=device)
    children = torch.as_tensor(data['children'], dtype=torch.long, device=device)
    costs = torch.as_tensor(data['costs'], device=device)
    labels = torch.as_tensor(data['labels'], device=device)
    rows = np.flatnonzero(data['split'] == 0)
    decision_rows = np.flatnonzero((data['split'] == 0) & np.isfinite(data['distances']) & ~data['terminal'])
    point_sampler = StratifiedEpochSampler(data['labels'][rows], rng)
    rank_sampler = StratifiedEpochSampler(data['labels'][decision_rows], rng)
    initial = predictions(model, data, device)
    atomic_json(out / 'initial.json', {name: evaluate(data, initial, config, i)
                                     for i, name in enumerate(('train', 'validation', 'test'))})
    best, started = float('inf'), time.monotonic()
    point_exposure, rank_exposure = np.zeros(len(x), dtype=np.int64), np.zeros(len(x), dtype=np.int64)
    for step in range(1, options['steps'] + 1):
        p = rows[point_sampler.draw(options['batch_size'])]
        r = decision_rows[rank_sampler.draw(options['decision_batch_size'])]
        np.add.at(point_exposure, p, 1); np.add.at(rank_exposure, r, 1)
        p = torch.as_tensor(p, device=device); r = torch.as_tensor(r, device=device)
        scores = model.distance(model.encode(x[p]), model.encode(goals[goal_ids[p]]))
        pointwise = F.smooth_l1_loss(scores, labels[p])
        candidates = children[r]
        valid, safe = candidates >= 0, candidates.clamp_min(0)
        candidate_q = model.encode(x[safe])
        goal_q = model.encode(goals[goal_ids[r]])
        predicted = costs[r] + model.distance(candidate_q, goal_q[:, None])
        truth = costs[r] + labels[safe]
        ranking = soft_rank_loss(predicted, truth, valid, options['rank_temperature'])
        loss = pointwise + options['rank_weight'] * ranking
        optimizer.zero_grad(set_to_none=True)
        loss.backward(); optimizer.step()
        if step == 1 or step % options['eval_every'] == 0 or step == options['steps']:
            values = predictions(model, data, device)
            validation = evaluate(data, values, config, 1)
            # Select only by validation policy quality; test cannot choose the checkpoint.
            objective = (1 - validation['q']['success_rate']) + (1 - validation['shortest_action_rate'])
            record = {'step': step, 'seconds': round(time.monotonic() - started, 2),
                      'distance_loss': float(pointwise.detach()), 'rank_loss': float(ranking.detach()),
                      'validation_success': validation['q']['success_rate'],
                      'validation_shortest_action': validation['shortest_action_rate']}
            with (out / 'progress.jsonl').open('a') as log: log.write(json.dumps(record) + '\n')
            print(json.dumps(record), flush=True)
            payload = {'model_type': f'{config["task"]}_fixed_goal_pilot', 'source_commit': source_commit,
                       'config': config, 'step': step, 'model': model.state_dict(),
                       'optimizer': optimizer.state_dict(), 'rng': rng.bit_generator.state,
                       'torch_rng': torch.get_rng_state(),
                       'cuda_rng': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                       'point_sampler': point_sampler.__dict__, 'rank_sampler': rank_sampler.__dict__,
                       'point_exposure': point_exposure, 'rank_exposure': rank_exposure,
                       'input_dim': x.shape[1]}
            atomic_save(payload, out / 'last.pt')
            if objective < best:
                best = objective
                atomic_save(payload, out / 'best.pt')
    saved = torch.load(out / 'best.pt', map_location=device, weights_only=False)
    model.load_state_dict(saved['model'])
    values = predictions(model, data, device)
    result = {'task': config['task'], 'source_commit': source_commit, 'best_step': saved['step'],
              'last_step': step, 'seconds': round(time.monotonic() - started, 2),
              'point_rows_exposed': int((point_exposure[rows] > 0).sum()), 'point_rows_available': len(rows),
              'decision_rows_exposed': int((rank_exposure[decision_rows] > 0).sum()),
              'decision_rows_available': len(decision_rows),
              'metrics': {name: evaluate(data, values, config, i)
                          for i, name in enumerate(('train', 'validation', 'test'))}}
    np.savez_compressed(out / 'predictions.npz', predicted=values, point_exposure=point_exposure, rank_exposure=rank_exposure)
    atomic_json(out / 'summary.json', result)
    print(json.dumps({'complete': str(out), 'test': result['metrics']['test']['q']}), flush=True)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--source-commit', required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    config=json.loads(args.config.read_text())
    if config.get('schema')=='exact_state_pairs':
        from .relations import train as train_pairs
        train_pairs(config,args.out,args.source_commit,args.device,args.resume,args.prepare_only)
    else:
        if args.resume or args.prepare_only: raise ValueError('Legacy fixed-goal pilot does not support these options')
        train(config, args.out, args.source_commit, args.device)


if __name__ == '__main__': main()
