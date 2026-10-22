"""Pack the three task-owned exact graphs for this fixed-goal pilot only."""
import importlib
import json
import gzip
from pathlib import Path
import numpy as np


def prepare(config, out):
    out = Path(out)
    build = importlib.import_module(f'grounded_llm.tasks.{config["task"]}.data').build
    groups = {k: [] for k in ('features', 'children', 'costs', 'distances', 'terminal', 'baseline', 'goal_ids', 'split')}
    goals, scenes, starts, offset = [], [], [], 0
    rng = np.random.default_rng(config['seed'] + 1)
    with gzip.open(out / 'scenes.jsonl.gz', 'wt', encoding='utf-8') as stream:
        for sid, (split, graph) in enumerate(build(config, out / 'source_data')):
            n = len(graph['features'])
            for name in ('features', 'costs', 'distances', 'terminal', 'baseline'):
                groups[name].append(graph[name])
            groups['children'].append(np.where(graph['children'] >= 0, graph['children'] + offset, -1))
            groups['goal_ids'].append(np.full(n, sid, dtype=np.int32))
            groups['split'].append(np.full(n, ('train', 'validation', 'test').index(split), dtype=np.int8))
            goals.append(graph['goal_features'])
            candidates = graph['starts']
            chosen = rng.choice(candidates, min(len(candidates), config['evaluation']['starts_per_scene']), replace=False)
            starts.extend((chosen + offset).tolist())
            record = {'scene': sid, 'split': split, 'offset': offset, 'states': n,
                      'reachable': int(np.isfinite(graph['distances']).sum()),
                      'max_finite_distance': float(graph['distances'][np.isfinite(graph['distances'])].max())
                      if np.isfinite(graph['distances']).any() else None,
                      'metadata': graph['metadata']}
            stream.write(json.dumps(record) + '\n')
            scenes.append({k: v for k, v in record.items() if k != 'metadata'})
            offset += n
            if (sid + 1) % 20 == 0: print(json.dumps({'prepared_scenes': sid + 1, 'states': offset}), flush=True)
    data = {key: np.concatenate(value) for key, value in groups.items()}
    data['goals'] = np.asarray(goals, dtype=np.float32)
    data['starts'] = np.asarray(starts, dtype=np.int32)
    finite = np.isfinite(data['distances'])
    cap = config['training']['unreachable_target']
    if data['distances'][finite].max() >= cap:
        raise ValueError('Unreachable proxy must exceed every finite cost in this pilot')
    data['labels'] = np.where(finite, data['distances'], cap).astype(np.float32)
    # Feature overlap is measured separately from puzzle/layout holdout.
    seen = {row.tobytes() for row in data['features'][data['split'] == 0]}
    data['train_feature_seen'] = np.asarray([row.tobytes() in seen for row in data['features']])
    np.savez_compressed(out / 'data.npz', **data)
    summary = {'scenes': scenes, 'states': len(finite), 'reachable': int(finite.sum()),
               'unreachable': int((~finite).sum()), 'feature_dim': data['features'].shape[1],
               'finite_max': float(data['distances'][finite].max()),
               'test_feature_overlap_count': int(data['train_feature_seen'][data['split'] == 2].sum()),
               'test_states': int((data['split'] == 2).sum())}
    (out / 'data_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    return data
