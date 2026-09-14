"""Export compact per-map evidence and grouped tables for Step 1 exploration."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .run import write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    raw = json.loads((args.run_dir / 'summary.json').read_text())
    if not raw['complete']:
        raise ValueError('Exploration is not complete')
    rows = []
    for case in raw['cases']:
        first, last = case['snapshots'][0], case['snapshots'][-1]
        g = last['geometry']
        rows.append({**{k: case[k] for k in ['name', 'case', 'nodes', 'seed',
                    'official_graph', 'state_dim', 'method', 'action_coverage', 'training_seconds']},
                     'transition_mse': g['transition_mse_all_actions'],
                     'successor_accuracy': g['successor_retrieval_accuracy'],
                     'state_centered_rms': g['state_centered_rms'],
                     'state_rms_ratio': g['state_centered_rms'] / first['geometry']['state_centered_rms'],
                     'minimum_pair_distance': g['latent_pair_min'],
                     'spearman_initial': first['geometry']['spearman'],
                     'spearman': g['spearman'], 'cosine_reach': last['planning']['reach_rate']})
    groups = []
    for name in dict.fromkeys(r['name'] for r in rows):
        for official, n in [(True, 32)] + [(False, n) for n in [32, 64, 128, 256, 512]]:
            selected = [r for r in rows if r['name'] == name and r['official_graph'] == official and r['nodes'] == n]
            group = {'name': name, 'official_graph': official, 'nodes': n, 'repeats': len(selected)}
            for field in ['transition_mse', 'successor_accuracy', 'state_centered_rms',
                          'state_rms_ratio', 'minimum_pair_distance', 'spearman', 'cosine_reach']:
                values = [r[field] for r in selected]
                group[field] = float(np.mean(values))
                group[field + '_sd'] = float(np.std(values, ddof=1)) if len(values) > 1 else None
            groups.append(group)
    args.output.mkdir(parents=True, exist_ok=True)
    write_json(args.output / 'exploration_summary.json', {
        **{k: raw[k] for k in ['complete', 'source_revision', 'python', 'numpy', 'device']},
        'cases': rows, 'groups': groups})
    with (args.output / 'exploration_data.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(groups, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
