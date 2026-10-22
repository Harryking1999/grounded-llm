"""Pair the existing held-out clearing roots with reachable nonempty goals."""
import argparse
import json
from collections import Counter
from pathlib import Path
import numpy as np
from grounded_llm.core.files import atomic_json
from grounded_llm.tasks.blocks import rules
from grounded_llm.tasks.blocks.protocol import action_from_key
from grounded_llm.tasks.blocks.oracle import DistanceOracle


def prepare(records, seed, fractions):
    rng = np.random.default_rng(seed)
    assignment = rng.permutation(len(records))
    rows = []
    oracle = DistanceOracle(cache_limit=2000000, seconds=86400)
    for order, index in enumerate(assignment):
        record = records[int(index)]
        actions = record['reference_actions']
        fraction = fractions[order % len(fractions)]
        cut = max(1, min(len(actions) - 1, round(len(actions) * fraction)))
        state = int(record['start'])
        for key in actions[:cut]:
            state = rules.apply(state, action_from_key(key))
        assert state and state != int(record['start'])
        distance = oracle.distance(int(record['start']), state)
        assert distance == cut
        rows.append(dict(record, trajectory_id=record['trajectory_id'] + '_nonempty',
                         goal=str(state), reference_actions=actions[:cut],
                         shortest_moves=distance, goal_fraction=fraction))
    return sorted(rows, key=lambda row: row['trajectory_id'])


def main():
    parser = argparse.ArgumentParser(__doc__)
    for name in ('config', 'source', 'out'):
        parser.add_argument('--' + name, required=True, type=Path)
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    if args.out.exists():
        raise ValueError('Use a fresh OOD data directory')
    source = json.loads((args.source / 'evaluation.json').read_text())
    rows = prepare(source, cfg['seed'], cfg['goal_fractions'])
    assert len(rows) == cfg['tasks']
    args.out.mkdir(parents=True)
    atomic_json(args.out / 'evaluation.json', rows)
    atomic_json(args.out / 'probe_indices.json', list(range(min(100, len(rows)))))
    atomic_json(args.out / 'acceptance.json', dict(config=cfg, tasks=len(rows),
        source=str(args.source), initial_states='same fixed clearing evaluation cohort',
        goal_selection='certified shortest reference prefix, without Q success filtering',
        goal_fractions=dict(Counter(r['goal_fraction'] for r in rows)),
        shortest_moves=dict(Counter(r['shortest_moves'] for r in rows)),
        nonempty_goals=sum(int(r['goal']) != 0 for r in rows),
        training_goals='empty only', limitation='Qmap may have seen nonempty relations; this is SFT goal OOD'))


if __name__ == '__main__':
    main()
