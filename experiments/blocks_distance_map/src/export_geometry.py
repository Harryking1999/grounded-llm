"""Export a frozen Q encoder on every descendant of one report case; no training."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from experiments.gcml_counterexamples.src import blocks
from .model import BoardEncoder, board_bits, distance
from .oracle import successors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--case-summary', type=Path, required=True)
    parser.add_argument('--training-data', type=Path, required=True)
    parser.add_argument('--case-key', default='case_sparse_vs_landmark')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--max-states', type=int, default=4096)
    parser.add_argument('--seed', type=int, default=20260927)
    args = parser.parse_args()
    torch.set_num_threads(4)
    case = json.loads(args.case_summary.read_text(encoding='utf-8'))[args.case_key]
    start = blocks.from_grid(case['board_grid_full'])
    seen, pending, transitions = {start}, [start], []
    while pending:
        source = pending.pop()
        for action, target in successors(source):
            transitions.append((source, action, target))
            if target not in seen:
                seen.add(target)
                if len(seen) > args.max_states:
                    raise ValueError('Case exceeds the complete-graph budget; no partial graph exported')
                pending.append(target)
    masks = sorted(seen, key=lambda s: (-s.bit_count(), s))
    ids = {mask: i for i, mask in enumerate(masks)}
    saved = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    if saved.get('model_type') != 'board_mlp':
        raise ValueError('This visualization requires the board encoder for unseen descendants')
    model = BoardEncoder(saved['config']['model']['state_dim'],
                         saved['config']['model']['encoder_hidden_dim'],
                         saved['cap'], saved['metric'])
    model.load_state_dict(saved['model'])
    model.eval()
    with torch.no_grad():
        q = model.encode(board_bits(masks))
        goal_scores = distance(q, q[ids[0]], saved['metric']).numpy()
    solve = blocks.solver()
    exact = [solve(mask)[0] for mask in masks]
    training_pool = {int(s): i for i, s in enumerate(np.load(args.training_data)['states'])}
    start_state_id = training_pool[start]

    def greedy(first_action=None):
        current, path, actions = start, [ids[start]], []
        rng = np.random.default_rng(args.seed + start_state_id)
        while current:
            legal = successors(current)
            if not legal:
                break
            if first_action is not None and not actions:
                chosen = next(i for i, (a, _) in enumerate(legal) if a == first_action)
            else:
                scores = np.array([goal_scores[ids[t]] for _, t in legal])
                tied = np.flatnonzero(np.isclose(scores, scores.min(), rtol=0, atol=1e-7))
                chosen = int(rng.choice(tied))
            action, current = legal[chosen]
            actions.append(action)
            path.append(ids[current])
        return {'nodes': path, 'actions': actions, 'solved': current == 0}

    metadata = {
        'scope': f'all descendants of the existing {start.bit_count()}-cell report case; not the entire initial board',
        'case_key': args.case_key,
        'model_run': args.checkpoint.parent.name,
        'model_source_commit': saved['source_commit'],
        'metric': saved['metric'], 'dimension': q.shape[1],
        'start': ids[start], 'goal': ids[0], 'start_state_id': start_state_id,
        'crop_origin': case['crop_origin'], 'crop_shape': [len(case['board_grid']), len(case['board_grid'][0])],
        'nodes': [{'id': i, 'mask': str(mask), 'cells': mask.bit_count(),
                   'exact_remaining': -1 if exact[i] is None else exact[i],
                   'q_goal_distance': round(float(goal_scores[i]), 6),
                   'in_training_state_pool': mask in training_pool}
                  for i, mask in enumerate(masks)],
        'edges': sorted([[ids[s], ids[t], a] for s, a, t in transitions]),
        'greedy': greedy(),
        'forced_bad_first_then_greedy': greedy(case['bad_action']['action_id']),
        'same_area_good_node': ids[blocks.apply(start, blocks.PLACEMENTS[case['good_action']['action_id']][1])],
        'seed': args.seed,
    }
    args.out.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(args.out / 'q_vectors.npz', q=q.numpy())
    (args.out / 'graph.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'states': len(masks), 'edges': len(transitions),
                      'solvable': sum(x is not None for x in exact),
                      'greedy': metadata['greedy'],
                      'forced_bad_first_then_greedy': metadata['forced_bad_first_then_greedy']}))


if __name__ == '__main__':
    main()
