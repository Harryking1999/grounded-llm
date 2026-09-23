"""Step 2 asset loading and summaries; stable task contracts live in grounded_llm."""
import json
from pathlib import Path

import numpy as np

from .core import action_catalog, latent_distances, shortest_distances, spearman
from grounded_llm.graph_data import examples, make_split
from grounded_llm.graph_prompts import adjacency_text, messages
from grounded_llm.graph_scoring import fraction, judge, parse_answer


def load_assets(config, asset_dir, root):
    spec = config['assets']
    with np.load(Path(asset_dir) / spec['map_file'], allow_pickle=False) as data:
        q, v = data['q'].copy(), data['v'].copy()
    with np.load(Path(asset_dir) / spec['inputs_file'], allow_pickle=False) as data:
        adj = data['adjacency'].astype(bool)
        actions, distances = data['actions'].copy(), data['graph_distances'].copy()
    assert q.shape == (spec['node_count'], spec['state_dim'])
    assert np.isfinite(q).all() and np.isfinite(v).all()
    expected = np.zeros_like(adj)
    reference = json.loads((Path(root) / spec['graph_reference']).read_text(encoding='utf-8'))
    for node, neighbors in reference['neighbors'].items():
        expected[int(node), neighbors] = True
    np.testing.assert_array_equal(adj, expected)
    np.testing.assert_array_equal(actions, action_catalog(adj)[0])
    np.testing.assert_array_equal(distances, shortest_distances(adj))
    assert v.shape == (len(actions), q.shape[1])
    rms = float(np.sqrt(np.mean(q.astype(np.float64) ** 2)))
    assert np.isfinite(rms) and rms > 0
    return q, adj, distances, rms


def summarize(rows):
    tasks = {task: [r for r in rows if r['task'] == task]
             for task in ('report_current', 'report_goal', 'action')}
    reports = {(r['u'], r['g'], r['task']): r['correct'] for r in rows if r['task'] != 'action'}
    pairs = sorted({(u, g) for u, g, _ in reports})
    both = {(u, g): reports.get((u, g, 'report_current'), False)
            and reports.get((u, g, 'report_goal'), False) for u, g in pairs}
    action = tasks['action']
    return {
        'current_report_exact_accuracy': fraction(r['correct'] for r in tasks['report_current']),
        'goal_report_exact_accuracy': fraction(r['correct'] for r in tasks['report_goal']),
        'both_reports_correct_rate': fraction(both.values()),
        'one_step_success_rate': fraction(r['correct'] for r in action),
        'action_legal_rate': fraction(r['action_legal'] for r in action),
        'format_valid_rate': {t: fraction(r['format_valid'] for r in rs) for t, rs in tasks.items()},
        'node_id_valid_rate': {t: fraction(r['node_id_valid'] for r in rs) for t, rs in tasks.items()},
        'action_success_when_both_reports_correct': fraction(
            r['correct'] for r in action if both.get((r['u'], r['g']), False)),
        'success_by_graph_distance': {str(d): fraction(r['correct'] for r in action if r['graph_distance'] == d)
                                      for d in sorted({r['graph_distance'] for r in action})},
    }


def template_consistency(rows):
    keyed = {(r['template'], r['u'], r['g'], r['task']): r for r in rows}
    result = {}
    for task in ('report_current', 'report_goal', 'action'):
        canonical = [r for r in rows if r['template'] == 'canonical' and r['task'] == task]
        result[task] = fraction(r['correct'] and keyed['heldout', r['u'], r['g'], task]['correct']
                                for r in canonical)
    return result


def swap_diagnostics(rows, adj, distances):
    """Both directions are tested, so the reverse call is the exact slot-swap call."""
    canonical = [r for r in rows if r['template'] == 'canonical']
    keyed = {(r['u'], r['g'], r['task']): r for r in canonical}
    records = []
    for row in canonical:
        reverse = keyed[row['g'], row['u'], row['task']]
        old_score = judge(row, reverse['raw_output'], adj, distances)
        records.append({k: row[k] for k in ('u', 'g', 'task')} | {
            'original_raw_output': row['raw_output'], 'swapped_raw_output': reverse['raw_output'],
            'raw_output_changed': row['raw_output'] != reverse['raw_output'],
            'parsed_output_changed': row['parsed'] != reverse['parsed'],
            'correct_against_original_labels': old_score['correct'],
            'correct_against_swapped_labels': reverse['correct'],
        })
    summary = {t: {name: fraction(r[name] for r in records if r['task'] == t)
                   for name in ('raw_output_changed', 'parsed_output_changed',
                                'correct_against_original_labels', 'correct_against_swapped_labels')}
               for t in ('report_current', 'report_goal', 'action')}
    return {'method': 'reuse_exact_reverse_pair_call', 'summary': summary, 'records': records}


def geometry_stat(vectors, distances):
    i, j = np.triu_indices(len(vectors), 1)
    return {'pairs': len(i), 'spearman': spearman(distances[i, j], latent_distances(vectors)[i, j])}
