"""Fixed unordered-pair splits and directed question records."""

from itertools import combinations

import numpy as np


def validate_node_labels(labels, node_count=None):
    """labels[physical_node] is its displayed ID; never reindex Q with these IDs."""
    labels = np.asarray(labels)
    count = len(labels) if node_count is None else node_count
    if labels.shape != (count,) or not np.issubdtype(labels.dtype, np.integer) or not np.array_equal(np.sort(labels), np.arange(count)):
        raise ValueError('Displayed node labels must be a permutation of 0..N-1')
    return labels


def relabel_matrix(matrix, labels):
    """Present adjacency/distances in display order, keeping physical data intact."""
    labels = validate_node_labels(labels, len(matrix))
    inverse = np.argsort(labels)
    return matrix[np.ix_(inverse, inverse)]


def relabel_item(item, labels):
    """Relabel a question for the external judge, not for vector-table lookup."""
    labels = validate_node_labels(labels)
    return {**item, **{key: int(labels[item[key]]) for key in ('u', 'g', 'target') if key in item}}


def make_split(config):
    """Return train/validation/test unordered pairs; both directions stay together.

    ``config`` supplies assets.node_count, data.split_seed and
    data.unordered_pair_counts. The result is JSON-serializable and should be
    saved once as split.json, then loaded unchanged by later stages.
    """
    pairs = list(combinations(range(config['assets']['node_count']), 2))
    order = np.random.default_rng(config['data']['split_seed']).permutation(len(pairs))
    counts = config['data']['unordered_pair_counts']
    if config['data'].get('sample_subset', False):
        assert sum(counts.values()) <= len(pairs)
    else:
        assert sum(counts.values()) == len(pairs)
    result, offset = {}, 0
    for name in ('train', 'validation', 'test'):
        result[name] = [list(pairs[int(i)]) for i in order[offset:offset + counts[name]]]
        offset += counts[name]
    sets = [set(map(tuple, values)) for values in result.values()]
    assert len(set.union(*sets)) == sum(map(len, sets))
    assert {n for pair in result['train'] for n in pair} == set(range(config['assets']['node_count']))
    return result


def examples(pairs, tasks=('report_current', 'report_goal'), template='canonical'):
    """Expand each unordered pair to both directions and requested tasks."""
    return [dict(u=u, g=g, task=task, template=template)
            for a, b in pairs for u, g in ((a, b), (b, a)) for task in tasks]
