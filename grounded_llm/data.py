"""Authority for graph assets, dataset construction, targets and saved splits."""
from collections import deque
from dataclasses import dataclass
from itertools import combinations
import json
from pathlib import Path
import copy
import numpy as np
from .artifacts import ROOT, read_json as read, write_json

@dataclass
class Dataset:
    adjacency: np.ndarray
    distances: np.ndarray
    split: dict
    split_path: Path
    q: np.ndarray | None = None
    rms: float | None = None

def validate_split(split, node_count, counts=None):
    if set(split) != {'train', 'validation', 'test'}:
        raise ValueError('Split must contain train, validation and test')
    seen = set()
    for name, pairs in split.items():
        if counts is not None and len(pairs) != counts[name]:
            raise ValueError(f'Saved {name} split count differs from config')
        for pair in pairs:
            if (len(pair) != 2 or any(type(n) is not int for n in pair)
                    or not 0 <= pair[0] < pair[1] < node_count or tuple(pair) in seen):
                raise ValueError('Invalid, repeated or leaking unordered pair')
            seen.add(tuple(pair))
    if {n for pair in split['train'] for n in pair} != set(range(node_count)):
        raise ValueError('Training split must cover all nodes')

def build_dataset(config, root=ROOT):
    """Load existing graph/map and split; new split creation requires opt-in.

    All relative asset paths resolve against the repository root, irrespective
    of the current shell directory. Existing splits are validated, never redrawn.
    """
    spec = config['data']
    def path(value):
        value = Path(value)
        return value if value.is_absolute() else Path(root) / value
    q, rms = None, None
    if spec.get('asset_dir'):
        q, adj, dist, rms = load_assets(config, path(spec['asset_dir']), root)
    else:
        graph = read(path(spec['graph_path']))['neighbors']
        n = config['assets']['node_count']
        if set(graph) != {str(i) for i in range(n)}:
            raise ValueError('Graph IDs do not match configured node count')
        adj = np.zeros((n, n), dtype=bool)
        for node, neighbors in graph.items():
            if any(type(v) is not int or not 0 <= v < n for v in neighbors):
                raise ValueError('Invalid graph neighbor')
            adj[int(node), neighbors] = True
        action_catalog(adj)
        dist = shortest_distances(adj)
    split_path = path(spec['split_path'])
    if split_path.exists():
        split = read(split_path)
    elif spec.get('create_split', False):
        split = make_split(config)
        split_path.parent.mkdir(parents=True, exist_ok=True)
        write_json(split_path, split)
    else:
        raise FileNotFoundError(f'Reuse an existing split: {split_path}; set data.create_split explicitly to create one')
    validate_split(split, len(adj), spec.get('unordered_pair_counts'))
    return Dataset(adj, dist, split, split_path, q, rms)

def training_examples(config, dataset, condition, split_name='train'):
    """Select report/action/replay pools using the same saved train split."""
    spec = config['training']
    if spec.get('data_spec') and split_name == 'train':
        pools = make_training_data(spec['data_spec'], dataset.split, dataset.adjacency, dataset.distances)
        second = pools[condition['pool']]
        return pools['reports'] + second
    items = examples(dataset.split[split_name], spec['tasks'])
    rng = np.random.default_rng(spec.get('action_label_seed', 0))
    for item in items:
        if item['task'] == 'action':
            valid = np.flatnonzero(dataset.adjacency[item['u']] &
                (dataset.distances[:, item['g']] < dataset.distances[item['u'], item['g']]))
            item['target'] = int(rng.choice(valid))
    return items

def random_graph(n, seed, min_edges=2, max_edges=5):
    rng = np.random.RandomState(seed)
    scores = rng.uniform(size=(n, n))
    scores += np.roll(np.eye(n), 1, axis=1)
    scores -= np.eye(n)
    low = int((min_edges - 1) / 2 + 0.5)
    high = int(max_edges / 2 + 0.5)
    ranks = rng.randint(low, high, n)
    thresholds = np.sort(scores, axis=1)[:, ::-1][np.arange(n), ranks]
    directed = scores > thresholds[:, None]
    return directed | directed.T

def action_catalog(adj):
    if not np.array_equal(adj, adj.T) or np.any(np.diag(adj)):
        raise ValueError("Expected a simple undirected graph")
    edges = [(int(i), int(j)) for i, j in zip(*np.nonzero(np.triu(adj, 1)))]
    actions = np.array([edge for i, j in edges for edge in [(i, j), (j, i)]])
    outgoing = [np.flatnonzero(actions[:, 0] == i) for i in range(len(adj))]
    if any(len(x) == 0 for x in outgoing):
        raise ValueError("Isolated node")
    return actions, outgoing

def shortest_distances(adj):
    n = len(adj)
    distances = np.full((n, n), -1, dtype=np.int32)
    neighbors = [np.flatnonzero(row) for row in adj]
    for start in range(n):
        distances[start, start] = 0
        queue = deque([start])
        while queue:
            node = queue.popleft()
            for other in neighbors[node]:
                if distances[start, other] < 0:
                    distances[start, other] = distances[start, node] + 1
                    queue.append(other)
    if np.any(distances < 0):
        raise ValueError("Disconnected graph")
    return distances

def sample_walks(actions, outgoing, count, steps, seed):
    rng = np.random.default_rng(seed)
    walks = np.empty((count, steps, 3), dtype=np.int32)
    for i in range(count):
        current = int(rng.integers(len(outgoing)))
        for t in range(steps):
            action = int(rng.choice(outgoing[current]))
            successor = int(actions[action, 1])
            walks[i, t] = current, action, successor
            current = successor
    return walks

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

def make_training_data(spec, split, adj, distances):
    action_spec = spec['action_data']
    pool = split[action_spec['source_split']]
    order = np.random.default_rng(action_spec['pair_seed']).permutation(len(pool))
    pair_count = action_spec['unordered_pairs'] if 'unordered_pairs' in action_spec else round(len(adj) * action_spec['unordered_pairs_per_node'])
    pairs = [pool[int(i)] for i in order[:pair_count]]
    actions = examples(pairs, ('action',))
    rng = np.random.default_rng(action_spec['label_seed'])
    for item in actions:
        valid = np.flatnonzero(adj[item['u']] & (distances[:, item['g']] < distances[item['u'], item['g']]))
        item['target'] = int(rng.choice(valid))
    report_spec = spec['report_replay']
    pool = examples(split[report_spec['source_split']])
    order = np.random.default_rng(report_spec['seed']).permutation(len(pool))
    selected, keys = [], set()
    for index in order:
        item = pool[int(index)]
        target = item['u'] if item['task'] == 'report_current' else item['g']
        key = (item['task'], target)
        if key not in keys:
            selected.append(int(index))
            keys.add(key)
    selected_set = set(selected)
    report_count = report_spec['examples'] if 'examples' in report_spec else round(len(adj) * report_spec['examples_per_node'])
    selected += [int(i) for i in order if int(i) not in selected_set][:report_count - len(selected)]
    reports = [pool[i] for i in selected]
    assert len(reports) == len(actions) == 2 * pair_count and len(keys) == len(adj) * 2
    return dict(action_pairs=pairs, actions=actions, reports=reports)

def case_config(spec, case, graph_reference):
    config = copy.deepcopy(read(ROOT/spec['base_training_config']))
    config['study'] = spec['study']
    config['assets'].update(case=case['case'],node_count=case['node_count'],graph_reference=str(graph_reference))
    config['data'].update(sample_subset=True,split_seed=spec['data']['split_seed'],
        unordered_pair_counts=dict(train=case['train_pairs'],validation=spec['data']['validation_pairs'],test=spec['data']['test_pairs']))
    config['outputs']['root'] = 'runs/cml_step2_scaling'
    return config

def graph_data(base):
    graph=read(base/'graph.json')['neighbors']
    adj=np.zeros((len(graph),len(graph)),dtype=bool)
    for a,bs in graph.items():
        adj[int(a),bs]=True
    return adj,shortest_distances(adj)

def make_items(base,spec):
    adj,dist=graph_data(base)
    anchors=examples(read(base/'split.json')['test'],('action',))
    rng=np.random.default_rng(spec['seed'])
    result={t:[] for t in spec['action_tasks']+spec['probe_tasks']}
    for anchor_index,x in enumerate(anchors):
        a,g=x['u'],x['g']
        neighbors=np.flatnonzero(adj[a]).tolist()
        for task in spec['action_tasks']:
            result[task].append(dict(id=f'{task}:{anchor_index}',anchor=anchor_index,task=task,a=a,g=g,
                options=list(map(str,neighbors)),answers=[str(b) for b in neighbors if dist[b,g]<dist[a,g]]))
        candidates=[b for b in range(len(adj)) if b not in (a,g) and dist[b,g]!=dist[a,g]]
        b=int(rng.choice(candidates))
        for swap,(aa,bb) in enumerate(((a,b),(b,a))):
            result['distance_compare'].append(dict(id=f'distance:{anchor_index}:{swap}',anchor=anchor_index,
                task='distance_compare',a=aa,b=bb,g=g,options=['A','B'],answers=['A' if dist[aa,g]<dist[bb,g] else 'B']))
        for b in neighbors:
            answer='更近' if dist[b,g]<dist[a,g] else '更远' if dist[b,g]>dist[a,g] else '相同'
            result['transition_compare'].append(dict(id=f'transition:{anchor_index}:{b}',anchor=anchor_index,
                task='transition_compare',a=a,b=b,g=g,options=['更近','相同','更远'],answers=[answer]))
    perm=rng.permutation(len(adj))
    while np.any(perm==np.arange(len(adj))):
        perm=rng.permutation(len(adj))
    return result,perm.tolist()
