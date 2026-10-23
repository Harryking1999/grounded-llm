"""Small Sokoban supervision pilot; reuse the existing exact-pair trainer."""
import json
from pathlib import Path

import numpy as np
import torch

from grounded_llm.core.files import atomic_json
from grounded_llm.core.qmap.exact_graph import UnitGraphOracle
from grounded_llm.core.qmap.physical_data import PhysicalCorpus, PhysicalScene
from grounded_llm.tasks.sokoban.environment import Sokoban
from grounded_llm.tasks.sokoban.relation_data import baseline, decode
from .relation_data import keys, materialize


class IntermediateGoals:
    """Exact auxiliary targets, without changing physical states or features.

    Intermediate targets are additional training information, not inference
    subgoals. Distances are independently computed on the complete directed
    physical graph; a walk length is never assumed to be a shortest distance.
    """
    def __init__(self, corpus):
        self.base = corpus
        self.oracles, self.labels = {}, {}

    def __getattr__(self, name):
        return getattr(self.base, name)

    def distance(self, source, goal, reverse=False):
        source, goal = np.broadcast_arrays(source, goal)
        scenes = self.locate(source)
        if np.any(scenes != self.locate(goal)):
            raise ValueError('Cross-layout auxiliary distance')
        result = np.empty(source.shape, dtype=np.float32)
        for sid in np.unique(scenes):
            offset = self.offsets[sid]; scene = self.scenes[sid]
            for target in np.unique(goal[scenes == sid]):
                mask = (scenes == sid) & (goal == target)
                local = int(target - offset)
                if local in scene.goal_columns:
                    method = scene.reverse if reverse else scene.distance
                    result[mask] = method(source[mask] - offset, goal[mask] - offset)
                else:
                    key = int(sid), local, reverse
                    if key not in self.labels:
                        if sid not in self.oracles:
                            self.oracles[sid] = UnitGraphOracle(scene.children)
                        oracle = self.oracles[sid]
                        self.labels[key] = (oracle.from_goal if reverse else oracle.to_goal)(local)
                    result[mask] = self.labels[key][source[mask] - offset]
        return result


def permitted(corpus, reserved, rows):
    """Keep every candidate; reject the entire context at protected boundaries."""
    rows = np.asarray(rows, dtype=np.int64).reshape(-1, 2)
    children = corpus.successors(rows[:, 0]); valid = children >= 0
    safe = corpus.locate(rows[:, 0]) == corpus.locate(rows[:, 1])
    relation_keys = keys(rows[:, 0], rows[:, 1], corpus.size)
    safe &= ~np.isin(relation_keys, reserved)
    candidate_keys = np.full(children.shape, -1, dtype=np.int64)
    candidate_keys[valid] = keys(children[valid], np.broadcast_to(rows[:, 1, None], children.shape)[valid], corpus.size)
    safe &= ~np.isin(candidate_keys, reserved).any(axis=1)
    scenes = corpus.locate(rows[:, 0])
    for sid in np.unique(scenes):
        scene = corpus.scenes[sid]; mask = scenes == sid
        held = scene.goals[scene.metadata['training_goals']:] + corpus.offsets[sid]
        safe[mask] &= (scene.metadata['split'] == 'train')
        safe[mask] &= ~np.isin(rows[mask], held).any(axis=1) & ~np.isin(children[mask], held).any(axis=1)
    return safe & valid.any(axis=1)


def exact_path(corpus, start, goal, horizon):
    path = [int(start)]
    distance = float(corpus.distance(np.array([start]), np.array([goal]))[0])
    if not 0 < distance <= horizon:
        raise ValueError('Complete path must fit the declared horizon')
    while path[-1] != goal:
        children = corpus.successors(np.array([path[-1]]))[0]
        legal = children >= 0
        labels = np.full(len(children), np.inf)
        labels[legal] = corpus.distance(children[legal], np.full(legal.sum(), goal))
        best = int(labels.argmin())
        if labels[best] != distance - 1:
            raise ValueError('Teacher path is not exactly shortest')
        path.append(int(children[best])); distance -= 1
    return np.asarray(path, dtype=np.int64)


def context_points(contexts, children):
    valid = children >= 0
    return np.concatenate([contexts, np.column_stack([
        children[valid], np.broadcast_to(contexts[:, 1, None], children.shape)[valid]])])


def load_corpus(config, out):
    description = json.loads((out / 'scenes.json').read_text())
    source = Path(description['graphs'])
    scenes = [PhysicalScene(source / row['path'], Sokoban(row['initial_board']), decode, baseline)
              for row in description['scenes']]
    return IntermediateGoals(PhysicalCorpus(scenes))


def prepare(config, out):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    if (out / 'relations.npz').exists():
        return load_corpus(config, out), dict(np.load(out / 'relations.npz')), json.loads((out / 'summary.json').read_text())
    options = config['training']['solution_structure']
    source = (out.parent / config['data']['structure_source']).resolve()
    manifest = json.loads((source / 'graphs/corpus.json').read_text())
    rng = np.random.default_rng(config['data']['structure_seed'])
    selected = []
    for split in ('train', 'validation'):
        pool = [i for i, row in enumerate(manifest['scenes']) if row['split'] == split]
        selected.extend(sorted(rng.choice(pool, config['data']['structure_layouts'][split], replace=False).tolist()))
    records = [manifest['scenes'][i] for i in selected]
    atomic_json(out / 'scenes.json', dict(graphs=str(source / 'graphs'), scenes=records))
    corpus = load_corpus(config, out)
    old_offsets = np.cumsum([0] + [row['states'] for row in manifest['scenes']])
    # Decode the old symmetric holdout keys and remap only the selected layouts.
    with np.load(source / 'relations.npz') as original:
        old_reserved = original['reserved_relations']
    left, right = old_reserved // old_offsets[-1], old_reserved % old_offsets[-1]
    reserved = []
    for sid, old_sid in enumerate(selected):
        low, high = old_offsets[old_sid:old_sid + 2]
        take = (left >= low) & (right < high)
        reserved.append(keys(left[take] - low + corpus.offsets[sid],
                             right[take] - low + corpus.offsets[sid], corpus.size))
    reserved = np.unique(np.concatenate(reserved))
    terminal_groups, auxiliary_groups, train_paths, eval_paths = [], [], [], {}
    rejected = dict(protected_train_paths=0, protected_auxiliary_contexts=0)
    for sid, scene in enumerate(corpus.scenes):
        offset = corpus.offsets[sid]; is_train = scene.metadata['split'] == 'train'
        cohort = 'validation' if is_train else 'unseen_layout_validation'
        eval_paths.setdefault(cohort, [])
        for role in (('training', 'evaluation') if is_train else ('evaluation',)):
            count = config['data']['paths_per_layout'][role]
            # Shared terminal anchors allow independently started paths to merge.
            columns = (list(range(scene.metadata['training_goals'])) if role == 'training'
                       else list(range(scene.metadata['training_goals'], len(scene.goals))))
            # Choose anchors by exact graph coverage before inspecting a model.
            # Multiple anchors avoid requiring 12 distinct starts when a single
            # terminal has only a few reachable physical states.
            columns.sort(key=lambda j:int(((scene.distances[j] > 0) & (scene.distances[j] <= config['evaluation']['horizon'])).sum()), reverse=True)
            bins = config['data']['distance_bins']
            # Interleave distance strata, instead of filling all short paths first.
            layers = [[] for _ in bins]
            for column in columns:
                labels=scene.distances[column];goal=int(scene.goals[column]+offset)
                for layer,lo,hi in zip(layers,[0]+bins[:-1],bins):
                    layer.extend((int(s+offset),goal) for s in rng.permutation(np.flatnonzero((labels>lo)&(labels<=hi))))
            for i in range(len(layers)): layers[i].reverse()
            chosen = 0
            while any(layers) and chosen < count:
                for layer in layers:
                    if not layer or chosen >= count: continue
                    start,goal = layer.pop()
                    path = exact_path(corpus, start, goal, config['evaluation']['horizon'])
                    contexts = np.column_stack([path[:-1], np.full(len(path) - 1, goal)])
                    if role == 'training':
                        if not permitted(corpus, reserved, contexts).all():
                            rejected['protected_train_paths'] += 1; continue
                        terminal_groups.append(contexts); train_paths.append(path)
                        # Multiple exact intermediate anchors; branches are labeled
                        # by the full graph, including detours and irreversible pushes.
                        for anchor in np.unique(np.linspace(1, len(path) - 2, min(3, len(path) - 2), dtype=int)) if len(path) > 2 else []:
                            rows = np.column_stack([path[:anchor], np.full(anchor, path[anchor])])
                            if not np.array_equal(corpus.distance(rows[:,0],rows[:,1]),np.arange(anchor,0,-1)):
                                raise ValueError('Intermediate anchor contradicts exact shortest-path structure')
                            safe = permitted(corpus, reserved, rows)
                            rejected['protected_auxiliary_contexts'] += int((~safe).sum())
                            if safe.all(): auxiliary_groups.append(rows)
                    else:
                        eval_paths[cohort].append(path)
                    chosen += 1
            if chosen == 0: raise ValueError(('No complete protected paths', sid, role))
        print(json.dumps(dict(prepared_scene=sid, split=scene.metadata['split'])), flush=True)
    terminal = np.unique(np.concatenate(terminal_groups), axis=0)
    auxiliary = np.unique(np.concatenate(auxiliary_groups), axis=0)
    all_contexts = np.unique(np.concatenate([terminal, auxiliary]), axis=0)
    # All auxiliary contexts must be disjoint from original terminal-only contexts.
    def codes(rows): return rows[:, 0] * np.int64(corpus.size) + rows[:, 1]
    terminal_codes = codes(terminal); auxiliary = auxiliary[~np.isin(codes(auxiliary), terminal_codes)]
    if not len(auxiliary): raise ValueError('No new intermediate-goal relationships')
    all_contexts = np.unique(np.concatenate([terminal, auxiliary]), axis=0)
    exact = materialize(corpus, all_contexts)
    if not permitted(corpus, reserved, all_contexts).all(): raise ValueError('Protected context in solution bank')
    pair_pool = np.unique(context_points(all_contexts, exact['children']), axis=0)
    data = dict(pairs_train=pair_pool, pair_distances_train=corpus.distance(pair_pool[:, 0], pair_pool[:, 1]),
                reserved_relations=reserved, terminal_ids=np.searchsorted(codes(all_contexts), codes(terminal)),
                auxiliary_ids=np.searchsorted(codes(all_contexts), codes(auxiliary)))
    for key, value in exact.items(): data[key + '_train'] = value
    for name, groups in [('terminal_groups', terminal_groups), ('auxiliary_groups', auxiliary_groups)]:
        array = np.full((len(groups), config['evaluation']['horizon']), -1, dtype=np.int64)
        for i, rows in enumerate(groups):
            ids = np.searchsorted(codes(all_contexts), codes(rows))
            # An intermediate prefix can overlap an existing terminal context.
            array[i, :len(rows)] = ids
        data[name] = array
    for cohort, paths in eval_paths.items():
        rows = np.asarray([[path[0], path[-1]] for path in paths])
        for key, value in materialize(corpus, rows).items(): data[key + '_' + cohort] = value
        array = np.full((len(paths), config['evaluation']['horizon'] + 1), -1, dtype=np.int64)
        for i, path in enumerate(paths): array[i, :len(path)] = path
        data['solution_paths_' + cohort] = array
    # Queries on training layouts also have a distinct terminal goal. They are
    # diagnostics, not evidence of cross-layout generalization.
    for cohort in eval_paths:
        held = data['tasks_' + cohort]
        if np.intersect1d(codes(all_contexts), codes(held)).size:
            raise ValueError('Pilot query leaked into training contexts')
        children = corpus.successors(held[:, 0]); closure = context_points(held, children)
        train_points = keys(pair_pool[:, 0], pair_pool[:, 1], corpus.size)
        if np.intersect1d(train_points, keys(closure[:, 0], closure[:, 1], corpus.size)).size:
            raise ValueError('Pilot query candidate closure leaked into training')
    np.savez_compressed(out / 'relations.npz', **data)
    summary = dict(task='sokoban', states=corpus.size, finite_max=float(np.max(exact['distances'])),
        terminal_contexts=len(terminal), auxiliary_contexts=len(auxiliary), point_pairs=len(pair_pool),
        terminal_paths=len(terminal_groups), intermediate_prefixes=len(auxiliary_groups),
        layouts=config['data']['structure_layouts'], rejected=rejected,
        training_path_lengths={str(n):sum(len(path)-1==n for path in train_paths) for n in sorted({len(path)-1 for path in train_paths})},
        shared_terminal_context_occurrences=sum(len(path)-1 for path in train_paths)-len(terminal),
        evaluation_distance_bins={name:{f'{lo+1}_{hi}':sum(lo<len(path)-1<=hi for path in paths)
            for lo,hi in zip([0]+config['data']['distance_bins'][:-1],config['data']['distance_bins'])} for name,paths in eval_paths.items()},
        all_candidates_preserved=True, original_holdout_closure_excluded=True,
        auxiliary_targets='Intermediate states are additional exact supervised targets; never supplied as inference subgoals.',
        scope='Small development pilot; validation layouts have previously been used for model selection. No final independent test claim.')
    atomic_json(out / 'summary.json', summary)
    return corpus, data, summary


class SolutionSampler:
    """Equal point and ranking budgets; whole shortest paths or flat contexts."""
    def __init__(self, corpus, data, options, rng):
        self.corpus, self.data, self.options, self.rng = corpus, data, options, rng
        self.pair_lookup = {tuple(row): i for i, row in enumerate(data['pairs_train'])}
        terminal_goals=np.unique(data['tasks_train'][data['terminal_ids'],1])
        self.terminal_point_ids=np.flatnonzero(np.isin(data['pairs_train'][:,1],terminal_goals))
        if options['mode'] not in ('flat_terminal', 'path_terminal', 'path_multiscale'):
            raise ValueError('Unknown solution pilot condition')

    def complete_groups(self, groups, count):
        selected = []
        # Every group fits the quota. Fill the leftover with individual contexts;
        # record counts rather than claiming every decision is a complete path.
        for index in self.rng.permutation(len(groups)):
            ids = groups[index]; ids = ids[ids >= 0]
            if len(ids) <= count - len(selected): selected.extend(ids.tolist())
            if count - len(selected) < 1: break
        return np.asarray(selected, dtype=np.int64)

    def draw(self, point_count, rank_count):
        mode = self.options['mode']; data = self.data
        if mode == 'flat_terminal':
            r = self.rng.choice(data['terminal_ids'], rank_count)
        else:
            terminal_quota = rank_count if mode == 'path_terminal' else rank_count // 2
            r = self.complete_groups(data['terminal_groups'], terminal_quota)
            r = np.concatenate([r, self.rng.choice(data['terminal_ids'], terminal_quota - len(r))])
            if mode == 'path_multiscale':
                extra = self.complete_groups(data['auxiliary_groups'], rank_count - terminal_quota)
                extra = np.concatenate([extra, self.rng.choice(data['auxiliary_ids'], rank_count - terminal_quota - len(extra))])
                r = np.concatenate([r, extra])
        p = self.rng.choice(self.terminal_point_ids, point_count)
        if mode != 'flat_terminal':
            contexts = data['tasks_train'][r]
            points = context_points(contexts, data['children_train'][r])
            if len(points) > point_count: raise ValueError('All-branch points exceed fixed point budget')
            p[:len(points)] = [self.pair_lookup[tuple(row)] for row in points]
        return p, r


@torch.no_grad()
def path_metrics(cache, corpus, paths):
    records = []
    for padded in paths:
        path = padded[padded >= 0]; goal = path[-1]
        prediction = cache.scores(path, np.full(len(path), goal))
        drop = prediction[:-1] - prediction[1:]
        children = corpus.successors(path[:-1]); valid = children >= 0
        from .relation_evaluate import safe_scores
        scores = safe_scores(cache, children, np.full(len(children), goal))
        truth = np.full(children.shape, np.inf)
        truth[valid] = corpus.distance(children[valid], np.full(valid.sum(), goal))
        expected = np.arange(len(path) - 2, -1, -1)
        optimal = valid & (truth == expected[:, None])
        good = np.min(np.where(optimal, scores, np.inf), axis=1)
        bad = np.min(np.where(valid & ~optimal, scores, np.inf), axis=1)
        comparable = np.isfinite(bad)
        margin = bad[comparable] - good[comparable]
        later = np.triu(np.ones((len(path), len(path)), dtype=bool), 1)
        ordered = (prediction[:, None] > prediction[None, :] + 1e-6)[later]
        records.append(dict(scene=int(corpus.locate(path[0])), length=len(path) - 1,
            strict_descent=bool((drop > 1e-6).all()), all_span_order=bool(ordered.all()),
            unit_drop_mae=float(np.abs(drop - 1).mean()), edge_descent_rate=float((drop > 1e-6).mean()),
            all_branch_correct=bool(optimal[np.arange(len(scores)), scores.argmin(axis=1)].all()),
            branch_margin_mean=float(margin.mean()) if len(margin) else None,
            branch_positive_margin_rate=float((margin > 1e-6).mean()) if len(margin) else None,
            all_span_order_rate=float(ordered.mean())))
    def aggregate(rows):
        if not rows: return dict(count=0)
        return dict(count=len(rows), **{key:float(np.mean([row[key] for row in rows if row[key] is not None]))
            if any(row[key] is not None for row in rows) else None for key in
            ['strict_descent', 'all_span_order', 'unit_drop_mae', 'edge_descent_rate', 'all_branch_correct',
             'all_span_order_rate', 'branch_margin_mean', 'branch_positive_margin_rate']})
    return dict(all=aggregate(records), long_21_32=aggregate([r for r in records if r['length'] >= 21]),
                by_scene={str(sid):aggregate([r for r in records if r['scene'] == sid]) for sid in sorted({r['scene'] for r in records})},
                meaning='Complete exact teacher paths; all-span order includes nonadjacent pairs. These diagnostics never choose or filter an inference action.')
