"""Streaming exact relations across expanded roots, goals and distance bands."""
import copy
import math
from pathlib import Path

import numpy as np
import torch

from .full_graph_sampling import JointData
from .clear_distance_index import ClearDistanceIndex


class Cardinality:
    """Mergeable HLL; identical packed state pairs deduplicate across roots/ranks."""
    def __init__(self, registers=None, precision=14):
        self.p = precision
        self.registers = (np.zeros(1 << precision, dtype=np.uint8) if registers is None
                          else np.asarray(registers, dtype=np.uint8).copy())

    @staticmethod
    def mix(x):
        with np.errstate(over="ignore"):
            x = (x ^ (x >> np.uint64(30))) * np.uint64(0xbf58476d1ce4e5b9)
            x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94d049bb133111eb)
        return x ^ (x >> np.uint64(31))

    def add(self, rows):
        rows = np.asarray(rows, dtype=np.uint64)
        if not len(rows):
            return
        h = np.full(len(rows), np.uint64(0x9e3779b97f4a7c15))
        for column in rows.T:
            h = self.mix(h ^ self.mix(column))
        index = (h & np.uint64((1 << self.p) - 1)).astype(np.int64)
        tail = h >> np.uint64(self.p)
        # Trailing zero count via isolated low bit; clamp the all-zero tail.
        low = tail & (~tail + np.uint64(1))
        power = np.zeros(len(low), dtype=float)
        np.log2(low, out=power, where=low != 0)
        rank = np.where(low == 0, 65 - self.p, power.astype(int) + 1).astype(np.uint8)
        np.maximum.at(self.registers, index, rank)

    def estimate(self):
        m = len(self.registers)
        value = (.7213 / (1 + 1.079 / m)) * m * m / np.exp2(-self.registers.astype(float)).sum()
        zero = int((self.registers == 0).sum())
        if value <= 2.5 * m and zero:
            value = m * math.log(m / zero)
        return int(round(value))


def distance_histogram(values):
    return {str(int(d)): int(n) for d, n in zip(*np.unique(values, return_counts=True))}


class BroadData(JointData):
    def __init__(self, path, legacy_path, options, rank=0, world=1):
        super().__init__(path, legacy_path, options, rank, world)
        self.phase_sketches = {k: Cardinality() for k in (
            "relations", "huber_relations", "rank_candidate_relations", "forks", "goals",
            "nonzero_huber_score_relations", "nonzero_rank_score_relations",
            "bundle_huber_relations", "bundle_rank_candidate_relations", "bundle_forks",
            "bundle_huber_groups", "bundle_shared_source_groups", "bundle_shared_goal_groups",
            "bundle_cross_branch_groups", "clear_fork_huber_relations", "clear_fork_huber_groups",
            "clear_initial_fork_huber_groups")}
        self.phase_roots = np.zeros(len(self.roots), dtype=bool)
        self.phase_histograms = {}
        self.requested_distances = {}
        self.initial_distances = {}
        self.actual_distances = {}
        self.fork_degrees = {}
        self.stream_calls = 0
        self.clear_requested_distances, self.clear_actual_distances = {}, {}
        self.bundle_histograms = {}
        self.legacy_update_distances = {}
        self.bundle_update_counts = {}
        self._pending_bundle_huber, self._pending_bundle_candidates = [], []
        self._pending_bundle_forks, self._pending_bundle_events = [], {}
        self._bundle_rank_roles = {}
        self._pending_clear_forks = []
        self._pending_clear_initial_forks = []
        self.layer_views, self.layer_suffixes = {}, {}
        index_path = options.get("clear_index_data")
        self.clear_indexes = ({gid: ClearDistanceIndex(Path(index_path) / str(gid)) for gid in self.graphs}
                              if index_path else None)
        self.clear_suffixes = []
        if self.clear_indexes is not None:
            for root in self.roots:
                suffix = [None] * (len(root.parts) + 1)
                suffix[-1] = np.asarray([1], dtype=np.uint64)
                for i in reversed(range(len(root.parts))):
                    suffix[i] = np.convolve(self.clear_indexes[root.record["components"][i]["graph"]].sizes, suffix[i + 1])
                self.clear_suffixes.append(suffix)
        previous = options.get("previous_sampler_root")
        if previous:
            previous_world = int(options.get("previous_sampler_world", world))
            if world % previous_world:
                raise ValueError("Sampler expansion requires an integer multiple of the parent shards")
            parent_rank, child_lane = rank % previous_world, rank // previous_world
            state = torch.load(Path(previous) / f"sampler_{options['previous_sampler_step']:06d}_rank{parent_rank}.pt",
                               map_location="cpu", weights_only=False)
            # The graph builder verifies the previous root/component index prefix.
            old_count = len(state["broad"])
            if old_count > len(self.roots):
                raise ValueError("Cannot shrink the graph domain on continuation")
            expanded = dict(state)
            for key in ("broad", "clear", "broad_rank", "clear_rank", "root_stars_seen"):
                expanded[key] = np.pad(state[key], (0, len(self.roots) - old_count))
            if world != previous_world:
                # A cursor is a GLOBAL stream offset; split its future residue
                # classes without resetting, skipping or replaying old pairs.
                for key in ("broad", "clear", "broad_rank", "clear_rank"):
                    cursor = expanded[key] + parent_rank - rank
                    expanded[key] = ((cursor + world - 1) // world) * world
                if child_lane:
                    generator = np.random.PCG64()
                    generator.state = expanded["rng"]
                    expanded["rng"] = generator.jumped(child_lane).state
            if old_count != len(self.roots):
                expanded["order"] = self.schedule_rng.permutation(len(self.roots))
                expanded["position"] = 0
                expanded["schedule_rng"] = self.schedule_rng.bit_generator.state
            super().load_state_dict(expanded)
            self.prefix_roots = old_count
        else:
            self.prefix_roots = 0

    @staticmethod
    def add_hist(target, values):
        for key, count in distance_histogram(values).items():
            target[key] = target.get(key, 0) + count

    def clear_rows(self, board, count, ranking=False):
        if self.clear_indexes is None:
            return super().clear_rows(board, count, ranking)
        broad = round(count * self.options.get("clear_broad_fraction", .15))
        old_source, old_goal = super().clear_rows(board, broad, ranking)
        count -= broad
        if not count:
            return old_source, old_goal
        root, suffix = self.roots[board], self.clear_suffixes[board]
        available = np.flatnonzero(suffix[0])
        available = available[available > 0]
        cycle = np.asarray(self.options.get("clear_distance_cycle", [2, 3, 4, 5, 6, 7, 8, 6, 7, 8, 9, 10, 11, 12, 13]))
        wanted = cycle[(np.arange(count) + int(self.rng.integers(len(cycle)))) % len(cycle)]
        actual = wanted.copy()
        for distance in np.unique(wanted):
            if distance in available:
                continue
            lo, hi = (14, 18) if distance >= 14 else (9, 13) if distance >= 9 else (6, 8) if distance >= 6 else (1, 5)
            pool = available[(available >= lo) & (available <= hi)]
            if not len(pool):
                pool = available[np.abs(available - distance) == np.abs(available - distance).min()]
            selected = wanted == distance
            actual[selected] = self.rng.choice(pool, int(selected.sum()))
        self.add_hist(self.clear_requested_distances, wanted)
        self.add_hist(self.clear_actual_distances, actual)
        remaining = actual.copy()
        source = np.empty((count, len(root.parts)), dtype=np.int64)
        for i, (graph, _) in enumerate(root.parts):
            index = self.clear_indexes[root.record["components"][i]["graph"]]
            distances = np.arange(len(index.sizes))
            rest = remaining[:, None] - distances
            valid = (rest >= 0) & (rest < len(suffix[i + 1]))
            weights = np.where(valid, index.sizes[None, :] * suffix[i + 1][np.clip(rest, 0, len(suffix[i + 1]) - 1)], 0).astype(float)
            choice = (np.cumsum(weights, -1) <= self.rng.random(count)[:, None] * weights.sum(-1)[:, None]).sum(-1)
            for distance in np.unique(choice):
                rows = choice == distance
                source[rows, i] = graph.rank(index.sample(distance, int(rows.sum()), self.rng))
            remaining -= choice
        assert np.all(remaining == 0)
        goal = np.tile([g.zero_id for g, _ in root.parts], (count, 1))
        initial = np.asarray([g.initial_id for g, _ in root.parts], dtype=np.int64)
        initial_fork = (self.options.get("clear_initial_fork_first", False)
                        and board % self.world == self.rank)
        if ranking and initial_fork:
            # Roots share a schedule across ranks. One designated rank supplies
            # this full initial fork; other ranks retain their downstream forks.
            source[0] = initial
        if not ranking and self.options.get("clear_fork_huber_fraction", 0):
            budget = int(count * self.options["clear_fork_huber_fraction"])
            # Complete forks also enter the existing distance Huber, including
            # unreachable children. Soft-rank alone need not fit their distances.
            anchors, ids = np.unique(source, axis=0, return_index=True)
            deep = np.flatnonzero((actual[ids] >= 10) & (actual[ids] <= 12))
            other = np.flatnonzero(~np.isin(np.arange(len(ids)), deep))
            self.rng.shuffle(deep); self.rng.shuffle(other)
            order = list(deep) + list(other)
            anchors = anchors[order]
            if initial_fork:
                # Unique distance-layer sampling can dilute the actual start
                # among many other deep states. Give each training root its
                # complete first fork while retaining downstream groups below.
                same = (anchors == initial).all(-1)
                anchors = np.concatenate((initial[None], anchors[~same]))
            degree = sum(g.degrees(anchors[:, i]) for i, (g, _) in enumerate(root.parts))
            anchors = anchors[np.cumsum(degree + 1) <= budget]
            children = [anchors]
            for i, (graph, _) in enumerate(root.parts):
                parents, successors = graph.star(anchors[:, i])
                child = anchors[parents].copy()
                child[:, i] = successors
                children.append(child)
            group_rows = np.concatenate(children)
            if len(group_rows):
                source[-len(group_rows):] = group_rows
                initial_rows = anchors[(anchors == initial).all(-1)]
                if len(initial_rows):
                    self._pending_clear_initial_forks.append(np.column_stack((
                        root.packed(initial_rows), np.zeros((len(initial_rows), 2), dtype=np.uint64))))
                self._pending_clear_forks.append((
                    np.column_stack((root.packed(group_rows), np.zeros((len(group_rows), 2), dtype=np.uint64))),
                    np.column_stack((root.packed(anchors), np.zeros((len(anchors), 2), dtype=np.uint64)))))
        # This changes sampling, never the graph or legal action set. The broad
        # stream and retained clear fraction still include unreachable states;
        # stars below retain every legal trap action at each clearable parent.
        return np.concatenate((old_source, source)), np.concatenate((old_goal, goal))

    def _paths(self, root, start, length=18, first_choices=None):
        # Walk directly on component bit masks: no repeated BDD unrank/rank at
        # every move. All legal actions retain nonzero sampling probability.
        current = start.copy()
        result = [current.copy()]
        for move in range(length):
            legal = [(current[:, i, None] & graph.tiles) == graph.tiles
                     for i, (graph, _) in enumerate(root.parts)]
            degrees = np.stack([x.sum(-1) for x in legal], -1)
            total = degrees.sum(-1)
            choice = (self.rng.random(len(start)) * total).astype(np.int64)
            if move == 0 and first_choices is not None:
                choice = np.asarray(first_choices)
            ends = np.cumsum(degrees, axis=-1)
            part = (choice[:, None] >= ends).sum(-1)
            begins = np.column_stack((np.zeros(len(start), dtype=np.int64), ends[:, :-1]))
            for i, (graph, _) in enumerate(root.parts):
                selected = np.flatnonzero((part == i) & (total > 0))
                if not len(selected):
                    continue
                offset = choice[selected] - begins[selected, i]
                tiles = (np.cumsum(legal[i][selected], -1) > offset[:, None]).argmax(-1)
                current[selected, i] ^= graph.tiles[tiles]
            result.append(current.copy())
        return np.stack(result, axis=1)

    def _sources(self, root, count, wanted):
        initial = np.tile([int(g.states[np.asarray([g.initial_id])][0]) for g, _ in root.parts], (count, 1)).astype(np.uint64)
        # Whole-root and several different intermediate sources, plus uniform
        # graph sources; not only optimal clearing paths or fixed failure goals.
        path = self._paths(root, initial)
        first_depth = self.rng.choice([0, 0, 0, 1, 2, 3, 4], count)
        # Spread medium-distance starts across both dense twelve-block roots
        # and smaller residual states; depth and goal distance are separate axes.
        deep = ((wanted >= 6) & (wanted <= 8) &
                (self.rng.random(count) < self.options.get("medium_deep_source_fraction", 0)))
        first_depth[deep] = self.rng.integers(2, 7, size=int(deep.sum()))
        first_depth[wanted >= 9] = self.rng.integers(2, size=int((wanted >= 9).sum()))
        # At least two cells per remaining move and a nonempty goal. Do not
        # choose a residual source too small even to admit its requested band.
        occupied = np.unpackbits(path.view(np.uint8), axis=-1).sum(-1)
        depth_cap = np.maximum((occupied >= (2 * wanted + 2)[:, None]).sum(-1) - 1, 0)
        first_depth = np.minimum(first_depth, depth_cap)
        source = path[np.arange(count), first_depth]
        uniform = (np.arange(count) % 2 == 0) & (wanted <= 5)
        local = root.decode(self.rng.integers(root.size, size=int(uniform.sum())))
        for i, (graph, _) in enumerate(root.parts):
            source[uniform, i] = graph.states[local[:, i]]
        return source

    def _layer_targets(self, board, wanted):
        """Sample existing exact BFS layers directly, without random-walk bias."""
        root = self.roots[board]
        if board not in self.layer_suffixes:
            suffix = [None] * (len(root.parts) + 1)
            suffix[-1] = np.asarray([1], dtype=np.uint64)
            for i in reversed(range(len(root.parts))):
                graph = root.parts[i][0]
                if id(graph) not in self.layer_views:
                    views = []
                    for node in graph.layers:
                        view = copy.copy(graph)
                        view.root = int(node)
                        view.size = int(graph.counts[node]) << int(graph.level[node])
                        views.append(view)
                    self.layer_views[id(graph)] = views
                sizes = np.asarray([v.size for v in self.layer_views[id(graph)]], dtype=np.uint64)
                suffix[i] = np.convolve(sizes, suffix[i + 1])
            self.layer_suffixes[board] = suffix
        suffix = self.layer_suffixes[board]
        counts = suffix[0].copy()
        empty_parts = [int(g.distance((1 << g.width) - 1, 0)) for g, _ in root.parts]
        if min(empty_parts) >= 0:
            counts[sum(empty_parts)] -= 1
        available = np.flatnonzero(counts)
        available = available[available > 0]
        if not len(available):
            raise ValueError("Root has no positive-distance nonempty target")
        wanted = np.asarray(wanted)
        actual = wanted.copy()
        for d in np.unique(wanted):
            if d not in available:
                lo, hi = (14, 18) if d >= 14 else (9, 13) if d >= 9 else (6, 8) if d >= 6 else (1, 5)
                pool = available[(available >= lo) & (available <= hi)]
                if not len(pool):
                    pool = available[np.abs(available - d) == np.abs(available - d).min()]
                actual[wanted == d] = self.rng.choice(pool, int((wanted == d).sum()))
        targets = np.zeros((len(actual), len(root.parts)), dtype=np.uint64)
        pending = np.arange(len(actual))
        while len(pending):
            remaining = actual[pending].copy()
            for i, (graph, _) in enumerate(root.parts):
                views = self.layer_views[id(graph)]
                sizes = np.asarray([v.size for v in views], dtype=np.uint64)
                rest = remaining[:, None] - np.arange(len(sizes))
                valid = (rest >= 0) & (rest < len(suffix[i + 1]))
                weights = np.where(valid, sizes[None, :] * suffix[i + 1][np.clip(rest, 0, len(suffix[i + 1]) - 1)], 0).astype(float)
                choice = (np.cumsum(weights, -1) <= self.rng.random(len(pending))[:, None] * weights.sum(-1)[:, None]).sum(-1)
                for d in np.unique(choice):
                    rows = choice == d
                    targets[pending[rows], i] = views[d].unrank(self.rng.integers(views[d].size, size=int(rows.sum())))
                remaining -= choice
            assert np.all(remaining == 0)
            # Conditional uniform sampling excludes only the empty target here;
            # the separate clearing stream continues to train empty targets.
            pending = pending[~targets[pending].any(-1)]
        return targets, actual

    def _structure_rows(self, board, count, cross):
        fraction = self.options.get("exact_layer_fraction", 0)
        if not 0 <= fraction <= 1:
            raise ValueError("exact_layer_fraction must be in [0, 1]")
        indexed = int(count * fraction)
        plain = self._path_structure_rows(board, count - indexed, cross)
        if not indexed:
            return plain
        root = self.roots[board]
        cycle = np.asarray(self.options["stream_distance_cycle"])
        wanted = cycle[(np.arange(indexed) + int(self.rng.integers(len(cycle)))) % len(cycle)]
        goals, actual = self._layer_targets(board, wanted)
        sources = np.tile([g.initial_id for g, _ in root.parts], (indexed, 1))
        goal_ids = np.stack([g.rank(goals[:, i]) for i, (g, _) in enumerate(root.parts)], -1)
        self.add_hist(self.requested_distances, wanted)
        self.add_hist(self.initial_distances, actual)
        self.add_hist(self.actual_distances, actual)
        return np.concatenate((sources, plain[0])), np.concatenate((goal_ids, plain[1]))

    def _path_structure_rows(self, board, count, cross):
        root = self.roots[board]
        self.stream_calls += 1
        if not count:
            empty = root.decode(np.zeros(0, dtype=np.uint64))
            return empty, empty.copy()
        cycle = np.asarray(self.options.get("stream_distance_cycle", [2, 3, 4, 5, 6, 7, 8, 6, 7, 8, 9, 10, 11, 12, 13]))
        wanted = cycle[(np.arange(count) + int(self.rng.integers(len(cycle)))) % len(cycle)]
        source = self._sources(root, count, wanted)
        # Continue independently after each sampled source, retaining every
        # intermediate goal along each random legal path for exact stratification.
        paths = self._paths(root, source)
        distances = sum(graph.distance(source[:, None, i], paths[:, :, i])
                        for i, (graph, _) in enumerate(root.parts))
        nonempty = (paths != 0).any(-1)
        exact = (distances == wanted[:, None]) & nonempty
        same_band = np.where(wanted[:, None] >= 9, (distances >= 9) & (distances <= 13),
                             np.where(wanted[:, None] >= 6, (distances >= 6) & (distances <= 8),
                                      (distances >= 1) & (distances <= 5))) & nonempty
        choices = np.where(exact.any(-1)[:, None], exact, same_band)
        fallback = (distances > 0) & nonempty
        hardest = np.where(fallback, distances, -1).max(-1)
        fallback &= (wanted < 6)[:, None] | (distances == hardest[:, None])
        choices = np.where(choices.any(-1)[:, None], choices, fallback)
        # If the sampled source is a leaf, equality remains a legitimate broad
        # relation, reported as distance0 instead of claiming a long example.
        choices[~choices.any(-1), 0] = True
        noise = self.rng.random(choices.shape)
        index = np.where(choices, noise, -1).argmax(-1)
        goal = paths[np.arange(count), index]
        original = distances[np.arange(count), index]
        self.add_hist(self.requested_distances, wanted)
        self.add_hist(self.initial_distances, original)
        # Cross-branch Huber relations provide true nonoptimal/unreachable
        # continuations as well as complete ranking forks. Long parents are not
        # repeatedly collapsed into one/two-step descendants.
        if cross:
            chosen = np.arange(count) % 5 == 0
            after = self._paths(root, source[chosen], length=2)
            source[chosen] = after[:, -1]
        actual_parts = [graph.distance(source[:, i], goal[:, i]) for i, (graph, _) in enumerate(root.parts)]
        actual = np.stack(actual_parts, -1)
        actual = np.where((actual < 0).any(-1), -1, actual.sum(-1))
        self.add_hist(self.actual_distances, actual)
        return (np.stack([g.rank(source[:, i]) for i, (g, _) in enumerate(root.parts)], -1),
                np.stack([g.rank(goal[:, i]) for i, (g, _) in enumerate(root.parts)], -1))

    def structure_rows(self, board, count, cross):
        # Twelve related contexts occupy existing structure slots, not extra loss
        # terms. Independent broad/clear/legacy streams remain in the batch.
        fraction = self.options.get("branch_bundle_fraction", 0)
        if not 0 <= fraction <= 1:
            raise ValueError("branch_bundle_fraction must be in [0, 1]")
        bundles = min(int(count * fraction) // 12, max(count - 1, 0) // 12)
        remaining = count - 12 * bundles
        plain = self._structure_rows(board, remaining, cross)
        if not bundles:
            return plain
        root = self.roots[board]
        cycle = np.asarray(self.options.get("stream_distance_cycle", [6, 7, 8, 9, 10, 11, 12, 13]))
        wanted = cycle[(np.arange(bundles) + int(self.rng.integers(len(cycle)))) % len(cycle)]
        masks = self._sources(root, bundles, wanted)
        indexed = int(bundles * self.options.get("exact_layer_fraction", 0))
        if indexed:
            masks[:indexed] = [np.uint64((1 << g.width) - 1) for g, _ in root.parts]
        degree = sum(((masks[:, i, None] & g.tiles) == g.tiles).sum(-1)
                     for i, (g, _) in enumerate(root.parts))
        left_choice = (self.rng.random(bundles) * degree).astype(int)
        # Different immediate actions whenever a real fork exists; independent
        # random walks alone can repeatedly select the same branch.
        offset = 1 + (self.rng.random(bundles) * np.maximum(degree - 1, 0)).astype(int)
        right_choice = (left_choice + offset) % np.maximum(degree, 1)
        third_offset = 1 + (self.rng.random(bundles) * np.maximum(degree - 2, 0)).astype(int)
        third_offset += third_offset >= offset
        third_choice = np.where(degree >= 3, (left_choice + third_offset) % np.maximum(degree, 1), left_choice)
        paths = [self._paths(root, masks, first_choices=choice)
                 for choice in (left_choice, right_choice, third_choice)]
        def target(path, previous):
            distance = sum(g.distance(masks[:, None, i], path[:, :, i])
                           for i, (g, _) in enumerate(root.parts))
            valid = (path != 0).any(-1) & (distance > 0)
            error = np.where(valid, np.abs(distance - wanted[:, None]), 1000)
            # Prefer different goals within the same closest exact distance;
            # retain/report genuine merges when no such alternative exists.
            distinct = valid.copy()
            for earlier in previous:
                distinct &= (path != earlier[:, None]).any(-1)
            distinct &= error == error.min(-1)[:, None]
            error = np.where(distinct.any(-1)[:, None] & ~distinct, 1000, error)
            chosen = (error + self.rng.random(error.shape) * .01).argmin(-1)
            chosen[~valid.any(-1)] = 0
            return path[np.arange(bundles), chosen]
        targets = []
        for path in paths:
            targets.append(target(path, targets))
        starts = np.stack([masks] + [path[:, 1] for path in paths], axis=1)
        targets = np.stack(targets, axis=1)
        if indexed:
            for j in range(3):
                targets[:indexed, j], _ = self._layer_targets(board, wanted[:indexed])
            # Shared reachable goals across distinct branches, while the other
            # two independently sampled goals retain reachable/unreachable cross
            # relations. No candidate is removed from the ranking forks.
            candidates = []
            for part, (graph, _) in enumerate(root.parts):
                for tile in graph.tiles:
                    successor = masks[:indexed].copy()
                    successor[:, part] ^= tile
                    candidates.append(successor)
            candidates = np.stack(candidates, axis=1)
            feasible = np.stack([g.distance(candidates[:, :, i], targets[:indexed, 0, i, None])
                                 for i, (g, _) in enumerate(root.parts)], -1)
            feasible = (feasible >= 0).all(-1)
            unused = np.ones(feasible.shape, dtype=bool)
            for j in range(3):
                choices = feasible & unused
                choices = np.where(choices.any(-1)[:, None], choices, unused)
                choices = np.where(choices.any(-1)[:, None], choices, True)
                selected = np.where(choices, self.rng.random(choices.shape), -1).argmax(-1)
                starts[:indexed, j + 1] = candidates[np.arange(indexed), selected]
                unused[np.arange(indexed), selected] = False
        # Shared source, shared goal, both branch continuations and cross-branch
        # goals. Crossing branches can be finite, unreachable or reconvergent;
        # query the exact graph instead of assuming a tree or assigning a label.
        source = np.repeat(starts, 3, axis=1).reshape(-1, masks.shape[-1])
        goal = np.tile(targets, (1, 4, 1)).reshape(source.shape)
        roles = np.tile([f"source_{s}_goal_{g}" for s in range(4) for g in range(3)], bundles)
        source_ids = np.stack([g.rank(source[:, i]) for i, (g, _) in enumerate(root.parts)], -1)
        goal_ids = np.stack([g.rank(goal[:, i]) for i, (g, _) in enumerate(root.parts)], -1)
        keys = np.column_stack((root.packed(source_ids), root.packed(goal_ids)))
        distances = root.distance_parts(source_ids, goal_ids)
        truth = np.where((distances < 0).any(-1), -1, distances.sum(-1))
        distinct_goals = ((targets[:, 0] != targets[:, 1]).any(-1) &
                          (targets[:, 0] != targets[:, 2]).any(-1) &
                          (targets[:, 1] != targets[:, 2]).any(-1))
        complete = (degree >= 3) & distinct_goals
        if cross:
            self._pending_bundle_huber.append((keys, truth, roles, wanted, complete))
        else:
            self._bundle_rank_roles.update((tuple(key), role) for key, role in zip(keys, roles))
        events = {"groups": bundles, "anchors_with_distinct_actions": int((degree > 1).sum()),
                  "anchors_with_three_distinct_actions": int((degree >= 3).sum()),
                  "three_distinct_goals": int(distinct_goals.sum()),
                  "complete_four_source_three_goal_groups": int(complete.sum()),
                  "goal_pair_merges": sum(int((targets[:, a] == targets[:, b]).all(-1).sum())
                                          for a, b in ((0, 1), (0, 2), (1, 2)))}
        for key, value in events.items():
            self._pending_bundle_events[key] = self._pending_bundle_events.get(key, 0) + value
        # Keep ordinary rows last: JointData may replace its final rank row by
        # the mandatory first full-root fork, leaving these bundles intact.
        return (np.concatenate((source_ids, plain[0])), np.concatenate((goal_ids, plain[1])))

    def batch(self, pair_count, rank_count):
        self._pending_bundle_huber, self._pending_bundle_candidates = [], []
        self._pending_bundle_forks, self._pending_bundle_events = [], {}
        self._bundle_rank_roles = {}
        self._pending_clear_forks = []
        self._pending_clear_initial_forks = []
        return super().batch(pair_count, rank_count)

    def boards(self):
        result = super().boards()
        self.phase_roots[result] = True
        return result

    def stars(self, root, source, goal):
        degree = sum(g.degrees(source[:, i]) for i, (g, _) in enumerate(root.parts))
        active = degree > 0
        self.phase_sketches["forks"].add(np.column_stack((root.packed(source[active]), root.packed(goal[active]))))
        self.add_hist(self.fork_degrees, degree)
        result = super().stars(root, source, goal)
        if result is not None and self._bundle_rank_roles:
            keys = np.column_stack((root.packed(source), root.packed(goal)))
            selected = np.asarray([tuple(key) in self._bundle_rank_roles for key in keys]) & active
            self._pending_bundle_forks.append(keys[selected])
            s, g, truth, groups = result
            use = selected[groups]
            self._pending_bundle_candidates.append((np.column_stack((s[use], g[use])), truth[use]))
        return result

    def record_update(self, arrays, huber_nonzero, rank_nonzero):
        # Called only after successful optimizer.step, so an aborted sampling or
        # forward call contributes no exposure to this phase's committed counts.
        for forks in self._pending_clear_initial_forks:
            self.phase_sketches["clear_initial_fork_huber_groups"].add(forks)
        for pairs, forks in self._pending_clear_forks:
            self.phase_sketches["clear_fork_huber_relations"].add(pairs)
            self.phase_sketches["clear_fork_huber_groups"].add(forks)
            key = "clear_complete_forks_in_huber"
            self.bundle_update_counts[key] = self.bundle_update_counts.get(key, 0) + len(forks)
        legacy_count = round(len(arrays[2]) * self.options["legacy_fraction"])
        self.add_hist(self.legacy_update_distances, arrays[2][:legacy_count])
        def canonical(groups):
            order = np.lexsort(tuple(groups[:, :, i] for i in reversed(range(4))))
            return np.take_along_axis(groups, order[:, :, None], axis=1).reshape(len(groups), -1)
        for keys, truth, roles, wanted, complete in self._pending_bundle_huber:
            self.phase_sketches["bundle_huber_relations"].add(keys)
            self.add_hist(self.bundle_histograms.setdefault("huber_requested_anchor_distance", {}), wanted)
            groups = keys.reshape(-1, 12, 4)[complete]
            if len(groups):
                self.phase_sketches["bundle_huber_groups"].add(canonical(groups))
                self.phase_sketches["bundle_shared_source_groups"].add(canonical(groups.reshape(-1, 3, 4)))
                shared_goal = groups.reshape(-1, 4, 3, 4).transpose(0, 2, 1, 3).reshape(-1, 4, 4)
                self.phase_sketches["bundle_shared_goal_groups"].add(canonical(shared_goal))
                self.phase_sketches["bundle_cross_branch_groups"].add(canonical(groups[:, 3:]))
            for role in np.unique(roles):
                self.add_hist(self.bundle_histograms.setdefault("huber_" + role, {}), truth[roles == role])
        for keys, truth in self._pending_bundle_candidates:
            self.phase_sketches["bundle_rank_candidate_relations"].add(keys)
            self.add_hist(self.bundle_histograms.setdefault("rank_candidates", {}), truth)
        for keys in self._pending_bundle_forks:
            self.phase_sketches["bundle_forks"].add(keys)
        for key, value in self._pending_bundle_events.items():
            self.bundle_update_counts[key] = self.bundle_update_counts.get(key, 0) + value
        for kind, source, goal, truth, mask in (
            ("huber", arrays[0], arrays[1], arrays[2], huber_nonzero),
            ("rank_candidate", arrays[3], arrays[4], arrays[5], rank_nonzero)):
            pairs = np.column_stack((source, goal))
            self.phase_sketches["relations"].add(pairs)
            self.phase_sketches[kind + "_relations"].add(pairs)
            name = "nonzero_huber_score_relations" if kind == "huber" else "nonzero_rank_score_relations"
            self.phase_sketches[name].add(pairs[mask])
            self.phase_sketches["goals"].add(goal)
            for empty, goal_kind in ((True, "clear"), (False, "nonempty")):
                chosen = (goal == 0).all(-1) == empty
                bucket = self.phase_histograms.setdefault(kind + "_" + goal_kind, {})
                self.add_hist(bucket, truth[chosen])

    def sketch_state(self):
        return {k: v.registers for k, v in self.phase_sketches.items()}

    @staticmethod
    def merged_exposure(states):
        return {"estimated_distinct_" + k: Cardinality(np.maximum.reduce([s[k] for s in states])).estimate()
                for k in states[0]}

    def state_dict(self):
        state = super().state_dict()
        state["broad_stream"] = {"sketches": self.sketch_state(), "roots": self.phase_roots,
            "histograms": self.phase_histograms, "requested": self.requested_distances,
            "initial": self.initial_distances, "actual": self.actual_distances, "fork_degrees": self.fork_degrees,
            "clear_requested": self.clear_requested_distances, "clear_actual": self.clear_actual_distances,
            "bundle_histograms": self.bundle_histograms, "bundle_update_counts": self.bundle_update_counts,
            "legacy_update_distances": self.legacy_update_distances,
            "stream_calls": self.stream_calls, "prefix_roots": self.prefix_roots}
        return state

    def load_state_dict(self, state):
        super().load_state_dict(state)
        stream = state["broad_stream"]
        for k, v in stream["sketches"].items():
            self.phase_sketches[k] = Cardinality(v)
        self.phase_roots = stream["roots"]
        self.phase_histograms = stream["histograms"]
        self.requested_distances, self.initial_distances = stream["requested"], stream["initial"]
        self.actual_distances = stream["actual"]
        self.clear_requested_distances = stream.get("clear_requested", {})
        self.clear_actual_distances = stream.get("clear_actual", {})
        self.bundle_histograms = stream.get("bundle_histograms", {})
        self.bundle_update_counts = stream.get("bundle_update_counts", {})
        self.legacy_update_distances = stream.get("legacy_update_distances", {})
        self.fork_degrees, self.stream_calls = stream["fork_degrees"], stream["stream_calls"]
        self.prefix_roots = stream["prefix_roots"]

    def report(self):
        result = super().report()
        result["historical_draw_counters_additive_across_ranks"] = (
            int(self.options.get("previous_sampler_world", self.world)) == self.world)
        result["stream_phase"] = {"roots": int(self.phase_roots.sum()), "local_distance_by_goal": self.phase_histograms,
            "requested_stream_distance_draws": self.requested_distances,
            "before_cross_branch_distance_draws": self.initial_distances,
            "actual_stream_distance_draws": self.actual_distances, "fork_degree_draws": self.fork_degrees,
            "requested_indexed_clear_distances": self.clear_requested_distances,
            "actual_indexed_clear_parent_distances": self.clear_actual_distances,
            "branch_bundle_actual_update_draws": self.bundle_update_counts,
            "branch_bundle_actual_distance_draws": self.bundle_histograms,
            "legacy_actual_huber_distances": self.legacy_update_distances,
            "local_estimated_distinct": {k: v.estimate() for k, v in self.phase_sketches.items()},
            "estimate_relative_standard_error": 1.04 / math.sqrt(1 << 14),
            "scope": "This new phase only, packed global state-pair HLL shared identity across roots; per-rank until merged at complete checkpoint. Nonzero score derivatives do not imply nonzero parameter gradients. Earlier distinct exposure unknown outside the certified broad prefix."}
        return result
