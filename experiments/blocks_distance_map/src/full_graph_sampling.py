"""Broad root-local pair traversal, complete forks and multiscale descendants."""
import json
import math
from pathlib import Path

import numpy as np

from experiments.gcml_counterexamples.src import blocks
from .full_graph_data import load_component, Root
from .multiboard_eval import listwise_groups


def packed_masks(masks):
    return np.asarray([(int(mask) & ((1 << 64) - 1), int(mask) >> 64) for mask in masks], dtype=np.uint64)


class JointData:
    def __init__(self, path, legacy_path, options, rank=0, world=1):
        path = Path(path)
        self.summary = json.loads((path / "summary.json").read_text())
        if not self.summary["complete"]:
            raise ValueError("Complete graph preparation is required")
        records = json.loads((path / "roots.json").read_text())
        gids = sorted({p["graph"] for r in records for p in r["components"]})
        self.graphs = {i: load_component(path / "graphs" / str(i)) for i in gids}
        self.roots = [Root(r, self.graphs) for r in records]
        self.options, self.rank, self.world = options, rank, world
        self.rng = np.random.default_rng(options["seed"] + 101 + rank)
        self.schedule_rng = np.random.default_rng(options["seed"] + 102)
        self.order = self.schedule_rng.permutation(len(self.roots))
        self.position = 0
        self.broad = np.zeros(len(self.roots), dtype=np.int64)
        self.clear = self.broad.copy()
        self.broad_rank = self.broad.copy()
        self.clear_rank = self.broad.copy()
        rng = np.random.default_rng(options["seed"] + 103)
        self.affine = []
        for root in self.roots:
            factors = []
            for _ in range(2):
                a = int(rng.integers(1, 1 << 20))
                while math.gcd(a, root.size) != 1:
                    a += 1
                factors.extend((a, int(rng.integers(root.size))))
            self.affine.append(factors)
        legacy = dict(np.load(legacy_path, allow_pickle=False))
        self.legacy = legacy
        self.legacy_masks = packed_masks(legacy["states"])
        self.legacy_rows = legacy["train"]
        self.legacy_groups = listwise_groups(self.legacy_rows, options["seed"], per_anchor=4)
        long_rows = (self.legacy_rows[:, 2] >= 6) & (self.legacy_rows[:, 2] <= 13)
        self.legacy_long_pairs = np.flatnonzero(long_rows)
        self.legacy_long_groups = np.flatnonzero(long_rows[self.legacy_groups].any(-1))
        self.legacy_strata = [np.flatnonzero(mask) for mask in (
            self.legacy_rows[:, 2] == 1, self.legacy_rows[:, 2] == 2,
            (self.legacy_rows[:, 2] >= 3) & (self.legacy_rows[:, 2] <= 4),
            self.legacy_rows[:, 2] >= 5, self.legacy_rows[:, 3] == 1,
            self.legacy_rows[:, 3] == 2)]
        self.legacy_strata = [(pool, .5 / sum(bool(len(p)) for p in self.legacy_strata[:4]))
                             for pool in self.legacy_strata[:4] if len(pool)] + [
                                 (pool, .5 / sum(bool(len(p)) for p in self.legacy_strata[4:]))
                                 for pool in self.legacy_strata[4:] if len(pool)]
        self.zero = {i: g.zero_id for i, g in self.graphs.items()}
        self.pair_counts = {name: 0 for name in ("legacy", "broad", "clear", "structure")}
        self.rank_counts = self.pair_counts.copy()
        self.action_exposed = np.zeros(len(blocks.PLACEMENTS), dtype=bool)
        action_index = {tile: i for i, (tile, _) in enumerate(blocks.PLACEMENTS)}
        self.action_maps = {}
        for root in self.roots:
            for graph, cells in root.parts:
                key = tuple(cells)
                self.action_maps[key] = {
                    tile: action_index[sum(((tile >> i) & 1) << cell for i, cell in enumerate(cells))]
                    for tile in graph.summary["tiles"]}
        self.rank_candidates = self.actual_rank_contexts = 0
        self.root_stars_seen = np.zeros(len(self.roots), dtype=bool)
        self.distance_bins = {kind: {key: 0 for key in ("0", "1", "2", "3-4", "5-8", "9+", "unreachable",
                                                     "finite_at_or_above_original_cap")}
                              for kind in ("huber", "rank_candidates")}
        self.action_candidate_contexts = np.zeros(len(blocks.PLACEMENTS), dtype=np.int64)
        self.ranking_gradient_counts = {"contexts_with_nonzero_score_gradient": 0,
            "candidates_with_nonzero_score_gradient": 0, "contexts_with_nonuniform_target": 0}

    def close(self):
        for graph in self.graphs.values():
            graph.close()

    def boards(self):
        result = []
        while len(result) < self.options["boards_per_batch"]:
            if self.position == len(self.order):
                self.order = self.schedule_rng.permutation(len(self.roots))
                self.position = 0
            result.append(int(self.order[self.position]))
            self.position += 1
        return result

    def broad_rows(self, board, count, ranking=False):
        root = self.roots[board]
        cursor = self.broad_rank if ranking else self.broad
        positions = np.uint64(cursor[board] % root.size) + self.rank + self.world * np.arange(count, dtype=np.uint64)
        layer = np.uint64(cursor[board] // root.size) + positions // root.size
        x = positions % root.size
        a, b, c, d = self.affine[board]
        source = (np.uint64(a) * x + np.uint64(b)) % root.size
        goal = (np.uint64(c) * ((x + layer) % root.size) + np.uint64(d)) % root.size
        cursor[board] += self.world * count
        return root.decode(source), root.decode(goal)

    def clear_rows(self, board, count, ranking=False):
        root = self.roots[board]
        cursor = self.clear_rank if ranking else self.clear
        x = (np.uint64(cursor[board] % root.size) + self.rank + self.world * np.arange(count, dtype=np.uint64)) % root.size
        a, b, _, _ = self.affine[board]
        cursor[board] += self.world * count
        source = root.decode((np.uint64(a) * x + np.uint64(b)) % root.size)
        # Half the clear-goal sources follow arbitrary subsets of the certified
        # twelve-placement tiling. These are reachable AND clearable, whereas
        # uniformly drawn reachable states are predominantly dead ends.
        witness = root.record["construction"]
        if witness and "shape_id" in witness[0]:
            rows = np.arange(0, count, 2)
            tiles = [blocks.PLACEMENTS[blocks.BY_ACTION[(move["shape_id"], move["row"], move["col"]) ]][0]
                     for move in witness]
            masks = []
            for depth in (np.arange(len(rows)) + int(self.rng.integers(len(tiles) + 1))) % (len(tiles) + 1):
                mask = int(root.record["initial"])
                for index in self.rng.choice(len(tiles), int(depth), replace=False):
                    mask ^= tiles[int(index)]
                masks.append(mask)
            for i, (graph, cells) in enumerate(root.parts):
                local = np.asarray([sum(((mask >> cell) & 1) << bit for bit, cell in enumerate(cells))
                                    for mask in masks], dtype=np.uint64)
                source[rows, i] = graph.rank(local)
        goal = np.tile([self.zero[p["graph"]] for p in root.record["components"]], (count, 1))
        return source, goal

    def structure_rows(self, board, count, cross):
        root = self.roots[board]
        source = root.decode(self.rng.integers(root.size, size=count))
        degree = sum(g.degrees(source[:, i])
                     for i, (g, _) in enumerate(root.parts))
        source[degree == 0] = [g.initial_id for g, _ in root.parts]
        # Keep full-root long continuations alongside uniformly drawn states;
        # uniform DAG states alone are usually near leaves or dead ends.
        source[::2] = [g.initial_id for g, _ in root.parts]
        scales = np.asarray([1, 2, 3, 4, 6, 8, 12, 16])
        steps = scales[(np.arange(count) + int(self.rng.integers(len(scales)))) % len(scales)]
        goal = root.advance(source, self.rng, steps)
        if cross:
            use = np.arange(count) % 2 == 0
            source[use] = root.advance(source[use], self.rng, np.ones(use.sum(), dtype=int))
        return source, goal

    def root_rows(self, board, count, ranking):
        a = round(count * self.options["broad_fraction"])
        b = round(count * self.options["clear_fraction"])
        groups = (self.broad_rows(board, a, ranking), self.clear_rows(board, b, ranking),
                  self.structure_rows(board, count - a - b, not ranking))
        counters = self.rank_counts if ranking else self.pair_counts
        for key, number in zip(("broad", "clear", "structure"), (a, b, count - a - b)):
            counters[key] += number
        source, goal = (np.concatenate([r[i] for r in groups]) for i in (0, 1))
        # Expose the complete fork at every construction root at least once.
        if ranking and count and not self.root_stars_seen[board]:
            source[-1] = [g.initial_id for g, _ in self.roots[board].parts]
            goal[-1] = [g.zero_id for g, _ in self.roots[board].parts]
            self.root_stars_seen[board] = True
        return source, goal

    def stars(self, root, source, goal):
        parts = root.distance_parts(source, goal)
        finite = (parts >= 0).all(-1)
        packed_source, packed_goal = root.packed(source), root.packed(goal)
        packed_children, targets, truth, group_ids = [], [], [], []
        for i, (graph, cells) in enumerate(root.parts):
            rows, children = graph.star(source[:, i])
            if not len(rows):
                continue
            new_part = graph.states[children]
            old_part = graph.states[source[rows, i]]
            changed = old_part ^ new_part
            action_ids = np.asarray([self.action_maps[tuple(cells)][int(tile)] for tile in changed])
            self.action_exposed[action_ids] = True
            np.add.at(self.action_candidate_contexts, action_ids, 1)
            packed = packed_source[rows].copy()
            for bit, cell in enumerate(cells):
                packed[:, cell // 64] ^= ((changed >> np.uint64(bit)) & np.uint64(1)) << np.uint64(cell % 64)
            distance = graph.distance(new_part, graph.states[goal[rows, i]])
            labels = np.where(finite[rows] & (distance >= 0),
                              parts[rows].sum(-1) - parts[rows, i] + distance, -1)
            packed_children.append(packed)
            targets.append(packed_goal[rows])
            truth.append(labels)
            group_ids.append(rows)
        if not packed_children:
            return None
        return tuple(np.concatenate(rows) for rows in (packed_children, targets, truth, group_ids))

    def batch(self, pair_count, rank_count):
        legacy_pair = round(pair_count * self.options["legacy_fraction"])
        legacy_rank = round(rank_count * self.options["legacy_fraction"])
        ids = []
        remaining = legacy_pair
        for i, (pool, weight) in enumerate(self.legacy_strata):
            count = remaining if i == len(self.legacy_strata) - 1 else round(legacy_pair * weight)
            ids.extend(self.rng.choice(pool, count).tolist())
            remaining -= count
        focus = self.options.get("legacy_long_fraction", 0)
        focused = round(legacy_pair * focus)
        if focused and len(self.legacy_long_pairs):
            # The strata above append finite rows before unreachable rows.
            # Replacing a prefix at focus=.5 erased ALL historical short finite
            # supervision, rather than mixing half focused and half broad rows.
            ids = np.asarray(ids, dtype=np.int64)
            slots = self.rng.choice(legacy_pair, focused, replace=False)
            ids[slots] = self.rng.choice(self.legacy_long_pairs, focused)
        old = self.legacy_rows[ids]
        sources, goals, labels = [self.legacy_masks[old[:, 0]]], [self.legacy_masks[old[:, 1]]], [old[:, 2]]
        group = self.legacy_groups[self.rng.integers(len(self.legacy_groups), size=legacy_rank)]
        focused = round(legacy_rank * focus)
        if focused and len(self.legacy_long_groups):
            group[:focused] = self.legacy_groups[self.rng.choice(self.legacy_long_groups, focused)]
        rows = self.legacy_rows[group]
        rank_sources = [self.legacy_masks[rows[..., 0]].reshape(-1, 2)]
        rank_goals = [self.legacy_masks[rows[..., 1]].reshape(-1, 2)]
        rank_labels, rank_groups = [rows[..., 2].ravel()], [np.repeat(np.arange(legacy_rank), 4)]
        group_offset = legacy_rank
        self.pair_counts["legacy"] += legacy_pair
        self.rank_counts["legacy"] += legacy_rank
        boards = self.boards()
        for number, board in enumerate(boards):
            n = (pair_count - legacy_pair) // len(boards) + (number < (pair_count - legacy_pair) % len(boards))
            root = self.roots[board]
            source, goal = self.root_rows(board, n, False)
            sources.append(root.packed(source)); goals.append(root.packed(goal))
            distance = root.distance_parts(source, goal)
            labels.append(np.where((distance < 0).any(-1), -1, distance.sum(-1)))
            n = (rank_count - legacy_rank) // len(boards) + (number < (rank_count - legacy_rank) % len(boards))
            source, goal = self.root_rows(board, n, True)
            star = self.stars(root, source, goal)
            if star is not None:
                s, g, d, groups = star
                _, groups = np.unique(groups, return_inverse=True)
                rank_sources.append(s); rank_goals.append(g); rank_labels.append(d)
                rank_groups.append(groups + group_offset)
                group_offset += int(groups.max()) + 1
        self.rank_candidates += sum(len(x) for x in rank_labels)
        self.actual_rank_contexts += group_offset
        pair_truth, rank_truth = np.concatenate(labels), np.concatenate(rank_labels)
        for kind, values in (("huber", pair_truth), ("rank_candidates", rank_truth)):
            for key, mask in (("0", values == 0), ("1", values == 1), ("2", values == 2),
                              ("3-4", (values >= 3) & (values <= 4)),
                              ("5-8", (values >= 5) & (values <= 8)),
                              ("9+", values >= 9), ("unreachable", values < 0),
                              ("finite_at_or_above_original_cap", values >= self.options.get("original_cap", 13))):
                self.distance_bins[kind][key] += int(mask.sum())
        return (np.concatenate(sources), np.concatenate(goals), pair_truth,
                np.concatenate(rank_sources), np.concatenate(rank_goals), rank_truth,
                np.concatenate(rank_groups))

    def state_dict(self):
        return {"rng": self.rng.bit_generator.state, "schedule_rng": self.schedule_rng.bit_generator.state,
                "order": self.order, "position": self.position, "broad": self.broad, "clear": self.clear,
                "broad_rank": self.broad_rank, "clear_rank": self.clear_rank,
                "pair_counts": self.pair_counts, "rank_counts": self.rank_counts,
                "action_exposed": self.action_exposed, "root_stars_seen": self.root_stars_seen,
                "distance_bins": self.distance_bins, "action_candidate_contexts": self.action_candidate_contexts,
                "rank_candidates": self.rank_candidates, "actual_rank_contexts": self.actual_rank_contexts,
                "ranking_gradient_counts": self.ranking_gradient_counts}

    def load_state_dict(self, state):
        self.rng.bit_generator.state = state["rng"]
        self.schedule_rng.bit_generator.state = state["schedule_rng"]
        for key in ("order", "position", "broad", "clear", "broad_rank", "clear_rank", "pair_counts", "rank_counts",
                    "action_exposed", "root_stars_seen", "rank_candidates", "actual_rank_contexts"):
            setattr(self, key, state[key])
        self.distance_bins = state["distance_bins"]
        self.action_candidate_contexts = state["action_candidate_contexts"]
        self.ranking_gradient_counts = state.get("ranking_gradient_counts", self.ranking_gradient_counts)

    def report(self):
        unique = sum(min(int(n), root.size ** 2) for n, root in zip(self.broad, self.roots))
        states = sum(min(int(n), root.size) for n, root in zip(self.broad, self.roots))
        return {"root_local_pairs_available": self.summary["total_root_local_pairs"],
                "broad_unique_root_local_pairs": unique,
                "broad_pair_fraction": unique / self.summary["total_root_local_pairs"],
                "broad_root_local_endpoint_states": states,
                "root_local_states_available": self.summary["total_root_local_states"],
                "broad_endpoint_fraction": states / self.summary["total_root_local_states"],
                "roots_exposed": int((self.broad > 0).sum()),
                "local_complete_initial_forks_exposed": int(self.root_stars_seen.sum()),
                "minimum_distinct_initial_root_contexts_per_action":
                    int(min(self.summary.get("action_training_root_counts", [0])))
                    if self.root_stars_seen.all() else None,
                "local_pair_draws": self.pair_counts, "local_rank_contexts": self.rank_counts,
                "world_size": self.world,
                "broad_rank_contexts_requested": int(self.broad_rank.sum()),
                "local_action_types_in_candidate_rows": int(self.action_exposed.sum()),
                "local_action_candidate_context_draws": self.action_candidate_contexts.tolist(),
                "distance_bin_draws": self.distance_bins,
                "actions_available": len(self.action_exposed),
                "local_rank_candidate_draws": self.rank_candidates,
                "local_actual_rank_contexts": self.actual_rank_contexts,
                "local_ranking_gradient_counts": self.ranking_gradient_counts,
                "note": "Broad pair cursors aggregate disjoint distributed ranks and count direct Huber pairs only. Rank counts are requested contexts; actionless parents contribute no ranking loss. Other draws are counts, not unique global-state coverage."}
