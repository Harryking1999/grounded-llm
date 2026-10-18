"""Replay training-only wrong forks and stream their exact optimal continuations."""
import json
from pathlib import Path

import numpy as np

from .full_graph_sampling import JointData


class FailureData(JointData):
    def __init__(self, path, legacy_path, options, rank=0, world=1):
        super().__init__(path, legacy_path, options, rank, world)
        diagnostic = json.loads(Path(options["failure_diagnosis"]).read_text())
        # Deliberately ignore monitor_long. Only training roots can seed repair.
        rows = diagnostic["training_long"]["rows"]
        self.packets = [[] for _ in self.roots]
        self.packet_weights = [[] for _ in self.roots]
        self.negative_packets = [[] for _ in self.roots]
        self.retention_packets = [[] for _ in self.roots]
        for row in rows:
            board = row["board"]
            if row["source"] != self.roots[board].record["initial"]:
                raise ValueError("Repair rows must belong to the exact training root")
            root = self.roots[board]
            # Keep both first wrong forks and the actual on-policy contexts.
            stars = ([row["failure"]] if not row["success"] else []) + row["trajectory"]
            seen = set()
            for star in stars:
                source, goal = int(star["source"]), int(star["goal"])
                if (source, goal) in seen:
                    continue
                seen.add((source, goal))
                self.packets[board].append(self.encode_pair(root, source, goal))
                first_error = not row["success"] and star is row["failure"]
                self.packet_weights[board].append(4 if first_error else 1)
            if not row["success"] and row["failure"].get("candidates"):
                failure = row["failure"]
                chosen = next(c for c in failure["candidates"] if c["action"] == failure["chosen_action"])
                packet = self.encode_pair(root, int(chosen["state"]), int(failure["goal"]))
                # Direct Huber negatives target the actual wrong legal child.
                # Ranking continues to use its reachable parent and full fork.
                repeat = 3 if failure["failure_type"] == "removed_goal_cell" else 1
                self.negative_packets[board].extend([packet] * repeat)
        if options.get("retention_diagnosis"):
            retained = json.loads(Path(options["retention_diagnosis"]).read_text())["training_long"]["rows"]
            for row in retained:
                if row["success"]:
                    root = self.roots[row["board"]]
                    if row["source"] != root.record["initial"]:
                        raise ValueError("Retention must use matching training roots")
                    for star in row["trajectory"]:
                        self.retention_packets[row["board"]].append(
                            self.encode_pair(root, int(star["source"]), int(star["goal"])))
        self.packets = [np.asarray(rows, dtype=np.int64) for rows in self.packets]
        self.negative_packets = [np.asarray(rows, dtype=np.int64) for rows in self.negative_packets]
        self.retention_packets = [np.asarray(rows, dtype=np.int64) for rows in self.retention_packets]
        self.packet_weights = [np.asarray(w, dtype=float) / sum(w) for w in self.packet_weights]
        if any(not len(rows) for rows in self.packets):
            raise ValueError("Each training root needs at least one diagnosed context")
        self.packet_seen = [np.zeros(len(rows), dtype=bool) for rows in self.packets]
        self.repair_pair_draws = self.repair_rank_requested = 0
        self.optimal_transition_draws = 0
        self.repair_distances = np.zeros(64, dtype=np.int64)
        self.repair_pairs_seen, self.repair_forks_seen = set(), set()
        self.repair_candidates_seen = set()
        self.last_repair_count = 0
        self.hard_negative_draws = self.retention_draws = 0
        # Continue the certified broad prefix, without calling repeats unique.
        previous = options.get("previous_sampler_root")
        if previous:
            import torch
            step = options["previous_sampler_step"]
            state = torch.load(Path(previous) / f"sampler_{step:06d}_rank{rank}.pt",
                               map_location="cpu", weights_only=False)
            super().load_state_dict(state)
            if "failure_repair" in state:
                self.load_repair_tracking(state["failure_repair"], same_pool=False)

    @staticmethod
    def encode_pair(root, source, goal):
        return [[int(graph.rank(np.asarray([sum(((mask >> cell) & 1) << bit
                          for bit, cell in enumerate(cells))], dtype=np.uint64))[0])
                 for graph, cells in root.parts] for mask in (source, goal)]

    def repaired_rows(self, board, count, ranking):
        root = self.roots[board]
        total = count
        negatives = (round(count * self.options.get("hard_negative_fraction", 0))
                     if not ranking and len(self.negative_packets[board]) else 0)
        count -= negatives
        selected = self.rng.choice(len(self.packets[board]), size=count, p=self.packet_weights[board])
        rows = self.packets[board][selected]
        source, goal = rows[:, 0].copy(), rows[:, 1].copy()
        # New arbitrary residual goals keep repair from memorizing the two
        # diagnostic targets per root. Preserve the last nonempty descendant.
        fresh = round(count * self.options.get("fresh_fraction", .5))
        retain = (round(count * self.options.get("retention_fraction", 0))
                  if len(self.retention_packets[board]) else 0)
        if fresh + retain > count:
            raise ValueError("Fresh and retention fractions exceed repair budget")
        self.packet_seen[board][selected[fresh:count-retain if retain else count]] = True
        if retain:
            kept = self.retention_packets[board][self.rng.integers(len(self.retention_packets[board]), size=retain)]
            source[-retain:], goal[-retain:] = kept[:, 0], kept[:, 1]
            self.retention_draws += retain
        if fresh:
            initial = np.tile([g.initial_id for g, _ in root.parts], (fresh, 1))
            current = initial.copy()
            zero = np.asarray([g.zero_id for g, _ in root.parts])
            for _ in range(18):
                child = root.advance(current, self.rng, np.ones(fresh, dtype=int))
                keep = (child != zero).any(-1)
                current[keep] = child[keep]
            source[:fresh], goal[:fresh] = initial, current
        levels = np.asarray(self.options["repair_descendant_steps"])
        steps = levels[(np.arange(count) + int(self.rng.integers(len(levels)))) % len(levels)]
        for depth in range(int(steps.max(initial=0))):
            parts = root.distance_parts(source, goal)
            distances = np.where((parts < 0).any(-1), -1, parts.sum(-1))
            active = (depth < steps) & (distances > 1)
            # Sample uniformly among every optimal outgoing action. This is a
            # label-side sampler, not search or filtering in the evaluated Q.
            choices = [[] for _ in range(count)]
            for i, (graph, _) in enumerate(root.parts):
                parents, children = graph.star(source[:, i])
                labels = graph.distance(graph.states[children], graph.states[goal[parents, i]])
                good = (active[parents] & (labels >= 0) &
                        (parts[parents].sum(-1) - parts[parents, i] + labels == distances[parents] - 1))
                for row, child in zip(parents[good], children[good]):
                    choices[int(row)].append((i, int(child)))
            for row in np.flatnonzero(active):
                if not choices[row]:
                    raise AssertionError("Exact reachable pair lacks optimal successor")
                part, child = choices[row][int(self.rng.integers(len(choices[row])))]
                source[row, part] = child
                self.optimal_transition_draws += 1
        labels = root.distance_parts(source, goal)
        distances = np.where((labels < 0).any(-1), -1, labels.sum(-1))
        if (distances < 0).any():
            raise AssertionError("Optimal continuation lost reachability")
        if negatives:
            bad = self.negative_packets[board][self.rng.integers(len(self.negative_packets[board]), size=negatives)]
            parts = root.distance_parts(bad[:, 0], bad[:, 1])
            if (parts >= 0).all(-1).any():
                raise AssertionError("Hard negative must be exactly unreachable")
            source, goal = np.concatenate((source, bad[:, 0])), np.concatenate((goal, bad[:, 1]))
            distances = np.r_[distances, np.full(negatives, len(self.repair_distances) - 1)]
            self.hard_negative_draws += negatives
        self.repair_distances += np.bincount(distances, minlength=len(self.repair_distances))
        if ranking:
            self.repair_rank_requested += total
        else:
            self.repair_pair_draws += total
        return source, goal

    def root_rows(self, board, count, ranking):
        repair = round(count * self.options["failure_fraction"])
        # The rest continues the same broad, clear, structure and legacy mix.
        source, goal = super().root_rows(board, count - repair, ranking)
        a, b = self.repaired_rows(board, repair, ranking)
        self.last_repair_count = repair if ranking else 0
        keys = self.repair_forks_seen if ranking else self.repair_pairs_seen
        keys.update(self.relation_keys(self.roots[board].packed(a), self.roots[board].packed(b)))
        (self.rank_counts if ranking else self.pair_counts).setdefault("failure_repair", 0)
        (self.rank_counts if ranking else self.pair_counts)["failure_repair"] += repair
        return np.concatenate((source, a)), np.concatenate((goal, b))

    @staticmethod
    def relation_keys(source, goal):
        return [(int(s[0]) | (int(s[1]) << 64)) << 100 | int(g[0]) | (int(g[1]) << 64)
                for s, g in zip(source, goal)]

    def stars(self, root, source, goal):
        result = super().stars(root, source, goal)
        if result is not None:
            s, g, _, group = result
            selected = group >= len(source) - self.last_repair_count
            self.repair_candidates_seen.update(self.relation_keys(s[selected], g[selected]))
        return result

    def state_dict(self):
        result = super().state_dict()
        result["failure_repair"] = {"packet_seen": self.packet_seen,
            "pair_draws": self.repair_pair_draws, "rank_requested": self.repair_rank_requested,
            "optimal_transition_draws": self.optimal_transition_draws,
            "distances": self.repair_distances,
            "pairs_seen": self.repair_pairs_seen, "forks_seen": self.repair_forks_seen,
            "candidates_seen": self.repair_candidates_seen,
            "hard_negative_draws": self.hard_negative_draws, "retention_draws": self.retention_draws}
        return result

    def load_state_dict(self, state):
        super().load_state_dict(state)
        self.load_repair_tracking(state["failure_repair"])

    def load_repair_tracking(self, repair, same_pool=True):
        if same_pool:
            self.packet_seen = repair["packet_seen"]
        self.repair_pair_draws = repair["pair_draws"]
        self.repair_rank_requested = repair["rank_requested"]
        self.optimal_transition_draws = repair["optimal_transition_draws"]
        self.repair_distances = repair["distances"]
        self.repair_pairs_seen = repair["pairs_seen"]
        self.repair_forks_seen = repair["forks_seen"]
        self.repair_candidates_seen = repair["candidates_seen"]
        self.hard_negative_draws = repair.get("hard_negative_draws", 0)
        self.retention_draws = repair.get("retention_draws", 0)

    def report(self):
        result = super().report()
        result["failure_repair"] = {"training_only_seed_contexts": sum(map(len, self.packets)),
            "different_seed_contexts_drawn": sum(int(v.sum()) for v in self.packet_seen),
            "seed_count_scope": "Current diagnostic pool; exact relation sets and draw counters retain the parent checkpoint prefix",
            "pair_draws": self.repair_pair_draws, "rank_contexts_requested": self.repair_rank_requested,
            "optimal_transition_draws": self.optimal_transition_draws,
            "local_distinct_huber_relations": len(self.repair_pairs_seen),
            "local_distinct_forks": len(self.repair_forks_seen),
            "local_distinct_candidate_relations": len(self.repair_candidates_seen),
            "parent_distance_draws": self.repair_distances.tolist(),
            "distance_histogram_unreachable_index": 63,
            "hard_negative_huber_draws": self.hard_negative_draws,
            "retention_context_draws": self.retention_draws,
            "scope": "All legal candidates at each selected fork; all optimal actions have nonzero sampling probability; full closure is queryable, not exhaustively gradient-exposed"}
        return result
