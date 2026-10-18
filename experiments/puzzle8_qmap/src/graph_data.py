"""One joint pool spanning the complete graph, all endpoints and solution forks."""
import json
import numpy as np
from .branch_data import Epoch
from grounded_llm.qmap_sampling import CoverageSampler


class StreamEpoch:
    """Cycle-walk a bounded integer permutation without allocating billions of IDs."""
    def __init__(self, size, rng):
        self.size, self.rng = int(size), rng
        self.mask = np.uint64((1 << max(1, (self.size - 1).bit_length())) - 1)
        self.position = self.draws = 0
        self.key = np.uint64(rng.integers(0, 2 ** 63)) & self.mask

    def permute(self, values):
        x = (values ^ self.key) & self.mask
        x = ((x ^ (x >> np.uint64(17))) * np.uint64(0xbf58476d1ce4e5b9)) & self.mask
        x = ((x ^ (x >> np.uint64(13))) * np.uint64(0x94d049bb133111eb)) & self.mask
        return (x ^ (x >> np.uint64(16))) & self.mask

    @property
    def passes(self):
        return self.draws / self.size

    def draw(self, count):
        pieces = []
        while count:
            n = min(count, self.size - self.position)
            ids = self.permute(np.arange(self.position, self.position + n, dtype=np.uint64))
            invalid = ids >= self.size
            while invalid.any():
                ids[invalid] = self.permute(ids[invalid])
                invalid = ids >= self.size
            pieces.append(ids.astype(np.int64))
            count -= n
            self.position += n
            self.draws += n
            if self.position == self.size:
                self.position = 0
                self.key = np.uint64(self.rng.integers(0, 2 ** 63)) & self.mask
        return np.concatenate(pieces) if pieces else np.empty(0, dtype=np.int64)


def structure(env, allowed, protected, levels):
    """Every first-action fork, multiscale descendants, cross-branch endpoints.

    Exact distances come from the full graph, including alternate routes between
    branches. Then close every included decision under all optimal successors.
    """
    ids = np.arange(env.size)
    depth = env.distance(ids, 0)
    children = env.next
    good = (children >= 0) & (depth[np.maximum(children, 0)] == depth[:, None] - 1)
    pointer = children[ids, good.argmax(-1)].copy()
    pointer[0] = 0
    parents, actions = np.where(children >= 0)
    first = children[parents, actions]
    parts = []
    for steps in levels:
        future = first.copy()
        for _ in range(steps):
            future = pointer[future]
        parts.extend((parents.astype(np.int64) * env.size + future,
                      first.astype(np.int64) * env.size + future))
    leaves = np.maximum(children, 0).copy()
    for _ in range(8):
        leaves = pointer[leaves]
    valid = children >= 0
    for a in range(children.shape[1]):
        for b in range(children.shape[1]):
            keep = valid[:, a] & valid[:, b]
            parts.append(leaves[keep, a].astype(np.int64) * env.size + leaves[keep, b])
    keys = np.unique(np.concatenate(parts))
    source, goal = keys // env.size, keys % env.size
    keys = keys[allowed[source] & ~np.isin(goal, protected) & (source != goal)]
    known, frontier = keys, keys
    for iteration in range(env.diameter):
        proposed = []
        for start in range(0, len(frontier), 200000):
            rows = frontier[start:start + 200000]
            source, goal = rows // env.size, rows % env.size
            candidates = env.next[source]
            truth = env.distance(np.maximum(candidates, 0), goal[:, None])
            good = (candidates >= 0) & (truth == env.distance(source, goal)[:, None] - 1)
            good &= allowed[np.maximum(candidates, 0)] & (candidates != goal[:, None])
            proposed.append((np.maximum(candidates, 0).astype(np.int64) * env.size + goal[:, None])[good])
        frontier = np.setdiff1d(np.unique(np.concatenate(proposed)), known, assume_unique=True)
        if not len(frontier):
            break
        known = np.union1d(known, frontier)
        print(json.dumps({"structure_closure_iteration": iteration + 1,
                          "structure_decisions": len(known), "new_decisions": len(frontier)}), flush=True)
    return np.column_stack((known // env.size, known % env.size)).astype(np.int32)


class GraphData:
    def __init__(self, env, data, options, rng, out):
        if not hasattr(env, "full_distances"):
            raise ValueError("Joint full-graph training requires the complete source-goal table")
        self.env, self.data, self.options, self.rng = env, data, options, rng
        self.protected = np.unique(np.r_[data["pairs_validation"][:, 1], data["pairs_test"][:, 1]])
        self.sources = np.flatnonzero(~np.isin(np.arange(env.size), self.protected))
        self.allowed = ~np.isin(np.arange(env.size), self.protected)
        self.allowed &= ~np.isin(env.next, self.protected).any(-1)
        self.parents = np.flatnonzero(self.allowed)
        self.ps = CoverageSampler(data["pairs_train"][:, 2], rng, len(data["pairs_train"]), .5)
        self.rs = CoverageSampler(data["parent_distances_train"], rng, len(data["tasks_train"]), .5)
        self.gp = StreamEpoch(len(self.sources) ** 2, rng)
        self.gr = StreamEpoch(len(self.parents) * len(self.sources), rng)
        self.sp, self.sr = Epoch(env.size, rng), Epoch(env.size - 1, rng)
        a, b = np.where(env.next >= 0)
        edges = np.column_stack((a, env.next[a, b]))
        self.edges = edges[~np.isin(edges, self.protected).any(-1)].astype(np.int32)
        self.es = Epoch(len(self.edges), rng)
        route_path = out / "graph_structure.npz"
        self.routes = structure(env, self.allowed, self.protected, options["descendant_steps"])
        np.savez_compressed(route_path, tasks=self.routes)
        self.rp, self.rr = StreamEpoch(len(self.routes), rng), StreamEpoch(len(self.routes), rng)
        self.hard = np.column_stack((self.parents[self.parents != 0], np.zeros((self.parents != 0).sum(), dtype=int)))
        self.hard_pair_draws = self.hard_decision_draws = 0
        self.seen_sources = np.zeros(env.size, dtype=bool)
        self.seen_goals = np.zeros(env.size, dtype=bool)
        summary = {"states": env.size, "full_exact_ordered_pairs": env.size ** 2,
                   "available_joint_pairs": self.gp.size, "available_joint_decisions": self.gr.size,
                   "structure_decisions": len(self.routes), "structure_parent_states": len(np.unique(self.routes[:, 0])),
                   "structure_goal_states": len(np.unique(self.routes[:, 1])),
                   "directed_edges": len(self.edges), "protected_goal_ids": self.protected.tolist(),
                   "scope": "One joint training pool. Full source-goal distances and full legal stars; every parent fork, multiscale downstream and cross-branch relations; no new loss or supervision type."}
        (out / "graph_data_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps({"joint_graph_data": summary}), flush=True)

    def refresh(self, hard_parents):
        parents = np.unique(np.r_[hard_parents, self.env.next[hard_parents].ravel()])
        parents = parents[(parents >= 0) & (parents != 0)]
        parents = parents[self.allowed[parents]]
        if len(parents):
            self.hard = np.column_stack((parents, np.zeros(len(parents), dtype=int))).astype(np.int32)

    def counts(self, count):
        a, b = [round(count * f) for f in self.options["fractions"][:2]]
        return a, b, count - a - b

    def graph(self, count, ranking):
        route_count = round(count * self.options["route_fraction"])
        hard_count = round(route_count * self.options["hard_fraction"])
        sampler, sources = (self.gr, self.parents) if ranking else (self.gp, self.sources)
        index = sampler.draw(count - route_count)
        rows = np.column_stack((sources[index % len(sources)], self.sources[index // len(sources)]))
        self.seen_sources[rows[:, 0]] = True
        self.seen_goals[rows[:, 1]] = True
        route_sampler = self.rr if ranking else self.rp
        routes = self.routes[route_sampler.draw(route_count - hard_count)]
        hard = self.hard[self.rng.integers(len(self.hard), size=hard_count)]
        if ranking:
            self.hard_decision_draws += hard_count
        else:
            self.hard_pair_draws += hard_count
        packed = np.concatenate((routes, hard)).copy()
        if not ranking and len(packed):
            choices = np.column_stack((packed[:, 0], self.env.next[packed[:, 0]]))
            random = self.rng.random(choices.shape)
            random[choices < 0] = -1
            packed[:, 0] = choices[np.arange(len(choices)), random.argmax(-1)]
        return np.concatenate((rows, packed))

    def pairs(self, count):
        a, b, c = self.counts(count)
        edge_count = round(c * self.options["edge_fraction"])
        restore = self.sp.draw(b)
        rows = np.concatenate((self.data["pairs_train"][self.ps.draw(a), :2],
                               np.column_stack((restore, np.zeros(b, dtype=int))),
                               self.graph(c - edge_count, False), self.edges[self.es.draw(edge_count)])).astype(np.int32)
        return np.column_stack((rows, self.env.distance(rows[:, 0], rows[:, 1])))

    def tasks(self, count):
        a, b, c = self.counts(count)
        rows = np.concatenate((self.data["tasks_train"][self.rs.draw(a)],
                               np.column_stack((self.sr.draw(b) + 1, np.zeros(b, dtype=int))),
                               self.graph(c, True))).astype(np.int32)
        children = self.env.next[rows[:, 0]]
        truth = np.where(children >= 0, self.env.distance(np.maximum(children, 0), rows[:, 1, None]),
                         self.env.diameter + 1).astype(np.int8)
        return rows, children, truth

    def report(self):
        return {"original_pair_passes": self.ps.passes, "original_decision_passes": self.rs.passes,
                "restoration_pair_passes": self.sp.passes, "restoration_decision_passes": self.sr.passes,
                "all_pairs_draws": self.gp.draws, "all_pairs_passes": self.gp.passes,
                "all_decision_draws": self.gr.draws, "all_decision_passes": self.gr.passes,
                "observed_broad_sources": int(self.seen_sources.sum()), "observed_broad_goals": int(self.seen_goals.sum()),
                "structure_pair_passes": self.rp.passes, "structure_decision_passes": self.rr.passes,
                "hard_pair_draws": self.hard_pair_draws, "hard_decision_draws": self.hard_decision_draws,
                "edge_pair_passes": self.es.passes, "structure_decisions": len(self.routes), "hard_contexts": len(self.hard)}
