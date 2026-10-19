"""Data-only sampling: original relations, complete restoration and local graph packs."""
import numpy as np
from grounded_llm.core.qmap.sampling import CoverageSampler


def local_relations(env, protected_goals):
    """All allowed one/two-edge pairs and full stars toward two-edge goals.

    Puzzle8 is bipartite. A candidate of a parent at distance two from a goal
    is exactly one or three steps away; adjacency resolves which, without an
    approximate distance label. Exclude heldout goal endpoints in either order.
    """
    parents, actions = np.where(env.next >= 0)
    children = env.next[parents, actions]
    edge = np.column_stack((parents, children))
    grandchild = env.next[children]
    source = np.repeat(parents, 4)
    target = grandchild.ravel()
    keep = (target >= 0) & (source != target)
    two = np.unique(np.column_stack((source[keep], target[keep])), axis=0)
    edge = edge[~np.isin(edge, protected_goals).any(-1)]
    two = two[~np.isin(two, protected_goals).any(-1)]
    pairs = np.concatenate((np.column_stack((edge, np.ones(len(edge), dtype=int))),
                            np.column_stack((two, np.full(len(two), 2)))))
    stars = env.next[two[:, 0]]
    keep = ~np.isin(stars, protected_goals).any(-1)
    tasks, stars = two[keep], stars[keep]
    adjacent = (env.next[np.maximum(stars, 0)] == tasks[:, 1, None, None]).any(-1)
    truth = np.where(stars < 0, env.diameter + 1, np.where(adjacent, 1, 3))
    return pairs.astype(np.int32), tasks.astype(np.int32), stars, truth.astype(np.int8)


class DataMix:
    def __init__(self, env, data, options, rng):
        self.env, self.data, self.options, self.rng = env, data, options, rng
        self.fractions = np.asarray(options["fractions"], dtype=float)
        if len(self.fractions) != 4 or np.any(self.fractions <= 0) or not np.isclose(self.fractions.sum(), 1):
            raise ValueError("Expected original/restoration/context/local sampling fractions")
        self.ps = CoverageSampler(data["pairs_train"][:, 2], rng, len(data["pairs_train"]), .5)
        self.rs = CoverageSampler(data["parent_distances_train"], rng, len(data["tasks_train"]), .5)
        self.standard_pairs = data["pairs_train"][data["pairs_train"][:, 1] == 0]
        self.standard_idx = np.flatnonzero(data["tasks_train"][:, 1] == 0)
        self.standard_lookup = np.full(env.size, -1, dtype=int)
        self.standard_lookup[data["tasks_train"][self.standard_idx, 0]] = self.standard_idx
        self.sp = CoverageSampler(self.standard_pairs[:, 2], rng, len(self.standard_pairs), 1.)
        self.sr = CoverageSampler(data["parent_distances_train"][self.standard_idx], rng, len(self.standard_idx), 1.)
        protected = np.unique(np.r_[data["pairs_validation"][:, 1], data["pairs_test"][:, 1]])
        self.lp, self.lt, self.lc, self.ld = local_relations(env, protected)
        self.lps = CoverageSampler(self.lp[:, 2], rng, len(self.lp), .5)
        self.lrs = CoverageSampler(np.full(len(self.lt), 2), rng, len(self.lt), 1.)
        self.context = data["tasks_train"][self.standard_idx, 0]

    def refresh(self, hard_parents):
        depth = self.env.distance(np.arange(self.env.size), 0)
        current = np.asarray(hard_parents, dtype=int)
        parts = [current]
        for _ in range(self.options["exit_steps"]):
            current = current[current != 0]
            if not len(current):
                break
            children = self.env.next[current]
            good = (children >= 0) & (depth[np.maximum(children, 0)] == depth[current, None] - 1)
            parts.append(children[children >= 0])
            current = children[np.arange(len(current)), good.argmax(-1)]
            parts.append(current)
        pool = np.unique(np.concatenate(parts))
        pool = pool[(pool != 0) & (self.standard_lookup[pool] >= 0)]
        if len(pool):
            self.context = pool

    def counts(self, count):
        sizes = np.floor(count * self.fractions).astype(int)
        sizes[0] += count - sizes.sum()
        return sizes

    def pairs(self, count):
        a, b, c, d = self.counts(count)
        context = self.rng.choice(self.context, c)
        # Include the selected context's complete legal star as ordinary pairs.
        choices = np.column_stack((context, self.env.next[context]))
        offset = self.rng.integers(choices.shape[1], size=c)
        selected = choices[np.arange(c), offset]
        selected = np.where(selected >= 0, selected, context)
        cp = np.column_stack((selected, np.zeros(c, dtype=int), self.env.distance(selected, 0)))
        return np.concatenate((self.data["pairs_train"][self.ps.draw(a)],
                               self.standard_pairs[self.sp.draw(b)], cp, self.lp[self.lps.draw(d)]))

    def tasks(self, count):
        a, b, c, d = self.counts(count)
        indices = np.r_[self.rs.draw(a), self.standard_idx[self.sr.draw(b)],
                         self.standard_lookup[self.rng.choice(self.context, c)]]
        local = self.lrs.draw(d)
        return (np.concatenate((self.data["tasks_train"][indices], self.lt[local])),
                np.concatenate((self.data["children_train"][indices], self.lc[local])),
                np.concatenate((self.data["child_distances_train"][indices], self.ld[local])))

    def report(self):
        return {"original_pair_passes": self.ps.passes, "original_decision_passes": self.rs.passes,
                "restoration_pair_passes": self.sp.passes, "restoration_decision_passes": self.sr.passes,
                "local_pair_passes": self.lps.passes, "local_decision_passes": self.lrs.passes,
                "local_pairs": len(self.lp), "local_stars": len(self.lt), "context_parents": len(self.context)}
