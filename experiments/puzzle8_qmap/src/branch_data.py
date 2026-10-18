"""Exact multi-goal graph relations and complete forks with downstream closure."""
from collections import deque
import json
import numpy as np
from grounded_llm.qmap_sampling import CoverageSampler


class Epoch:
    def __init__(self, size, rng):
        self.size, self.rng = int(size), rng
        self.order = rng.permutation(self.size)
        self.position = self.draws = 0

    @property
    def passes(self):
        return self.draws / self.size

    def draw(self, count):
        pieces = []
        while count:
            n = min(count, self.size - self.position)
            pieces.append(self.order[self.position:self.position + n])
            self.position += n
            self.draws += n
            count -= n
            if self.position == self.size:
                self.order = self.rng.permutation(self.size)
                self.position = 0
        return np.concatenate(pieces) if pieces else np.empty(0, dtype=np.int64)


def choose_goals(env, forbidden, count, seed, root_count):
    """Select diverse intermediate endpoints from actual full-fork continuations."""
    rng = np.random.default_rng(seed)
    depth = env.distance(np.arange(env.size), 0)
    roots = []
    for level in range(1, env.diameter + 1):
        pool = np.flatnonzero(depth == level)
        roots.extend(rng.choice(pool, min(len(pool), max(1, root_count // env.diameter)), replace=False))
    contexts = {}
    for root in roots:
        for child in env.next[root]:
            if child < 0:
                continue
            state = int(child)
            while state:
                contexts.setdefault(state, int(root))
                children = env.next[state]
                good = children[(children >= 0) & (depth[np.maximum(children, 0)] == depth[state] - 1)]
                state = int(rng.choice(good))
    # Rotate depth layers, spreading selected goals across blank positions too.
    selected = []
    pools = []
    for level in range(1, env.diameter + 1):
        pool = np.array([s for s in contexts if depth[s] == level and s not in forbidden], dtype=int)
        rng.shuffle(pool)
        if len(pool):
            blank = (env.positions[pool] == 0).argmax(-1)
            groups = [pool[blank == b].tolist() for b in range(9)]
            ordered = []
            while any(groups):
                for group in groups:
                    if group:
                        ordered.append(group.pop())
            pools.append(ordered)
    while len(selected) < count and any(pools):
        for pool in pools:
            if pool and len(selected) < count:
                selected.append(pool.pop(0))
    if len(selected) != count:
        raise ValueError("Fork trajectories did not supply enough distinct goal endpoints")
    goals = np.asarray(selected, dtype=np.int32)
    return goals, np.array([contexts[s] for s in goals], dtype=np.int32)


def route_closure(env, goals, roots, allowed):
    """For every root, include all first branches and every optimal continuation.

    All legal candidates remain present at every included decision. No branch is
    called unreachable in this reversible puzzle. Protected evaluation endpoints
    can interrupt training support; count these exceptions explicitly.
    """
    tasks, blocked = [], 0
    for goal, seed_roots in zip(goals, roots):
        depth = env.distance(np.arange(env.size), goal)
        seeds = np.unique(np.r_[seed_roots, env.next[seed_roots].ravel()])
        pending = deque(int(s) for s in seeds if s >= 0)
        seen = set()
        while pending:
            source = pending.popleft()
            if source == goal or source in seen:
                continue
            seen.add(source)
            if allowed[source]:
                tasks.append((source, int(goal)))
            else:
                blocked += 1
            children = env.next[source]
            good = children[(children >= 0) & (depth[np.maximum(children, 0)] == depth[source] - 1)]
            if not len(good):
                raise ValueError("Exact graph has a nonterminal state without an optimal successor")
            pending.extend(map(int, good))
    return np.asarray(tasks, dtype=np.int32).reshape(-1, 2), blocked


class BranchData:
    def __init__(self, env, data, options, rng, out):
        self.env, self.data, self.options, self.rng = env, data, options, rng
        self.ps = CoverageSampler(data["pairs_train"][:, 2], rng, len(data["pairs_train"]), .5)
        self.rs = CoverageSampler(data["parent_distances_train"], rng, len(data["tasks_train"]), .5)
        self.protected = np.unique(np.r_[data["pairs_validation"][:, 1], data["pairs_test"][:, 1]])
        original = np.unique(data["pairs_train"][:, 1])
        forbidden = set(np.r_[original, self.protected].tolist())
        self.goals, fork_roots = choose_goals(env, forbidden, options["new_goals"],
                                             options["seed"], options["fork_roots"])
        old_goals, old_tables = list(env.goal_columns), env.tables
        env.prepare_goals(self.goals)
        env.tables = np.concatenate((old_tables, env.tables))
        all_goals = np.r_[old_goals, self.goals]
        env.goal_columns = {int(g): i for i, g in enumerate(all_goals)}
        np.savez_compressed(out / "expanded_oracle.npz", goals=all_goals, distances=env.tables)
        self.sources = np.flatnonzero(~np.isin(np.arange(env.size), self.protected))
        self.allowed = ~np.isin(np.arange(env.size), self.protected)
        self.allowed &= ~np.isin(env.next, self.protected).any(-1)
        self.parents = np.flatnonzero(self.allowed)
        self.ep = Epoch(len(self.sources) * len(self.goals), rng)
        self.er = Epoch(len(self.parents) * len(self.goals), rng)
        self.sp = Epoch(env.size, rng)
        self.sr = Epoch(env.size - 1, rng)
        a, b = np.where(env.next >= 0)
        edge = np.column_stack((a, env.next[a, b]))
        edge = edge[~np.isin(edge, self.protected).any(-1)]
        self.edges = np.column_stack((edge, np.ones(len(edge), dtype=int))).astype(np.int32)
        self.edge_sampler = Epoch(len(edge), rng)
        self.route_goals = np.r_[original, self.goals]
        self.roots = []
        for goal in self.route_goals:
            depth = env.distance(self.parents, goal)
            chosen = []
            for level in np.unique(depth[depth > 0]):
                chosen.append(int(rng.choice(self.parents[depth == level])))
            if goal in self.goals:
                chosen.append(int(fork_roots[np.flatnonzero(self.goals == goal)[0]]))
            self.roots.append(np.unique(chosen).astype(int))
        self.routes, blocked = route_closure(env, self.route_goals, self.roots, self.allowed)
        self.pack = Epoch(len(self.routes), rng)
        self.base_routes = self.routes.copy()
        summary = {"new_goal_ids": self.goals.tolist(), "new_goal_origin_fork_roots": fork_roots.tolist(),
                   "original_goal_count": len(original), "new_goal_count": len(self.goals),
                   "available_new_pairs": self.ep.size, "available_new_decisions": self.er.size,
                   "complete_directed_edge_pairs": len(self.edges),
                   "protected_goal_ids": self.protected.tolist(),
                   "fixed_route_decisions": len(self.routes), "protected_decisions_omitted": blocked,
                   "full_fork_roots": sum(map(len, self.roots)),
                   "scope": "Exact full-state/new-goal Cartesian relations; forks include every legal first action and all optimal downstream actions. Heldout endpoint relations stay excluded."}
        (out / "branch_data_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        np.savez_compressed(out / "branch_routes.npz", tasks=self.routes,
                            new_goals=self.goals, goal_origin_roots=fork_roots)

    def refresh(self, hard_parents):
        if not self.options["route_fraction"]:
            return
        # Mine only standard-goal policy errors. The fixed multi-goal closures
        # remain present; refresh adds the full forks and exits of current errors.
        hard = np.intersect1d(np.asarray(hard_parents), self.parents)
        if len(hard):
            roots = self.rng.choice(hard, min(len(hard), self.options["hard_roots"]), replace=False)
            added, _ = route_closure(self.env, [0], [roots], self.allowed)
            self.routes = np.unique(np.concatenate((self.base_routes, added)), axis=0)
            self.pack = Epoch(len(self.routes), self.rng)

    def counts(self, count):
        a, b = [round(count * f) for f in self.options["fractions"][:2]]
        return a, b, count - a - b

    def expanded(self, count, ranking):
        packed = round(count * self.options["route_fraction"])
        sampler, sources = (self.er, self.parents) if ranking else (self.ep, self.sources)
        index = sampler.draw(count - packed)
        rows = np.column_stack((sources[index % len(sources)], self.goals[index // len(sources)]))
        if packed:
            routes = self.routes[self.pack.draw(packed)].copy()
            if not ranking:
                # Ordinary distance pairs include every branch's candidate-to-goal
                # relations, not only the selected continuation.
                for i, (source, goal) in enumerate(routes):
                    children = self.env.next[source]
                    choices = np.r_[source, children[children >= 0]]
                    routes[i, 0] = self.rng.choice(choices)
            rows = np.concatenate((rows, routes))
        return rows

    def pairs(self, count):
        a, b, c = self.counts(count)
        edges = round(c * self.options["edge_fraction"])
        restore = self.sp.draw(b)
        extra = self.expanded(c - edges, False)
        return np.concatenate((self.data["pairs_train"][self.ps.draw(a)],
                               np.column_stack((restore, np.zeros(b, dtype=int), self.env.distance(restore, 0))),
                               np.column_stack((extra, self.env.distance(extra[:, 0], extra[:, 1]))),
                               self.edges[self.edge_sampler.draw(edges)]))

    def tasks(self, count):
        a, b, c = self.counts(count)
        original = self.data["tasks_train"][self.rs.draw(a)]
        restore = np.column_stack((self.sr.draw(b) + 1, np.zeros(b, dtype=int)))
        tasks = np.concatenate((original, restore, self.expanded(c, True))).astype(np.int32)
        children = self.env.next[tasks[:, 0]]
        truth = np.where(children >= 0, self.env.distance(np.maximum(children, 0), tasks[:, 1, None]),
                         self.env.diameter + 1).astype(np.int8)
        return tasks, children, truth

    def report(self):
        return {"original_pair_passes": self.ps.passes, "original_decision_passes": self.rs.passes,
                "restoration_pair_passes": self.sp.passes, "restoration_decision_passes": self.sr.passes,
                "expanded_pair_passes": self.ep.passes, "expanded_decision_passes": self.er.passes,
                "edge_pair_passes": self.edge_sampler.passes,
                "new_goals": len(self.goals), "route_decisions": len(self.routes),
                "route_draws_since_refresh": self.pack.draws}
