"""Exact symbolic BFS and vectorized indexing of complete removal graphs."""
import json
from pathlib import Path

import numpy as np


def build_symbolic(width, tiles, out, maximum_nodes):
    try:
        from dd.cudd import BDD
        backend = "dd.cudd"
    except ImportError:
        from dd.autoref import BDD
        backend = "dd.autoref"
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    bdd = BDD()
    bdd.configure(reordering=False)
    names = [f"x{i}" for i in range(width)]
    bdd.declare(*names)
    actions = []
    for tile in tiles:
        variables = {names[i]: True for i in range(width) if tile & (1 << i)}
        actions.append((variables, bdd.cube({name: False for name in variables})))
    frontier = bdd.cube({name: True for name in names})
    visited, layers = frontier, [frontier]
    for depth in range(1, width // 2 + 2):
        image = bdd.false
        for substitution, cleared in actions:
            image |= bdd.let(substitution, frontier) & cleared
        frontier = image & ~visited
        if frontier == bdd.false:
            break
        visited |= frontier
        layers.append(frontier)
        if len(bdd) > maximum_nodes:
            raise RuntimeError(f"Symbolic graph exceeds node budget without completing: {out}")
    else:
        raise AssertionError("Removal BFS exceeded its occupied-cell bound")
    size = int(bdd.count(visited, nvars=width))
    edges = sum(int(bdd.count(visited & bdd.cube(substitution), nvars=width))
                for substitution, _ in actions)
    level, low, high, counts = [width, width], [0, 1], [0, 1], [0, 1]
    memo = {int(bdd.false): 0, int(bdd.true): 1}

    def export(node):
        key = int(node)
        if key in memo:
            return memo[key]
        left, right = node.low, node.high
        if node.negated:
            left, right = ~left, ~right
        a, b = export(left), export(right)
        at = int(node.var[1:])
        index = len(level)
        memo[key] = index
        level.append(at); low.append(a); high.append(b)
        counts.append((counts[a] << (level[a] - at - 1)) +
                      (counts[b] << (level[b] - at - 1)))
        return index

    root = export(visited)
    layer_ids = [export(layer) for layer in layers]
    assert (counts[root] << level[root]) == size
    np.savez(out / "diagram.npz", level=np.asarray(level, dtype=np.int16),
             low=np.asarray(low, dtype=np.int32), high=np.asarray(high, dtype=np.int32),
             counts=np.asarray(counts, dtype=np.uint64), root=np.asarray(root),
             layers=np.asarray(layer_ids, dtype=np.int32))
    summary = {"width": width, "states": size, "edges": edges, "tiles": tiles,
               "representation": "symbolic_complete_BFS", "backend": backend,
               "diagram_nodes": len(level), "complete": True}
    (out / "summary.json").write_text(json.dumps(summary) + "\n")
    return summary


class StateIndex:
    def __init__(self, graph):
        self.graph = graph

    def __getitem__(self, indices):
        return self.graph.unrank(indices)


class SymbolicComponent:
    def __init__(self, path):
        path = Path(path)
        self.summary = json.loads((path / "summary.json").read_text())
        if not self.summary["complete"]:
            raise ValueError("Incomplete symbolic graph")
        with np.load(path / "diagram.npz") as data:
            for key in data.files:
                setattr(self, key, data[key].copy())
        self.root = int(self.root)
        self.width, self.size = self.summary["width"], self.summary["states"]
        self.tiles = np.asarray(self.summary["tiles"], dtype=np.uint64)
        self.states = StateIndex(self)
        self.initial_id, self.zero_id = self.size - 1, 0

    def close(self):
        pass

    def _branches(self, nodes, bit):
        split = self.level[nodes] == bit
        left = np.where(split, self.low[nodes], nodes)
        right = np.where(split, self.high[nodes], nodes)
        count = self.counts[left] << (self.level[left] - bit - 1).astype(np.uint64)
        return left, right, count

    def unrank(self, ids):
        ids = np.asarray(ids, dtype=np.uint64).copy()
        if np.any(ids >= self.size):
            raise ValueError("State index out of range")
        nodes = np.full(ids.shape, self.root, dtype=np.int32)
        masks = np.zeros(ids.shape, dtype=np.uint64)
        for bit in range(self.width):
            left, right, count = self._branches(nodes, bit)
            choose = ids >= count
            masks |= choose.astype(np.uint64) << np.uint64(bit)
            ids -= np.where(choose, count, np.uint64(0))
            nodes = np.where(choose, right, left)
        assert np.all(nodes == 1)
        return masks

    def rank(self, masks):
        masks = np.asarray(masks, dtype=np.uint64)
        nodes = np.full(masks.shape, self.root, dtype=np.int32)
        ids = np.zeros(masks.shape, dtype=np.uint64)
        for bit in range(self.width):
            left, right, count = self._branches(nodes, bit)
            choose = ((masks >> np.uint64(bit)) & np.uint64(1)) != 0
            ids += np.where(choose, count, np.uint64(0))
            nodes = np.where(choose, right, left)
        if np.any(nodes != 1):
            raise ValueError("State is outside the reachable graph")
        return ids.astype(np.int64)

    def contains(self, masks, root):
        masks = np.asarray(masks, dtype=np.uint64)
        nodes = np.full(masks.shape, int(root), dtype=np.int32)
        for _ in range(self.width):
            active = nodes >= 2
            if not np.any(active):
                break
            at = self.level[nodes]
            choose = ((masks >> at.astype(np.uint64)) & np.uint64(1)) != 0
            nodes = np.where(active, np.where(choose, self.high[nodes], self.low[nodes]), nodes)
        return nodes == 1

    def distance(self, source, goal):
        source, goal = np.broadcast_arrays(np.asarray(source, dtype=np.uint64), np.asarray(goal, dtype=np.uint64))
        target = np.uint64((1 << self.width) - 1) ^ (source ^ goal)
        valid = (source & goal) == goal
        result = np.full(source.shape, -1, dtype=np.int16)
        remaining = np.flatnonzero(valid)
        for depth, root in enumerate(self.layers):
            if not len(remaining):
                break
            selected = self.contains(target.ravel()[remaining], root)
            result.ravel()[remaining[selected]] = depth
            remaining = remaining[~selected]
        return result

    def degrees(self, ids):
        masks = self.unrank(ids)
        return ((masks[..., None] & self.tiles) == self.tiles).sum(-1)

    def take(self, ids, offsets):
        masks = self.unrank(ids)
        valid = (masks[:, None] & self.tiles) == self.tiles
        choice = (np.cumsum(valid, axis=-1) > offsets[:, None]).argmax(-1)
        return self.rank(masks ^ self.tiles[choice])

    def star(self, ids):
        masks = self.unrank(ids)
        rows, actions = np.where((masks[:, None] & self.tiles) == self.tiles)
        return rows, self.rank(masks[rows] ^ self.tiles[actions])
