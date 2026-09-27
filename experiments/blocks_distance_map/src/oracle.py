"""Exact directed removal distance under the existing official GCML rules."""
import time

from experiments.gcml_counterexamples.src import blocks


class OracleBudgetExceeded(RuntimeError):
    """An incomplete search is unknown, never an unreachable certificate."""


class DistanceOracle:
    def __init__(self, placements=None, cache_limit=500000, seconds=600):
        self.tiles = tuple(mask for mask, _ in (placements or blocks.PLACEMENTS))
        self.by_cell = {}
        for tile in self.tiles:
            for cell in blocks.cells(tile):
                self.by_cell.setdefault(cell, []).append(tile)
        self.cache = {0: 0}
        self.cache_limit = cache_limit
        self.deadline = time.monotonic() + seconds
        self.largest_tile = max(t.bit_count() for t in self.tiles)

    def cover(self, mask):
        if mask in self.cache:
            return self.cache[mask]
        if len(self.cache) >= self.cache_limit or time.monotonic() > self.deadline:
            raise OracleBudgetExceeded("Exact distance remains unknown: oracle budget exhausted")
        choices = None
        for cell in blocks.cells(mask):
            options = [t for t in self.by_cell.get(cell, ()) if t & mask == t]
            if not options:
                self.cache[mask] = -1
                return -1
            if choices is None or len(options) < len(choices):
                choices = options
        best = -1
        for tile in sorted(choices, key=int.bit_count, reverse=True):
            rest = self.cover(mask ^ tile)
            if rest >= 0 and (best < 0 or rest + 1 < best):
                best = rest + 1
            if best == (mask.bit_count() + self.largest_tile - 1) // self.largest_tile:
                break
        self.cache[mask] = best
        return best

    def distance(self, source, target):
        return -1 if target & source != target else self.cover(source ^ target)


def successors(state):
    return [(i, state ^ tile) for i, (tile, _) in enumerate(blocks.PLACEMENTS)
            if tile & state == tile]
