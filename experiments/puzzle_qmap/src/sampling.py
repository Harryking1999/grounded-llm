"""Coverage passes plus exact-distance balance and shared-goal contexts."""
import numpy as np


class CoverageSampler:
    def __init__(self, labels, rng, broad_count, broad_fraction, context_start=None, context_fraction=0):
        self.rng, self.broad_count = rng, int(broad_count)
        self.broad_fraction, self.context_fraction = broad_fraction, context_fraction
        self.context_start, self.count = context_start, len(labels)
        if not 0 < broad_fraction <= 1 or not 0 <= context_fraction <= 1 - broad_fraction:
            raise ValueError("Invalid sampling fractions")
        self.order = rng.permutation(self.broad_count)
        self.position = self.completed = 0
        self.groups = [np.flatnonzero(labels == d) for d in np.unique(labels) if d > 0]

    @property
    def passes(self):
        return self.completed + self.position / self.broad_count

    def draw(self, count):
        broad = round(count * self.broad_fraction)
        pieces, remaining = [], broad
        while remaining:
            n = min(remaining, self.broad_count - self.position)
            pieces.append(self.order[self.position:self.position + n])
            self.position += n
            remaining -= n
            if self.position == self.broad_count:
                self.completed += 1
                self.order = self.rng.permutation(self.broad_count)
                self.position = 0
        context = round(count * self.context_fraction) if self.context_start is not None else 0
        if context:
            pieces.append(self.rng.integers(self.context_start, self.count, size=context))
        balanced = count - broad - context
        kinds = self.rng.integers(len(self.groups), size=balanced)
        for i, group in enumerate(self.groups):
            n = int((kinds == i).sum())
            if n:
                pieces.append(self.rng.choice(group, n))
        result = np.concatenate(pieces)
        self.rng.shuffle(result)
        return result
