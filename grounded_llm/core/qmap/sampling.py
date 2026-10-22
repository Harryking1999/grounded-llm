"""Coverage passes plus exact-distance balance and shared-goal contexts."""
import numpy as np
import torch


def task_indices(env, data, split, count, config):
    """Deterministic distance-stratified evaluation selection; no learning stack."""
    tasks = data[f"tasks_{split}"]
    count = min(count, len(tasks))
    if not config["evaluation"].get("stratified_tasks", False) or split == "reference_test":
        return np.arange(count)
    labels = data.get(f"parent_distances_{split}")
    if labels is None:
        labels = env.distance(tasks[:, 0], tasks[:, 1])
    rng = np.random.default_rng(config["seed"] + (101 if split == "validation" else 102) + count)
    levels = np.unique(labels)
    selected = []
    for level in levels:
        pool = np.flatnonzero(labels == level)
        selected.extend(rng.choice(pool, min(len(pool), count // len(levels)), replace=False).tolist())
    remaining = np.setdiff1d(np.arange(len(tasks)), selected)
    if len(selected) < count:
        selected.extend(rng.choice(remaining, count - len(selected), replace=False).tolist())
    result = np.asarray(selected)
    rng.shuffle(result)
    return result


class StratifiedEpochSampler:
    """Equal layer quotas, with an independent shuffled epoch in every layer."""
    def __init__(self, labels, rng):
        labels = np.asarray(labels)
        if labels.ndim != 1 or not len(labels) or np.any(labels < 0):
            raise ValueError("Expected nonempty nonnegative distance labels")
        self.rng = rng
        self.levels = np.unique(labels)
        self.orders = [rng.permutation(np.flatnonzero(labels == level)) for level in self.levels]
        self.positions = np.zeros(len(self.levels), dtype=int)
        self.drawn = np.zeros(len(self.levels), dtype=np.int64)
        self.offset = 0

    @property
    def passes(self):
        return min(self.drawn[i] / len(order) for i, order in enumerate(self.orders))

    def coverage(self):
        return {str(level): {"rows": len(order), "draws": int(self.drawn[i]),
                             "passes": float(self.drawn[i] / len(order))}
                for i, (level, order) in enumerate(zip(self.levels, self.orders))}

    def steps_for_passes(self, batch_size, passes):
        # Rotating remainder quotas make cumulative layer counts differ by <= 1.
        required = max(int(np.ceil(passes * len(order))) for order in self.orders)
        return int(np.ceil(len(self.levels) * required / batch_size))

    def draw(self, count):
        if count <= 0:
            raise ValueError("Batch size must be positive")
        quotas = np.full(len(self.levels), count // len(self.levels))
        remainder = count % len(self.levels)
        quotas[(self.offset + np.arange(remainder)) % len(self.levels)] += 1
        self.offset = (self.offset + remainder) % len(self.levels)
        pieces = []
        for i, quota in enumerate(quotas):
            remaining = int(quota)
            while remaining:
                order, start = self.orders[i], self.positions[i]
                n = min(remaining, len(order) - start)
                pieces.append(order[start:start + n])
                self.positions[i] += n
                self.drawn[i] += n
                remaining -= n
                if self.positions[i] == len(order):
                    self.orders[i] = self.rng.permutation(order)
                    self.positions[i] = 0
        result = np.concatenate(pieces)
        self.rng.shuffle(result)
        return result

    def state_dict(self):
        return {key: value for key, value in self.__dict__.items() if key != 'rng'}

    def load_state_dict(self, state):
        if not np.array_equal(state['levels'], self.levels):
            raise ValueError('Sampler distance layers changed during resume')
        self.__dict__.update(state)


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

    def state_dict(self):
        return {key: value for key, value in self.__dict__.items() if key != 'rng'}

    def load_state_dict(self, state):
        for key in ('count','broad_count','broad_fraction','context_fraction','context_start'):
            if state[key] != getattr(self,key):
                raise ValueError('Coverage sampler configuration changed during resume')
        self.__dict__.update(state)


def draw(groups, size, device):
    parts, remaining = [], size
    for i, (ids, weight) in enumerate(groups):
        count = remaining if i == len(groups) - 1 else round(size * weight)
        remaining -= count
        parts.append(ids[torch.randint(len(ids), (count,), device=device)])
    return torch.cat(parts)
