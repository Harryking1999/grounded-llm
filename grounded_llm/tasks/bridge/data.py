"""Disjoint crossing-time scenarios, including the classic sanity case."""
import numpy as np
from .environment import Bridge


def build(config, cache):
    rng = np.random.default_rng(config['seed'])
    spec = config['data']
    used = set()
    for split, count in spec['scenes'].items():
        for _ in range(count):
            while True:
                times = tuple(sorted(rng.integers(1, spec['max_time'] + 1, size=spec['people']).tolist()))
                if times not in used: break
            used.add(times)
            yield split, Bridge(times).graph()
