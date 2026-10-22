"""Pinned XoT puzzle lists; deduplicate unordered puzzles before partitioning."""
import csv
import io
from pathlib import Path
import urllib.request
import numpy as np
from .environment import Game24


def build(config, cache):
    spec = config['data']
    cache = Path(cache)
    cache.mkdir(parents=True, exist_ok=True)
    pools = {}
    for split in ('train', 'test'):
        path = cache / f'xot_{spec["source_commit"]}_{split}.csv'
        if not path.exists():
            url = f'https://raw.githubusercontent.com/microsoft/Everything-of-Thoughts-XoT/{spec["source_commit"]}/xot_mcts/game24/data/{split}.csv'
            path.write_bytes(urllib.request.urlopen(url, timeout=60).read())
        pools[split] = list(dict.fromkeys(tuple(sorted(map(int, row['Puzzles'].split())))
                                        for row in csv.DictReader(io.StringIO(path.read_text()))))
    test_set = set(pools['test'])
    training = [p for p in pools['train'] if p not in test_set]
    rng = np.random.default_rng(config['seed'])
    rng.shuffle(training)
    ntrain, nval = spec['scenes']['train'], spec['scenes']['validation']
    selected = {'train': training[:ntrain], 'validation': training[ntrain:ntrain + nval],
                'test': pools['test'][:spec['scenes']['test']]}
    env = Game24()
    for split, puzzles in selected.items():
        if len(puzzles) != spec['scenes'][split]: raise ValueError('Insufficient unique puzzles')
        for numbers in puzzles:
            yield split, env.graph(numbers)
