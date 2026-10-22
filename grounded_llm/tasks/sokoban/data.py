"""Small exhaustive diagnostic boards, not claimed to be Boxoban benchmark data."""
import numpy as np
from .environment import Sokoban


def build(config, cache):
    rng = np.random.default_rng(config['seed'])
    spec = config['data']
    height, width = spec['shape']
    interior = [(r, c) for r in range(1, height - 1) for c in range(1, width - 1)]
    used = set()
    for split, count in spec['scenes'].items():
        accepted = 0
        for _ in range(spec['max_generation_attempts']):
            if accepted == count: break
            wall_count = int(rng.integers(spec['max_internal_walls'] + 1))
            order = rng.permutation(len(interior))
            walls = {interior[i] for i in order[:wall_count]}
            available = [p for p in interior if p not in walls]
            chosen = rng.choice(len(available), 2 * spec['boxes'] + 1, replace=False)
            goals = {available[i] for i in chosen[:spec['boxes']]}
            # Hold out complete wall geometries, even when target placement differs.
            signature = tuple(sorted(walls))
            if signature in used: continue
            lines = [['#'] * width for _ in range(height)]
            for r, c in available: lines[r][c] = '.' if (r, c) in goals else ' '
            for i in chosen[spec['boxes']:2 * spec['boxes']]:
                r, c = available[i]; lines[r][c] = '$'
            r, c = available[chosen[-1]]; lines[r][c] = '@'
            graph = Sokoban([''.join(row) for row in lines]).graph()
            if len(graph['starts']) < spec['minimum_solvable_states']: continue
            used.add(signature)
            accepted += 1
            graph['metadata']['ascii_seed_board'] = [''.join(row) for row in lines]
            # The seed board is only a layout carrier; evaluation uses exact-solvable states.
            yield split, graph
        if accepted != count:
            raise RuntimeError(f'Only generated {accepted}/{count} {split} scenes')
