"""Diverse pixel states for report learning, separate from solvable planning boards."""
from collections import Counter
import random


def state_key(task, mask):
    return task.normalized_key(mask) if mask else ('empty',)


def describe(task, mask):
    occupied = [i for i in range(task.size ** 2) if mask >> i & 1]
    isolated = 0
    for index in occupied:
        r, c = divmod(index, task.size)
        neighbors = [(rr, cc) for rr, cc in ((r-1, c), (r+1, c), (r, c-1), (r, c+1))
                     if 0 <= rr < task.size and 0 <= cc < task.size]
        isolated += not any(mask >> (rr * task.size + cc) & 1 for rr, cc in neighbors)
    return dict(occupied_cells=len(occupied), isolated_cells=isolated)


def build_readout_dataset(task, planning, spec):
    """All derivatives stay with their family; unrelated states cannot cross splits.

    Empty/full are explicitly training anchors, never counted as unseen validation.
    Arbitrary grids may be untileable: only report labels, no action supervision.
    """
    rng = random.Random(spec['seed'])
    owners = {}
    for split, cases in planning.items():
        for case in cases:
            for rows in case['report_grids']:
                owners[state_key(task, task.from_grid(rows))] = split
    result = {'train': [], 'validation': []}
    seen = {s: set() for s in result}
    n = task.size ** 2

    def add(split, mask, family, category):
        key = state_key(task, mask)
        if mask in seen[split] or owners.get(key, split) != split:
            return False
        owners[key] = split
        seen[split].add(mask)
        result[split].append(dict(id=f'{split}_readout_{len(result[split]):05}',
                    task='report_board', rows=task.to_rows(mask), family=family,
                    category=category, **describe(task, mask)))
        return True

    for split in result:
        for case in planning[split]:
            for i, rows in enumerate(case['report_grids']):
                add(split, task.from_grid(rows), case['id'],
                    'constructed' if i == 0 else 'legal_intermediate')
    for mask in (0, (1 << n) - 1):
        add('train', mask, 'boundary_anchor', 'empty' if not mask else 'full')

    for split in result:
        quotas = spec['per_split'][split]
        categories = [(f'coverage_{low}_{high}', quotas['per_coverage_bin'], (low, high))
                      for low, high in spec['coverage_bins']]
        categories += [(c, quotas[c], None) for c in ('isolated', 'patch_holes', 'damaged_intermediate')]
        for category, quota, limits in categories:
            accepted = 0
            for attempt in range(max(10000, quota * 100)):
                if accepted == quota:
                    break
                family = f'{split}_{category}_{attempt}'
                if limits:
                    cells = rng.sample(range(n), rng.randint(*limits))
                    mask = sum(1 << i for i in cells)
                elif category == 'isolated':
                    parity = rng.randrange(2)
                    cells = [i for i in range(n) if sum(divmod(i, task.size)) % 2 == parity]
                    mask = sum(1 << i for i in rng.sample(cells, rng.randint(2, len(cells))))
                elif category == 'patch_holes':
                    mask = 0
                    for _ in range(rng.randint(1, 4)):
                        r, c = rng.randrange(task.size), rng.randrange(task.size)
                        height, width = rng.randint(1, task.size-r), rng.randint(1, task.size-c)
                        mask |= sum(1 << (rr * task.size + cc) for rr in range(r, r+height) for cc in range(c, c+width))
                    cells = [i for i in range(n) if mask >> i & 1]
                    if len(cells) < 3:
                        continue
                    for i in rng.sample(cells, rng.randint(1, len(cells)-1)):
                        mask ^= 1 << i
                else:
                    case = rng.choice(planning[split])
                    family = case['id']
                    original = task.from_grid(rng.choice(case['report_grids']))
                    cells = [i for i in range(n) if original >> i & 1]
                    mask = sum(1 << i for i in rng.sample(cells, rng.randint(1, len(cells)-1)))
                if add(split, mask, family, category):
                    accepted += 1
            if accepted != quota:
                raise ValueError(f'Cannot fill non-leaking readout quota: {split}/{category}')
    return result


def coverage_summary(dataset):
    return {split: dict(examples=len(items), categories=dict(Counter(x['category'] for x in items)),
                  occupied_cells=sum(x['occupied_cells'] for x in items),
                  total_cells=sum(len(''.join(x['rows'])) for x in items),
                  states_with_isolated_cells=sum(x['isolated_cells'] > 0 for x in items))
            for split, items in dataset.items()}
