"""Diverse pixel states for report learning, separate from solvable planning boards."""
from collections import Counter
import random


def report_training_items(boards, spec):
    """Expand the same boards into a matched number of full/row reports."""
    rng = random.Random(spec['seed'])
    items = []
    for board in boards:
        for repeat in range(spec['full_board_repeats']):
            items.append(dict(board, id=f"{board['id']}_full{repeat}"))
        for row in rng.sample(range(len(board['rows'])), spec['rows_per_board']):
            items.append(dict(board, id=f"{board['id']}_row{row}", task='report_row', report_row=row))
    return items


def report_question(config, item):
    if item['task'] == 'report_cell':
        row, col = item['report_row'], item['report_col']
        if type(row) is not int or type(col) is not int or not (0 <= row < len(item['rows']) and 0 <= col < len(item['rows'][row])):
            raise ValueError('Cell coordinate is outside the board')
        return config['blocks']['cell_report_prompt'].format(row=row, col=col)
    if item['task'] == 'report_row':
        row = item['report_row']
        if type(row) is not int or not 0 <= row < len(item['rows']):
            raise ValueError('Report row is outside the board')
        return config['blocks']['row_report_prompt'].format(row=row)
    if item['task'] != 'report_board':
        raise ValueError('Unknown readout task')
    return config['blocks']['report_prompt']


def report_target(item):
    import json
    if item['task'] == 'report_cell':
        return item['rows'][item['report_row']][item['report_col']]
    target = item['rows'][item['report_row']] if item['task'] == 'report_row' else item['rows']
    return json.dumps(target, separators=(',', ':'))


def spatial_probe_items(boards, spec):
    """Balanced category sample, paired row queries differing in exactly one pixel."""
    rng = random.Random(spec['seed'])
    groups = {}
    for item in boards:
        groups.setdefault(item['category'], []).append(item)
    for values in groups.values():
        rng.shuffle(values)
    selected = []
    while len(selected) < min(spec['boards'], len(boards)):
        for category in sorted(groups):
            if groups[category] and len(selected) < spec['boards']:
                selected.append(groups[category].pop())
    pairs = []
    for item in selected:
        row, col = rng.randrange(len(item['rows'])), rng.randrange(len(item['rows'][0]))
        original = dict(item, task='report_row', report_row=row, probe_pair=item['id'], variant='original')
        rows = list(item['rows'])
        cells = list(rows[row])
        cells[col] = '1' if cells[col] == '0' else '0'
        rows[row] = ''.join(cells)
        changed = dict(original, id=item['id'] + '_flip', rows=rows, variant='one_pixel_flip', flipped_col=col)
        pairs.extend([original, changed])
    return pairs


def summarize_spatial_probe(records):
    """Count exact rows and localized responses to saved single-pixel interventions.

    Invalid outputs receive no credit. Localized change does not imply the other
    pixels are correct; full row/pair accuracy remains a separate metric.
    """
    import json
    groups = {}
    summary = dict(total=len(records), rows_correct=0, valid_rows=0,
                   cell_matches=0, total_cells=0, pairs=0, pairs_valid=0,
                   pairs_both_correct=0, flip_bit_both_correct=0,
                   localized_correct_change=0, unchanged_cells_equal=0,
                   unchanged_cells_total=0)
    for record in records:
        target = json.loads(record['target'])
        try:
            output = json.loads(record['raw_output'])
        except ValueError:
            output = None
        valid = isinstance(output, str) and len(output) == len(target) and set(output) <= {'0', '1'}
        if not valid:
            output = None
        summary['valid_rows'] += valid
        summary['rows_correct'] += output == target
        summary['total_cells'] += len(target)
        summary['cell_matches'] += sum(a == b for a, b in zip(output, target)) if valid else 0
        groups.setdefault(record['pair'], {})[record['variant']] = (target, output)
    for pair in groups.values():
        before, pred_before = pair['original']
        after, pred_after = pair['one_pixel_flip']
        changed = [i for i, (a, b) in enumerate(zip(before, after)) if a != b]
        if len(before) != len(after) or len(changed) != 1:
            raise ValueError('Spatial probe targets must differ at exactly one cell')
        index = changed[0]
        summary['pairs'] += 1
        summary['unchanged_cells_total'] += len(before) - 1
        summary['pairs_both_correct'] += pred_before == before and pred_after == after
        if pred_before is None or pred_after is None:
            continue
        summary['pairs_valid'] += 1
        flip_correct = pred_before[index] == before[index] and pred_after[index] == after[index]
        same = sum(pred_before[i] == pred_after[i] for i in range(len(before)) if i != index)
        summary['flip_bit_both_correct'] += flip_correct
        summary['unchanged_cells_equal'] += same
        summary['localized_correct_change'] += flip_correct and same == len(before) - 1
    return summary


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
