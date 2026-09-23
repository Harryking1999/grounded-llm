"""Cell query sampling and exact reconstruction; no trainable auxiliary decoder."""
import random


def query(board, row, col):
    return dict(board, id=f"{board['id']}_cell{row}_{col}", board_id=board['id'],
                task='report_cell', report_row=row, report_col=col)


def all_queries(boards):
    return [query(board, r, c) for board in boards
            for r in range(len(board['rows'])) for c in range(len(board['rows'][r]))]


class BalancedCellQueries:
    """Exactly balance coordinate × bit; redraw boards each epoch from training only."""
    def __init__(self, boards, spec):
        self.boards, self.spec = boards, spec
        self.buckets = {}
        for index, board in enumerate(boards):
            for row, values in enumerate(board['rows']):
                for col, value in enumerate(values):
                    self.buckets.setdefault((row, col, int(value)), []).append(index)
        expected = 2 * len(boards[0]['rows']) * len(boards[0]['rows'][0])
        if len(self.buckets) != expected:
            raise ValueError('Each coordinate needs both occupied and empty training examples')
        self.set_epoch(0)

    def set_epoch(self, epoch):
        rng = random.Random(self.spec['seed'] + epoch)
        self.items = []
        count = self.spec['samples_per_coordinate_label']
        for (row, col, bit), indices in sorted(self.buckets.items()):
            chosen = rng.sample(indices, count) if len(indices) >= count else rng.choices(indices, k=count)
            self.items.extend(query(self.boards[index], row, col) for index in chosen)
        rng.shuffle(self.items)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]


def summarize_cells(records):
    """Invalid/non-bit argmax counts wrong; constrained binary predictions reported separately."""
    boards = {}
    result = dict(total=len(records), correct=0, binary_correct=0, invalid=0,
                  occupied=0, occupied_correct=0, empty=0, empty_correct=0,
                  loss_sum=0.0, binary_loss_sum=0.0)
    for row in records:
        target, predicted = row['target'], row['predicted']
        result['correct'] += predicted == target
        result['binary_correct'] += row['binary_predicted'] == target
        result['invalid'] += predicted not in (0, 1)
        label = 'occupied' if target else 'empty'
        result[label] += 1
        result[label + '_correct'] += predicted == target
        result['loss_sum'] += row['loss']
        result['binary_loss_sum'] += row['binary_loss']
        boards.setdefault(row['board_id'], []).append(row)
    result['boards'] = len(boards)
    complete = [rows for rows in boards.values() if len(rows) == rows[0]['board_cells'] and
                len({(r['row'], r['col']) for r in rows}) == rows[0]['board_cells']]
    result['complete_boards'] = len(complete)
    result['board_exact'] = sum(all(r['predicted'] == r['target'] for r in rows) for rows in complete)
    result['binary_board_exact'] = sum(all(r['binary_predicted'] == r['target'] for r in rows) for rows in complete)
    result['mean_loss'] = result['loss_sum'] / len(records)
    result['binary_mean_loss'] = result['binary_loss_sum'] / len(records)
    result['balanced_accuracy'] = (0.5 * (result['occupied_correct'] / result['occupied'] +
                                        result['empty_correct'] / result['empty'])
                                   if result['occupied'] and result['empty'] else None)
    return result
