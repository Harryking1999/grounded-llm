import unittest
from collections import Counter
from experiments.map_sft_baselines.src.subsample import select_rows


class SubsampleTests(unittest.TestCase):
    def test_preserve_boards_terminal_rate_identity_and_determinism(self):
        rows = [dict(board_row=b, shortest_moves=0 if n == 49 else 6+n % 4,
                     trajectory_id=f'{b}_{n}') for b in range(1000) for n in range(50)]
        selected = select_rows(rows, 20261013)
        self.assertEqual(len(selected), 10000)
        self.assertEqual(len({r['trajectory_id'] for r in selected}), 10000)
        self.assertEqual(set(Counter(r['board_row'] for r in selected).values()), {10})
        self.assertEqual(sum(r['shortest_moves'] == 0 for r in selected), 200)
        self.assertEqual(selected, select_rows(rows, 20261013))
        self.assertNotEqual(selected, select_rows(rows, 20261014))


if __name__ == '__main__':
    unittest.main()
