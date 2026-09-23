import unittest
from collections import Counter

from grounded_llm.blocks_cell import BalancedCellQueries, all_queries, summarize_cells
from grounded_llm.blocks_readout import report_question, report_target


class CellTests(unittest.TestCase):
    def test_sampler_balances_every_coordinate_and_bit_without_changing_boards(self):
        boards = [dict(id=str(i), category='test', rows=rows)
                  for i, rows in enumerate([['00', '11'], ['11', '00'], ['10', '10'], ['01', '01']])]
        spec = dict(seed=31, samples_per_coordinate_label=3)
        pool = BalancedCellQueries(boards, spec)
        pool.set_epoch(7)
        self.assertEqual(len(pool), 24)
        counts = Counter((x['report_row'], x['report_col'], report_target(x)) for x in pool)
        self.assertEqual(set(counts.values()), {3})
        other = BalancedCellQueries(boards, spec)
        other.set_epoch(7)
        self.assertEqual(pool.items, other.items)
        for item in pool:
            self.assertEqual(item['rows'], boards[int(item['board_id'])]['rows'])
        config = {'blocks': {'cell_report_prompt': 'row {row} col {col}'}}
        item = all_queries(boards[:1])[0]
        self.assertEqual(report_question(config, item), 'row 0 col 0')
        self.assertEqual(report_target(item), '0')

    def test_reconstruction_requires_all_distinct_cells_and_counts_invalid_answers(self):
        records = [dict(board_id='a', category='test', row=i // 2, col=i % 2, board_cells=4,
                        target=i % 2, predicted=i % 2, binary_predicted=i % 2, loss=0.1, binary_loss=0.05)
                   for i in range(4)]
        self.assertEqual(summarize_cells(records)['board_exact'], 1)
        self.assertEqual(summarize_cells(records[:2])['complete_boards'], 0)
        self.assertEqual(summarize_cells(records[:3] + records[:1])['complete_boards'], 0)
        records[1]['predicted'] = None
        result = summarize_cells(records)
        self.assertEqual(result['board_exact'], 0)
        self.assertEqual(result['binary_board_exact'], 1)
        self.assertEqual(result['invalid'], 1)
        self.assertEqual(result['occupied_correct'], 1)
        self.assertEqual(result['empty_correct'], 2)


if __name__ == '__main__':
    unittest.main()
