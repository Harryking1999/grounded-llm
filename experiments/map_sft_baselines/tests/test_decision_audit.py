import unittest
from experiments.map_sft_baselines.src.decision_audit import select_records, first_rank_candidate, summarize, reference_candidate, donor_records


class DecisionAuditTests(unittest.TestCase):
    def test_donors_include_intermediate_states_and_exclude_terminal(self):
        rows=[dict(trajectory_id='a',pairs=[['15','1'],['7','1'],['1','1']])]
        donors=donor_records(rows)
        self.assertEqual([x['start'] for x in donors],['15','7'])
        self.assertEqual(donors[1]['trajectory_id'],'a:turn-1')

    def test_reference_action_survives_json_list_conversion(self):
        self.assertEqual(reference_candidate([(1, 2, 3), (4, 5, 6)], [4, 5, 6]), 2)
        with self.assertRaises(ValueError):
            reference_candidate([(1, 2, 3)], [4, 5, 6])

    def test_fixed_nonterminal_distinct_board_cohort(self):
        rows = [dict(start=i+1, goal=0, board_row=i//2, trajectory_id=str(i)) for i in range(20)]
        rows.append(dict(start=0, goal=0, board_row=99, trajectory_id='done'))
        selected = select_records(rows, 8, 17, True)
        self.assertEqual(selected, select_records(rows, 8, 17, True))
        self.assertEqual(len({r['board_row'] for r in selected}), 8)
        self.assertNotIn('done', [r['trajectory_id'] for r in selected])

    def test_parse_candidate_not_reason_coordinate_or_single_digit(self):
        self.assertEqual(first_rank_candidate('Map-distance ranking to the goal, closest to farthest: current < 49 < 2.\nReason: (1,2)'), 49)
        self.assertIsNone(first_rank_candidate('Reason: Clearing (1,2)\n<action>3</action>'))

    def test_errors_and_duplicates_stay_in_denominator(self):
        a=dict(split='test',condition='natural',answer='same',legal=True,optimal=True,
               reference_action_match=True,rank_map_optimal=True)
        b=dict(a,legal=False,optimal=False,reference_action_match=False,rank_map_optimal=False)
        result=summarize([a,b])['test/natural']
        self.assertEqual(result['optimal'],0.5)
        self.assertEqual(result['duplicate_fraction'],0.5)
        self.assertEqual(result['dominant_answer_fraction'],1)


if __name__ == '__main__':
    unittest.main()
