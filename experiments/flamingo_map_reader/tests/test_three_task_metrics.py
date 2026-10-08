import unittest
from pathlib import Path

from experiments.flamingo_map_reader.src.trajectory_metrics import termination_counts, summarize_turns
from experiments.flamingo_map_reader.src.blocks_checkpoint_queue import (
    evaluation_modes, evaluation_selection, validate_resume_contract)
from experiments.flamingo_map_reader.src.analyze_blocks_ranking import check_metric
from experiments.flamingo_map_reader.src.summarize_blocks_results import (
    add_readout_scores, summarize_cases, overall_completion)


class ThreeTaskMetricsTest(unittest.TestCase):
    def test_done_precision_recall_and_correct_failure_have_distinct_denominators(self):
        rows = [dict(done=True, candidates=0, answer='<done/>'),
                dict(done=False, candidates=0, answer='<action>none</action><done/>'),
                dict(done=False, candidates=3, answer='<done/>'),
                dict(done=True, candidates=0, answer='<action>1</action>'),
                dict(done=True, candidates=0, answer='<done/><done/>')]
        counts = termination_counts(rows)
        self.assertEqual(counts['done_outputs'], 4)
        self.assertEqual(counts['done_at_goal'], 2)
        self.assertEqual(counts['goal_claims'], 2)
        self.assertEqual(counts['correct_goal_claims'], 1)
        self.assertEqual(counts['goal_states'], 3)
        self.assertEqual(counts['correct_no_solution'], 1)
        value = summarize_turns(rows)
        self.assertEqual(value['done_goal_precision'], .5)
        self.assertEqual(value['goal_stop_recall'], 1/3)

    def test_live_format_failure_is_an_action_error_even_without_score_field(self):
        rows = [dict(done=False, candidates=3, remaining_shortest=2, answer='bad'),
                dict(done=False, candidates=2, remaining_shortest=1, answer='<action>1</action>',
                     action_keeps_goal_reachable=True),
                dict(done=False, candidates=2, remaining_shortest=-1, answer='<action>1</action>')]
        value = summarize_turns(rows)
        self.assertEqual(value['solvable_decisions'], 2)
        self.assertEqual(value['action_keeps_goal_reachable_rate'], .5)

    def test_half_epoch_coverage_includes_both_terminal_tasks_without_training_records(self):
        records = [dict(split='test')]*510 + [dict(split='test_no_solution')]*100
        records += [dict(split='validation', start='1', goal='1')]*100
        records += [dict(split='validation', start='2', goal='1')]*1000
        records += [dict(split='train', start='1', goal='1')]*1000
        modes = evaluation_modes(dict(path=Path('checkpoint-6875')), dict(records=records), True)
        self.assertEqual(modes, [('test', 'rollout', 510), ('test', 'reference', 510),
                                 ('test_no_solution', 'reference', 100), ('initial_goal', 'reference', 100)])
        self.assertEqual(evaluation_selection('initial_goal'),
                         ['--split', 'validation', '--task-type', 'initial_goal'])
        validate_resume_contract(dict(run='same'), dict(run='same', terminal_tasks=True))

    def test_existing_ndcg_ties_cutoff_and_malformed_checks(self):
        check_metric()

    def test_readout_and_choice_consistency_do_not_require_environment_reachability(self):
        row = dict(done=False, candidates=3, remaining_shortest=-1,
                   answer='Map-distance ranking to the goal, closest to farthest: 2 < current < 1 < 3.\n<action>2</action>')
        add_readout_scores(row, [0., 2., 4.], dict(k=10, gain='rank_grade', discount='log2'))
        self.assertTrue(row['ranking_parseable'])
        self.assertTrue(row['follows_own_top'])
        self.assertEqual(row['topk_recall'], 1)
        self.assertLess(row['ranking_ndcg_at_10'], 1)
        row['turn'] = 0
        values = summarize_cases([dict(trajectory_id='a', group='ordinary', turns=[row])])
        self.assertEqual(values['overall']['map_decisions'], 1)
        self.assertEqual(values['overall']['solvable'], 0)
        self.assertEqual(values['overall']['ndcg_n'], 1)
        self.assertEqual(values['overall']['ndcg_mean'], row['ranking_ndcg_at_10'])

    def test_prefix_survival_charges_first_error_once(self):
        good = dict(turn=0, done=False, candidates=2, remaining_shortest=2,
                    legal_action=True, action_keeps_goal_reachable=True)
        bad = dict(good, turn=1, action_keeps_goal_reachable=False)
        dead = dict(bad, turn=2, remaining_shortest=-1)
        cases = [dict(group='a', turns=[good,bad,dead]), dict(group='b', turns=[good])]
        values = summarize_cases(cases)
        self.assertEqual(values['first_loss_by_round'], {2:1})
        self.assertEqual(values['safe_prefix_survival'], {'1':2,'2':1,'3':1})

    def test_overall_uses_task_weights_and_waits_for_complete_groups(self):
        ordinary = dict(cases=510, success=51)
        initial = dict(cases=100, overall=dict(correct_goal_claims=100))
        failure = dict(cases=100, overall=dict(correct_no_solution=50))
        weights = dict(ordinary=.8, initial_goal=.1, failure_context=.1)
        expected = dict(ordinary=510, initial_goal=100, failure_context=100)
        self.assertAlmostEqual(overall_completion(ordinary,initial,failure,weights,expected)['weighted_rate'], .23)
        self.assertIsNone(overall_completion(ordinary,dict(initial,cases=99),failure,weights,expected))


if __name__ == '__main__':
    unittest.main()
