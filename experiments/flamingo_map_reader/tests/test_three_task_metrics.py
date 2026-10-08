import unittest
from pathlib import Path

from experiments.flamingo_map_reader.src.trajectory_metrics import termination_counts, summarize_turns
from experiments.flamingo_map_reader.src.blocks_checkpoint_queue import (
    evaluation_modes, evaluation_selection, validate_resume_contract)
from experiments.flamingo_map_reader.src.analyze_blocks_ranking import check_metric


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


if __name__ == '__main__':
    unittest.main()
