import unittest
from experiments.cml_map_scaling.src.step2_action_assessment import assessment_groups, block_summary


class AssessmentTest(unittest.TestCase):
    def test_uses_exact_seen_labels_and_existing_test_pairs(self):
        seen=[dict(u=1,g=3,task='action',template='canonical',target=2)]
        split=dict(train=[[1,3]],test=[[0,3]])
        spec=dict(templates=['canonical','heldout'],action_decoding=['free','node_id_grammar'],report_decoding=['free'])
        groups=assessment_groups(spec,split,dict(actions=seen))
        self.assertIs(groups['seen_actions__canonical']['items'],seen)
        self.assertEqual(groups['seen_actions__canonical']['items'][0]['target'],2)
        self.assertEqual([(x['u'],x['g']) for x in groups['test_actions__heldout']['items']],[(0,3),(3,0)])
        self.assertEqual(len(groups['test_reports__canonical']['items']),4)
        with self.assertRaises(AssertionError):assessment_groups(spec,dict(train=[],test=split['test']),dict(actions=seen))

    def test_success_accepts_other_correct_successor_but_label_agreement_is_separate(self):
        row=dict(u=1,g=3,task='action',correct=True,action_legal=True,format_valid=True,
                 node_id_valid=True,graph_distance=2,hit_token_limit=False,training_label_match=False)
        summary=block_summary([row])
        self.assertEqual(summary['metrics']['one_step_success_rate']['successes'],1)
        self.assertEqual(summary['stored_training_label_agreement']['successes'],0)


if __name__=='__main__':unittest.main()
