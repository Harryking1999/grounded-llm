import json
from pathlib import Path
import unittest

from experiments.map_sft_baselines.src.contract import training_steps,checkpoint_steps
from experiments.map_sft_baselines.src.train import learning_rate_factor
from experiments.map_sft_baselines.src.extend_data import select_additional
from experiments.map_sft_baselines.src.leakage_audit import audit_exposure


class ExpansionAuditTests(unittest.TestCase):
    def test_100k_has_exact_original_lr_curve_and_required_checkpoints(self):
        root=Path(__file__).parents[1]/'configs'
        new=json.loads((root/'b2_100k.json').read_text());old=json.loads((root/'study.json').read_text())
        self.assertEqual(training_steps(new),250000)
        self.assertEqual(new['training']['epochs'],10)
        self.assertIn(50000,checkpoint_steps(new));self.assertIn(250000,checkpoint_steps(new))
        for step in (0,1,7500,12500,50000,150000,250000):
            self.assertEqual(learning_rate_factor(step,250000,new['training']),learning_rate_factor(step,250000,old['training']))

    def test_extra_routes_follow_design_length_before_coverage(self):
        import random
        def row(state,length):return dict(start=str(state),goal='0',shortest_moves=length,pairs=[(state,0)])
        old=[row(1,6),row(2,7)]
        pool=[row(1,6),row(3,6),row(4,7),row(5,8)]
        selected=select_additional(pool,old,random.Random(0))
        self.assertEqual({r['start'] for r in selected},{'5','4'})
        self.assertIsNone(select_additional([row(3,6)],old,random.Random(0)))

    def test_audit_distinguishes_direct_pair_candidate_slot_and_board_overlap(self):
        from experiments.gcml_counterexamples.src import blocks as rules
        from experiments.llm_task_sft.src.protocol import action_key
        tile,action=rules.PLACEMENTS[0]
        other,other_action=next((mask,a) for mask,a in rules.PLACEMENTS if not(mask&tile))
        train=[dict(start=str(tile|other),goal='0',pairs=[(tile|other,0)],route_ids=[1],board_row=1)]
        # A one-step held-out task appears as an unchosen successor of training,
        # without its own current-state action label having been supervised.
        reference=dict(start=str(tile),goal='0',reference_actions=[action_key(action)],
                       trajectory_id='heldout',board_row=1,group='same_board')
        result=audit_exposure(train,[reference])['groups']['same_board']
        self.assertEqual(result['source_board_seen'],1)
        self.assertEqual(result['supervised_pair_matches'],0)
        self.assertEqual(result['candidate_slot_pair_matches'],1)
        train.append(dict(start=str(tile),goal='0',pairs=[(tile,0)],route_ids=[1],board_row=2))
        result=audit_exposure(train,[reference])['groups']['same_board']
        self.assertEqual(result['initial_pair_in_training_initial'],1)
        self.assertEqual(result['exact_physical_route_seen'],1)


if __name__=='__main__':unittest.main()
