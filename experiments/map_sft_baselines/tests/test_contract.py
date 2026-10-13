import unittest
import random
import tempfile,json
from pathlib import Path
import torch
from torch import nn
from types import SimpleNamespace
from experiments.map_sft_baselines.src.prepare import choose
from experiments.map_sft_baselines.src.modeling import TokenLM
from experiments.map_sft_baselines.src.queue import evaluation_jobs,collect
from experiments.map_sft_baselines.src.metrics import summarize
from experiments.map_sft_baselines.src.train import validate_resume_contract
from experiments.map_sft_baselines.src.contract import steps_per_epoch,checkpoint_steps,plateau_status

class ContractTests(unittest.TestCase):
    def test_ranking_first_preserves_reason_action_and_terminal(self):
        from experiments.map_sft_baselines.src.common import answer,prompt
        from experiments.gcml_counterexamples.src import blocks as rules
        from experiments.llm_task_sft.src.protocol import legal_actions
        state=rules.PLACEMENTS[0][0]
        step=SimpleNamespace(current=state,goal=0,current_map_distance=2.,
            candidate_map_distances=tuple(float(i) for i in range(len(legal_actions(state)))))
        old=answer(step,1,True).splitlines();new=answer(step,1,True,'ranking_first').splitlines()
        self.assertEqual(new,[old[1],old[0],old[2]])
        self.assertIn('First output the map-distance ranking.',prompt(state,0,[],True,state,True,'ranking_first'))
        terminal=SimpleNamespace(current=0,goal=0)
        self.assertEqual(answer(terminal,None,True),answer(terminal,None,True,'ranking_first'))

    def test_retired_runs_are_not_dispatched_but_replacements_are(self):
        cfg=json.loads((Path(__file__).parents[1]/'configs/study.json').read_text())
        records=[dict(step=6250,epoch=.5,baseline=b) for b in ['B1','B2','B1_sortfirst','B2_sortfirst']]
        with tempfile.TemporaryDirectory() as folder:
            jobs=evaluation_jobs(folder,cfg,records)
        self.assertEqual({j[2] for j in jobs},{'B1_sortfirst','B2_sortfirst'})
    def test_longest_before_coverage_then_diversity(self):
        rows=[dict(start='1',goal='0',shortest_moves=9,pairs=[(1,0)]),
              dict(start='2',goal='0',shortest_moves=8,pairs=[(2,0),(3,0)]),
              dict(start='4',goal='0',shortest_moves=8,pairs=[(1,0)])]
        result=choose(rows,2,set(),random.Random(0))
        self.assertEqual([r['start'] for r in result],['1','2'])
    def test_token_mean_matches_direct_ce_with_unequal_lengths(self):
        class Base(nn.Module):
            def __init__(self):
                super().__init__();self.emb=nn.Embedding(9,4);self.lm_head=nn.Linear(4,9)
            def model(self,input_ids,**kwargs):return SimpleNamespace(last_hidden_state=self.emb(input_ids))
        torch.manual_seed(1);base=Base();model=TokenLM(base,2)
        ids=torch.tensor([[1,2,3,4],[2,4,5,6]]);labels=torch.tensor([[-100,2,3,4],[-100,-100,-100,6]])
        expected=nn.functional.cross_entropy(base.lm_head(base.emb(ids)[:,:-1]).reshape(-1,9),labels[:,1:].reshape(-1),ignore_index=-100)
        actual=model(ids,torch.ones_like(ids),labels)
        self.assertTrue(torch.allclose(actual,expected));actual.backward()
        self.assertTrue(torch.isfinite(base.emb.weight.grad).all())
    def test_latest_first_keeps_old_checkpoints(self):
        with tempfile.TemporaryDirectory() as path:
            for step in [6250,12500]:
                p=Path(path)/'models/B0'/f'checkpoint-{step}';p.mkdir(parents=True)
                (p/'evaluation_ready.json').write_text(json.dumps(dict(step=step,epoch=step/12500,baseline='B0')))
            cfg=dict(baselines={'B0':{'real_map':False}},evaluation={'shard_tasks':510,'tasks':510,'modes':['TF','RO']})
            jobs=evaluation_jobs(path,cfg)
            self.assertEqual([j[0] for j in jobs],[12500,12500,6250,6250])
    def test_dead_state_excluded_from_safe_not_map(self):
        cases=[dict(arrived=False,arrived_and_stopped=False,turns=[
            dict(live=True,rank_eligible=True,safe=True,optimal=False,map_optimal=False),
            dict(live=False,rank_eligible=True,safe=False,optimal=False,map_optimal=True)])]
        result=summarize(cases)
        self.assertEqual(result['safe']['denominator'],1)
        self.assertEqual(result['map_optimal']['denominator'],2)
    def test_stop_confirmation_does_not_change_training_recipe(self):
        cfg=json.loads((Path(__file__).parents[1]/'configs/study.json').read_text())
        saved=json.loads(json.dumps(cfg));saved['early_stop']['mode']='flag_only_pending_user_choice'
        validate_resume_contract(dict(config=saved,baseline='B0'),cfg,'B0')
        changed=json.loads(json.dumps(cfg));changed['training']['learning_rate']*=2
        with self.assertRaises(ValueError):validate_resume_contract(dict(config=saved,baseline='B0'),changed,'B0')
    def test_configured_budgets(self):
        cfg=dict(data={'samples_per_epoch':24},training={'global_batch':6,'epochs':3,'save_steps':2})
        self.assertEqual(steps_per_epoch(cfg),4)
        self.assertEqual(list(checkpoint_steps(cfg)),[2,4,6,8,10,12])
    def test_plateau_requires_contiguous_complete_points(self):
        policy=dict(minimum_epochs=2,patience_points=3,minimum_gain_tasks=2,mode='flag_only')
        curve=[dict(step=i*8,epoch=i/2,metrics={'RO':{'arrival':10}}) for i in range(1,6)]
        self.assertEqual(plateau_status(curve,policy,8)['through_epoch'],2.5)
        self.assertIsNone(plateau_status(curve[:1]+curve[2:],policy,8))
        curve[-1]['metrics']['RO']['arrival']=12
        self.assertIsNone(plateau_status(curve,policy,8))
    def test_collection_uses_configured_cohort_and_epoch_scale(self):
        cfg=dict(data={'samples_per_epoch':24},training={'global_batch':6,'epochs':3,'save_steps':2},
                 baselines={'B0':{'real_map':False}},evaluation={'tasks':3,'probe_tasks':1,'shard_tasks':2,'modes':['TF','RO']})
        with tempfile.TemporaryDirectory() as path:
            for mode in ['TF','RO']:
                for start,stop in [(0,2),(2,3)]:
                    folder=Path(path)/'evaluation/B0/checkpoint-2'/f'{mode}_{start:04d}_{stop:04d}'
                    folder.mkdir(parents=True)
                    (folder/'summary.json').write_text('{}')
                    cases=[dict(trajectory_id=str(i),arrived=i<2,arrived_and_stopped=False,
                        turns=[dict(live=True,rank_eligible=True,safe=True,optimal=False,map_optimal=False)]) for i in range(start,stop)]
                    (folder/'cases.jsonl').write_text('\n'.join(map(json.dumps,cases)))
            collect(path,cfg)
            curve=json.loads((Path(path)/'results/B0/curve.json').read_text())
            self.assertEqual(curve[0]['epoch'],0.5)
            report=(Path(path)/'results/B0/report.md').read_text()
            self.assertIn('2/3（66.67%）',report)
            self.assertIn('| 3 |',report)
            self.assertNotIn('/510',report)

if __name__=='__main__':unittest.main()
