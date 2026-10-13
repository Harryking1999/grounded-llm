import unittest
import random
import tempfile,json
from pathlib import Path
import torch
from torch import nn
from types import SimpleNamespace
from experiments.map_sft_baselines.src.prepare import choose
from experiments.map_sft_baselines.src.modeling import TokenLM
from experiments.map_sft_baselines.src.queue import evaluation_jobs
from experiments.map_sft_baselines.src.evaluate import summarize
from experiments.map_sft_baselines.src.train import validate_resume_contract

class ContractTests(unittest.TestCase):
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
            cfg=dict(baselines={'B0':{'real_map':False}},evaluation={'shard_tasks':510})
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

if __name__=='__main__':unittest.main()
