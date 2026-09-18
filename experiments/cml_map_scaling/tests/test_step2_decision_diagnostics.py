import json
from pathlib import Path
import tempfile
import unittest
import torch
from experiments.cml_map_scaling.src.step2_decision_diagnostics import make_items,messages,ChoiceGrammar,diagnostic_checkpoint,reused_items

class Tokenizer:
    eos_token_id=99
    pad_token_id=98
    def encode(self,s,add_special_tokens=False):return [int(x) for x in s]

class DiagnosticsTest(unittest.TestCase):
    def test_continuation_checkpoint_groups_do_not_fall_back_to_original_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);base=root/'base';continuation=root/'continuation'
            for name in ('report_continuation','report_plus_action'):
                (continuation/name).mkdir(parents=True);(continuation/name/'final.safetensors').touch()
            spec=dict(checkpoint='final',continuation_run=str(continuation),checkpoint_groups={
                'report':dict(adapter='mlp',checkpoint_source='continuation',checkpoint_condition='report_continuation'),
                'action':dict(adapter='mlp',checkpoint_source='continuation',checkpoint_condition='report_plus_action')})
            self.assertEqual(diagnostic_checkpoint('action_mismatch',spec,base),('mlp',continuation/'report_plus_action/final.safetensors'))
            self.assertEqual(diagnostic_checkpoint('report_latent',spec,base),('mlp',continuation/'report_continuation/final.safetensors'))
            source=root/'items';source.mkdir()
            for name,value in [('config',dict(node_count=128)),('items',dict(distance_compare=[dict(id='original:1')])),('permutation',[1,0])]:
                (source/(name+'.json')).write_text(json.dumps(value))
            items,permutation=reused_items(source,128)
            self.assertEqual(items,dict(distance_compare=[dict(id='original:1')]))
            self.assertEqual(permutation,[1,0])
    def test_item_labels_and_pairing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            (root/'graph.json').write_text(json.dumps(dict(neighbors={'0':[1,2],'1':[0,2],'2':[0,1,3],'3':[2]})))
            (root/'split.json').write_text(json.dumps(dict(test=[[0,3]])))
            spec=dict(seed=1,action_tasks=['neighbor_direct','neighbor_reason'],probe_tasks=['distance_compare','transition_compare'])
            items,perm=make_items(root,spec)
            self.assertEqual(len(items['neighbor_direct']),2)
            self.assertEqual(items['neighbor_direct'][0]['answers'],['2'])
            self.assertEqual(items['neighbor_direct'][0]['options'],['1','2'])
            self.assertEqual(len(items['distance_compare']),4)
            self.assertNotEqual(items['distance_compare'][0]['answers'],items['distance_compare'][1]['answers'])
            self.assertEqual({x['answers'][0] for x in items['transition_compare']},{'更近','相同'})
            self.assertTrue(all(i!=v for i,v in enumerate(perm)))
    def test_grammar_is_per_example_and_requires_complete_label(self):
        g=ChoiceGrammar(Tokenizer(),[['1','12'],['2']])
        self.assertEqual(g(0,torch.tensor([])),[1])
        self.assertEqual(g(1,torch.tensor([])),[2])
        self.assertEqual(g(0,torch.tensor([1])),[2,99])
        self.assertEqual(g(0,torch.tensor([1,2])),[99])
        self.assertEqual(g(0,torch.tensor([1,2,99])),[98])
    def test_mismatch_does_not_change_text_or_expose_answer(self):
        item=dict(task='distance_compare',a=7,b=9,g=3,options=['A','B'],answers=['A'])
        good=messages(item,'linear_correct','graph')
        self.assertEqual(good,messages(item,'linear_mismatch','graph'))
        latent=messages(item,'linear_latent','graph')[1]['content']
        self.assertNotIn('节点编号',latent);self.assertNotIn('graph',latent)
        self.assertEqual(latent.count('<|diag_'),3)

if __name__=='__main__':unittest.main()
