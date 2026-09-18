import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from experiments.cml_map_scaling.src.step2_augmentation import augmentation_messages, extract_decision, AugmentedInterface, checkpoint_for_condition, pending_items
from experiments.cml_map_scaling.src.step2_augmentation_report import build, write_report


class AugmentationTest(unittest.TestCase):
    def test_resume_keeps_reverse_direction_and_only_missing_items(self):
        items=[{'u':0,'g':1},{'u':1,'g':0},{'u':2,'g':1}]
        self.assertEqual(pending_items(items,[items[0]]),items[1:])
        self.assertEqual(pending_items(items,items),[])
    def test_all_text_is_preserved_and_only_roadmap_block_is_added(self):
        spec = json.loads(Path('experiments/cml_map_scaling/configs/step2_augmentation.json').read_text())
        item = dict(u=12,g=15,task='action',template='canonical')
        baseline = augmentation_messages(spec, '12: 3, 4\n15: 7', item, False)
        augmented = augmentation_messages(spec, '12: 3, 4\n15: 7', item, True)
        block = spec['prompts']['roadmap_block'].format(current_slot='<|cml_current_state|>',goal_slot='<|cml_goal_state|>')
        self.assertEqual(augmented[1]['content'].replace(block,''),baseline[1]['content'])
        self.assertEqual(augmented[0],baseline[0])
        self.assertIn('Current node: 12\nGoal node: 15', augmented[1]['content'])

    def test_prompt_cache_must_include_text_ids_for_augmented_conditions(self):
        class Tokenizer:
            def apply_chat_template(self, messages, **kwargs):
                return [10,20]+list(messages[1]['content'].encode())
        instance = object.__new__(AugmentedInterface)
        instance.config = {'assets': {'node_count': 32}}
        instance.spec = json.loads(Path('experiments/cml_map_scaling/configs/step2_augmentation.json').read_text())
        instance.tokenizer, instance.slot_ids = Tokenizer(), [10,20]
        # Use out-of-byte IDs to avoid collisions with text encoding.
        instance.slot_ids = [300,301]
        instance.tokenizer.apply_chat_template = lambda messages,**kwargs: [300,301]+list(messages[1]['content'].encode())
        instance.adjacency, instance.prompt_cache = '0: 1\n1: 0', {}
        first = instance.prompt_ids(dict(u=0,g=1,task='action',template='canonical'),False)
        second = instance.prompt_ids(dict(u=1,g=0,task='action',template='canonical'),False)
        self.assertNotEqual(first,second)

    def test_prose_and_markdown_choices_are_not_format_failures(self):
        for answer in ('7','Node 7.','This reaches the goal.\nNext node: **7**',
                       'We are at 12, aiming for 15. I choose node 7.',
                       'The next node is 7.', 'Final answer: \\boxed{7}'):
            self.assertEqual(extract_decision(answer)['node'],7,answer)

    def test_does_not_pick_numbers_from_context_or_multiple_options(self):
        for answer in ('Current node: 12. Goal node: 15.', 'Next node: 7 or 8',
                       'Next node: 7\nNext node: 8', 'We could visit node 7 or node 8.'):
            self.assertEqual(extract_decision(answer)['status'],'needs_review',answer)

    def test_action_transfer_reuses_exact_full_text_prompt_and_resolves_existing_weights(self):
        root=Path('experiments/cml_map_scaling')
        original=json.loads((root/'configs/step2_augmentation.json').read_text(encoding='utf-8'))
        transfer=json.loads((root/'configs/step2_action_transfer_eval.json').read_text(encoding='utf-8'))
        self.assertEqual(transfer['prompts'],original['prompts'])
        self.assertEqual(transfer['generation'],original['generation'])
        self.assertEqual(transfer['split'],original['split'])
        self.assertEqual(transfer['tasks'],original['tasks'])
        with TemporaryDirectory() as temporary:
            base=Path(temporary)/'base';continuation=Path(temporary)/'continuation'
            for path in (base/'mlp_report/selected.safetensors',continuation/'report_continuation/final.safetensors',continuation/'report_plus_action/final.safetensors'):
                path.parent.mkdir(parents=True,exist_ok=True);path.touch()
            conditions=transfer['conditions']
            self.assertIsNone(checkpoint_for_condition(conditions[0],transfer,base,continuation))
            self.assertEqual(checkpoint_for_condition(conditions[1],transfer,base,continuation),base/'mlp_report/selected.safetensors')
            self.assertEqual(checkpoint_for_condition(conditions[2],transfer,base,continuation),continuation/'report_continuation/final.safetensors')
            self.assertEqual(checkpoint_for_condition(conditions[3],transfer,base,continuation),continuation/'report_plus_action/final.safetensors')

    def test_report_names_come_from_conditions(self):
        metric={'successes':1,'total':2,'rate':0.5}
        block={'decision_identified':metric,'action_legal':metric,'one_step_success':metric,
               'token_limit':metric,'mean_generated_tokens':2,'success_by_graph_distance':{'1':metric}}
        result={'interpretation':'Matched transfer','conditions':{'text_baseline':block,'text_plus_report_plus_action':block},
                'matched_comparison':{},'failure_categories':{},'manual_review_counts':{},
                'evaluated_records':4,'unresolved_semantic_reviews':0,'source_commit':'abc1234',
                'base_source_commit':'def5678','wall_seconds':60}
        with TemporaryDirectory() as temporary:
            write_report(Path(temporary),result)
            report=(Path(temporary)/'report.md').read_text(encoding='utf-8')
        self.assertIn('| 图距离 | text_baseline | text_plus_report_plus_action |',report)

    def test_scaling_action_eval_only_generates_new_conditions(self):
        root=Path('experiments/cml_map_scaling/configs')
        original=json.loads((root/'step2_augmentation.json').read_text(encoding='utf-8'))
        scaled=json.loads((root/'step2_action_scaling_eval.json').read_text(encoding='utf-8'))
        self.assertEqual(scaled['prompts'],original['prompts'])
        self.assertEqual(scaled['generation'],original['generation'])
        self.assertEqual([c['name'] for c in scaled['conditions']],
                         ['text_plus_report_continuation','text_plus_report_plus_action'])
        self.assertEqual(scaled['reference_condition'],scaled['conditions'][0]['name'])

    def test_merge_ignores_sibling_shard_log_and_accepts_report_reference(self):
        spec=json.loads(Path('experiments/cml_map_scaling/configs/step2_action_scaling_eval.json').read_text())
        with TemporaryDirectory() as temporary:
            root=Path(temporary);base=root/'base';run=root/'eval';shard=run/'shard_0'
            base.mkdir();shard.mkdir(parents=True)
            graph=root/'graph.json';graph.write_text(json.dumps({'neighbors':{'0':[1],'1':[0]}}))
            (base/'config.json').write_text(json.dumps({'assets':{'node_count':2,'graph_reference':str(graph)}}))
            (base/'split.json').write_text(json.dumps({'test':[[0,1]]}))
            (run/'shard_0.log').write_text('A log is not a shard directory.')
            (shard/'config.json').write_text(json.dumps(spec))
            (shard/'runtime.json').write_text(json.dumps({'num_shards':1,'source_commit':'abc1234',
                'started_utc':'2026-09-18T00:00:00Z','completed_utc':'2026-09-18T00:01:00Z'}))
            rows=[{'condition':c['name'],'u':u,'g':g,'task':'action','raw_output':str(g),
                   'decision':{'node':g,'status':'identified','source':'direct_answer'},
                   'generated_token_count':2,'hit_token_limit':False}
                  for c in spec['conditions'] for u,g in ((0,1),(1,0))]
            (shard/'predictions.jsonl').write_text('\n'.join(json.dumps(r) for r in rows))
            result=build(run,base)
        self.assertEqual(result['evaluated_records'],4)
        self.assertEqual(result['unresolved_semantic_reviews'],0)
        self.assertEqual(result['matched_comparison']['text_plus_report_plus_action']['both_success'],2)


if __name__ == '__main__':
    unittest.main()
