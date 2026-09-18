import json
from pathlib import Path
import unittest

from experiments.cml_map_scaling.src.step2_scaling import case_config
from experiments.cml_map_scaling.src.step2_data import make_split
from experiments.cml_map_scaling.src.step2_augmentation import augmentation_messages


class ScalingTest(unittest.TestCase):
    def test_splits_cover_all_nodes_and_keep_pairs_disjoint(self):
        spec=json.loads(Path('experiments/cml_map_scaling/configs/step2_scaling.json').read_text())
        for case in spec['assets']['cases']:
            cfg=case_config(spec,case,'unused.json')
            split=make_split(cfg)
            self.assertEqual(len(split['train']),case['train_pairs'])
            self.assertEqual(len(split['test']),50)
            self.assertEqual({n for pair in split['train'] for n in pair},set(range(case['node_count'])))
            groups=[set(map(tuple,split[name])) for name in ('train','validation','test')]
            self.assertFalse(groups[0]&groups[1] or groups[0]&groups[2] or groups[1]&groups[2])

    def test_large_node_ids_remain_visible_in_both_conditions(self):
        spec=json.loads(Path('experiments/cml_map_scaling/configs/step2_augmentation.json').read_text())
        item=dict(u=255,g=128,task='action',template='canonical')
        for roadmap in (False,True):
            text=augmentation_messages(spec,'255: 128\n128: 255',item,roadmap,256)[1]['content']
            self.assertIn('0 to 255',text)
            self.assertIn('Current node: 255\nGoal node: 128',text)


if __name__=='__main__':
    unittest.main()
