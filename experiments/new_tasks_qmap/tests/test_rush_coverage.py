"""Expanded training boards preserve reference scenes and geometry holdouts."""
from contextlib import ExitStack
import gzip
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from grounded_llm.core.files import atomic_json
from grounded_llm.core.qmap.physical_data import enumerate_component,save_scene
from grounded_llm.tasks.rush_hour.environment import RushHour
from grounded_llm.tasks.rush_hour.relation_data import catalogue_key,build


def board(column,length=2):
    cells=['o']*36;cells[12:14]=['A','A']
    for row in range(length):cells[row*6+column]='B'
    return ''.join(cells)


class CoverageTests(unittest.TestCase):
    def test_large_group_setup_preserves_original_member_order_and_rng(self):
        from grounded_llm.core.qmap.sampling import GroupCoverageSampler
        groups=np.array([3,1,3,9,1,1,9,3])
        left=np.random.default_rng(41);right=np.random.default_rng(41)
        original=[left.permutation(np.flatnonzero(groups==g)) for g in np.unique(groups)]
        sampler=GroupCoverageSampler(groups,right)
        for a,b in zip(original,sampler.orders):np.testing.assert_array_equal(a,b)
        self.assertEqual(left.bit_generator.state,right.bit_generator.state)

    def test_new_catalogues_do_not_count_duplicate_states_or_cross_reference_holds(self):
        with tempfile.TemporaryDirectory() as directory,ExitStack() as stack:
            root=Path(directory);reference=root/'reference/data/graphs';reference.mkdir(parents=True)
            records=[];source=[]
            for i,(text,split) in enumerate([(board(2),'train'),(board(3),'test')]):
                env=RushHour(text);states,index,children=enumerate_component(env,env.start,1000)
                goals=[j for j,s in enumerate(states) if env.terminal(s)][:3]
                metadata=dict(task='rush_hour',split=split,catalogue_group=catalogue_key(env),
                    source=dict(row=i,original_minimum=1),initial_board=text,original_state=0,training_goals=2,symmetric=True)
                record=save_scene(reference/f'scene_{i:03d}',states,children,goals,metadata,symmetric=True)
                records.append(dict(path=f'scene_{i:03d}',**record))
                actions,_=env.solve();source.append(f'{len(actions)} {text} {len(states)}\n')
                # Another placement in the same catalogue must not add a board.
                other=''.join(env.render(states[-1]));source.append(f'1 {other} {len(states)}\n')
            original=json.loads((reference/'scene_001/scene.json').read_text())
            for text in [board(2,3),board(4)]:
                env=RushHour(text);states,_,_=enumerate_component(env,env.start,1000);actions,_=env.solve()
                source.append(f'{len(actions)} {text} {len(states)}\n')
            atomic_json(reference/'corpus.json',dict(scenes=records))
            cache=root/'source.txt.gz'
            with gzip.open(cache,'wt') as stream:stream.writelines(source)
            config=dict(seed=1,data=dict(append_from='../reference/data/graphs',source_cache=str(cache),
                source_url='unused',max_states=1000,minimum_training_catalogues=3,terminal_goals_per_scene=3,
                intermediate_goals_per_scene=2,heldout_goals=1,heldout_intermediate_goals=1,
                layout_strata=[dict(minimum_moves=1,maximum_moves=10,training_catalogues=3)],selection_reserve=2))
            corpus=build(config,root/'expanded/data/graphs')
            for scene in corpus.scenes:
                for field in ('states','children','distances','reverse_distances'):stack.callback(getattr(scene,field)._mmap.close)
            self.assertEqual(len(corpus.scenes),4)
            train={s.metadata['catalogue_group'] for s in corpus.scenes if s.metadata['split']=='train'}
            test={s.metadata['catalogue_group'] for s in corpus.scenes if s.metadata['split']=='test'}
            self.assertEqual(len(train),3);self.assertFalse(train & test)
            self.assertEqual(corpus.scenes[1].metadata,original)
            np.testing.assert_array_equal(corpus.scenes[1].goals,np.load(reference/'scene_001/goal_ids.npy'))


if __name__=='__main__':unittest.main()
