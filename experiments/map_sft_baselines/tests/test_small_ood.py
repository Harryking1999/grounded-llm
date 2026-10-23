import tempfile
import unittest
import json
from unittest.mock import patch
from pathlib import Path

import numpy as np

from experiments.map_sft_baselines.src.small_ood_prepare import block_goal, restoration_goals, prepare
from grounded_llm.tasks.blocks import rules
from grounded_llm.tasks.blocks.protocol import action_key, action_from_key
from grounded_llm.tasks.puzzle8.environment import Puzzle8


class SmallOODTests(unittest.TestCase):
    def test_candidate_map_is_attached_only_to_cube_small_ood(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); source = root/'source'; out = root/'out'
            for task in ('cube2', 'path'):
                (source/task).mkdir(parents=True)
                (source/task/'evaluation.json').write_text(json.dumps([
                    dict(trajectory_id=task, start=2, goal=0, reference_actions=['U', 'R'])]))
                (source/task/'qmap.pt').write_bytes(b'original')
            candidate = root/'candidate.pt'; candidate.write_bytes(b'2000')
            identity = dict(run='cube_near_low_lr_ed21cd4', step=2000)
            config = dict(seed=1, tasks_per_cohort=1, original_tasks=['cube2', 'path'],
                small_ood={'cube2': dict(goal_distances_from_solved=[1,2,3],
                    map_checkpoint='qmap_near_goal_2000.pt', map_identity=identity)},
                evaluation={'tasks': 3})
            with self.assertRaisesRegex(ValueError, 'cube-small-ood-checkpoint'):
                prepare(config, source, out)
            self.assertFalse(out.exists())
            with patch('experiments.map_sft_baselines.src.small_ood_prepare.restoration_goals',
                       return_value=[dict(trajectory_id='cube2', start=2, goal=1,
                           reference_actions=['U'], goal_distance_from_solved=1)]):
                prepare(config, source, out, candidate)
            rows = json.loads((out/'evaluation.json').read_text())
            self.assertEqual([r['trajectory_id'] for r in rows], ['cube2','cube2_small_ood','path'])
            self.assertNotIn('map_checkpoint', rows[0])
            self.assertNotIn('map_checkpoint', rows[2])
            self.assertEqual(rows[1]['map_checkpoint'], 'qmap_near_goal_2000.pt')
            self.assertEqual(rows[1]['map_identity'], identity)
            self.assertEqual((out/'cube2'/'qmap_near_goal_2000.pt').read_bytes(), b'2000')
            for task in ('cube2', 'path'):
                self.assertEqual((out/task/'qmap.pt').read_bytes(), b'original')

    def test_block_keeps_exactly_last_removable_shape(self):
        first=rules.BY_ACTION[(4,0,0)];last=rules.BY_ACTION[(5,8,7)]
        masks=[rules.PLACEMENTS[i][0] for i in [first,last]]
        actions=[action_key(rules.PLACEMENTS[i][1]) for i in [first,last]]
        row=dict(start=str(masks[0]|masks[1]),goal='0',reference_actions=actions)
        result=block_goal(row)
        self.assertEqual(int(result['goal']),masks[1])
        self.assertEqual(rules.apply(int(row['start']),action_from_key(result['reference_actions'][0])),masks[1])

    def test_puzzle_new_targets_and_reference_routes_are_exact_and_paired(self):
        env=Puzzle8()
        records=[dict(start=s,goal=0,trajectory_id=str(s)) for s in range(100,106)]
        with tempfile.TemporaryDirectory() as folder:
            rows=restoration_goals('puzzle8',records,Path(folder),[1,2,3],np.random.default_rng(1))
        self.assertEqual([r['start'] for r in rows],[r['start'] for r in records])
        env.prepare_goals([0,*sorted({r['goal'] for r in rows})])
        self.assertEqual(sorted(r['goal_distance_from_solved'] for r in rows),[1,1,2,2,3,3])
        for row in rows:
            state=row['start']
            self.assertEqual(len(row['reference_actions']),int(env.distance(state,row['goal'])))
            self.assertEqual(int(env.distance(row['goal'],0)),row['goal_distance_from_solved'])
            for action in row['reference_actions']:
                state=env.step(state,action)
            self.assertEqual(state,row['goal'])


if __name__=='__main__':unittest.main()
