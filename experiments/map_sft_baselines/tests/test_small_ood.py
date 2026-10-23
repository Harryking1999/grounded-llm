import tempfile
import unittest
from pathlib import Path

import numpy as np

from experiments.map_sft_baselines.src.small_ood_prepare import block_goal, restoration_goals
from grounded_llm.tasks.blocks import rules
from grounded_llm.tasks.blocks.protocol import action_key, action_from_key
from grounded_llm.tasks.puzzle8.environment import Puzzle8


class SmallOODTests(unittest.TestCase):
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
